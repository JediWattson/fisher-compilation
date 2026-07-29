"""Frozen one-shot shadow protocol for the Gemma L3/L4 partial SVD edge.

This module is deliberately prompt blind.  It binds the already-verified
compiler artifacts and structured-strong-v9 corpus files by SHA-256, but it
does not open, parse, tokenize, or materialize any prompt role.

The private streaming evaluator is composed only by the fused host-global
one-shot transaction.  Native sequential-refit logits remain authoritative.
The candidate path uses the source-model clamped-Y3 reference as an oracle
fallback, so a passing report authorizes only this partial-edge shadow scope;
it never authorizes a standalone or full-model deployment claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor

from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    SourceAuthoritativeShadowFidelityAccumulator,
    ShadowFidelityExample,
    ShadowFidelityGates,
)


ShadowArm = Literal["all_on"]

__all__ = [
    "Gemma3L3L4GraphOrganizedSVDShadowObservation",
    "Gemma3L3L4GraphOrganizedSVDShadowProtocol",
    "default_gemma3_l3_l4_graph_organized_svd_shadow_protocol",
    "derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt",
    "derive_gemma3_l3_l4_graph_organized_svd_shadow_masks",
    "frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest",
    "gemma3_l3_l4_graph_organized_svd_evidence_payload_sha256",
    "gemma3_l3_l4_graph_organized_svd_model_inputs_sha256",
    "gemma3_l3_l4_graph_organized_svd_prompt_sha256",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol"
_FORMAT_VERSION = 1
_PROTOCOL_DOMAIN = b"fisher-graph:gemma3-l3-l4-svd-shadow-protocol:v1\0"
_OBSERVATION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-observation:v1\0"
)
_MANIFEST_DOMAIN = b"fisher-graph:gemma3-l3-l4-shadow-manifest:v1\0"
_ASSESSMENT_CLAIM_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-assessment-claim:v1\0"
)
_RUNTIME_BINDING_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-runtime-binding:v1\0"
)
_INPUT_PROVENANCE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-input-provenance:v1\0"
)
_FIVE_PASS_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-five-pass-receipt:v1\0"
)
_EVIDENCE_PAYLOAD_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-evidence-payload:v1\0"
)
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l3-l4-shadow-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARMS = frozenset({"all_on"})

_MODEL_ID = "google/gemma-3-270m"
_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_TOKENIZER_CLASS = (
    "transformers.models.gemma.tokenization_gemma.GemmaTokenizer"
)
_TOKENIZER_CONFIGURATION_SHA256 = (
    "b02c42b40d0c95c70024c617c8774cde360991e2c949de1d35b51288ded31372"
)
_TOKENIZER_BACKEND_SERIALIZED_BYTES = 14_386_244
_TOKENIZER_BACKEND_SERIALIZED_SHA256 = (
    "c1a087240686a7d141101217051f76d5cd4cbe2b6093e3c3553fb26dcc4d0e9a"
)
_TOKENIZER_POST_TOKENIZATION_BACKEND_SERIALIZED_BYTES = 14_386_431
_TOKENIZER_POST_TOKENIZATION_BACKEND_SERIALIZED_SHA256 = (
    "09afbc35a2fa856bf2baf6f3d140ac7ccddb97179b616b797b5688a96763c189"
)
_TOKENIZER_CANONICAL_VOCAB_COUNT = 262_145
_TOKENIZER_CANONICAL_VOCAB_SHA256 = (
    "8a2dcfa056d1a48a1cfcb752524bf3a19ff7c996c4f5d4625ad331ca5e0b6eb1"
)
_TOKENIZER_ADDED_TOKEN_COUNT = 6_415
_TOKENIZER_ADDED_TOKENS_SHA256 = (
    "7e24459f9c42fe138dfc7ee71cf68a1b1e3b8098d18690c4c28645bed3a5360d"
)
_TOKENIZER_SPECIAL_TOKENS_MAP_SHA256 = (
    "a237afa0a3964f4db32b59e1031adcc948cf01b552badfdf5a96092422a19884"
)
_TOKENIZER_TRANSFORMERS_VERSION = "5.14.1"
_TOKENIZER_TOKENIZERS_VERSION = "0.22.2"
_TOKENIZER_SENTENCEPIECE_VERSION = "0.2.2"
_SOURCE_MODEL_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_GRAPH_CANDIDATE_FILE_SHA256 = (
    "d77a60532b660160413331ceddbe8d970c2828d53ff5788642250ff3c5d49fa1"
)
_GRAPH_CANDIDATE_ARTIFACT_SHA256 = (
    "b3e011d8067ff3538888851c476fba03c57f4e9f172f923c20fdd90ac0799f84"
)
_FACTORIZED_LIVE_EXECUTION_SHA256 = (
    "ead03074b87898c9e6c5b068b738420ab0dcf178f07603e885a71964b94ebb7a"
)
_FACTORIZED_REFIT_EXECUTION_SHA256 = (
    "911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9"
)
_GLOBAL_SVD_BASE_PLAN_SHA256 = (
    "8e34336c34f07a75bffa6b9e57618183a9129024b85c9c0cd4fbf3b23fca6b5f"
)
_GRAPH_BASIS_ARTIFACT_SHA256 = (
    "855a047ef20ca3e11a105d7d62752575381ce6eeccd7d12bf98b72dc43067730"
)
_SIGNED_GFA_PLAN_SHA256 = (
    "10299119071f215c97979edf6b02bec4e7e7cde5a6d2c316a662b802a84aa469"
)
_BASIS_FILE_SHA256 = (
    "359c9659358cbaf97232848a10bdf0e2261d95820ad5effda9bdafeead6a7605"
)
_BASIS_PAYLOAD_SHA256 = (
    "b2217153911436673f2ff7475c658c928112e802f5999619393287d2b0803c01"
)
_HIERARCHY_ARTIFACT_SHA256 = (
    "2e35cbd0e54a1db4b483f11ebbc3b1f9cdd3472d55aac5f1a046c434477b08ff"
)
_BASE_ARTIFACT_FILE_SHA256 = (
    "272af3fa25992aefa6fa12b3aae6da64c836be0a3d616a1780444bde0caf95d3"
)
_BASE_SCIENTIFIC_PAYLOAD_SHA256 = (
    "babed58e93ff09bd65a7ce0062eb8e1f657672f3cc8bcf4e9fb03f446a48f5ec"
)
_REFIT_ARTIFACT_FILE_SHA256 = (
    "48b51412ff552950bad3794833d9313e4f384231bcdb3590e460a28f10aff655"
)
_REFIT_SCIENTIFIC_PAYLOAD_SHA256 = (
    "4c04775672f74204d1d3176c05780e97107f3edf57fac54d35b1c06ab96d36b1"
)
_CORPUS_ID = "structured-strong-v9"
_PROMPT_FILE_SHA256 = (
    "d03a514287afd6b607f4307db58092edfa1c75c2927779b01595eaa0ca106c07"
)
_FAMILY_FILE_SHA256 = (
    "db8dfafb23c249067f18aaf413b0fe077f4dfc132be666469568e633693d21dd"
)
_AUDIT_FILE_SHA256 = (
    "ff11401b61562e02854654657a7c9a46470032e99c43d7697e6dfe1ef536df52"
)
_CALIBRATION_B_MANIFEST_SHA256 = (
    "986ee9da505fb056853f4fc7ed4f5eee6e9313f0419f2ca9ebc54e0df8607bdd"
)
_CALIBRATION_B_EXAMPLE_COUNT = 96
_CALIBRATION_B_FAMILY_COUNT = 8
_SOURCE_ORIGIN_MIN = 8
_SOURCE_ORIGIN_MAX = 40
_LAG_COUNT = 32
_TARGET_MODAL_WIDTH = 64
_TARGET_FULL_WIDTH = 640
_GEMMA_VOCAB_SIZE = 262_144
_BOUNDARY_RELATIVE_ERROR_MAX = 0.25
_BOUNDARY_COSINE_MIN = 0.95
_VALID_TARGET_COVERAGE_MIN = 0.80
_PROJECTION_RELATIVE_ERROR_MAX = 0.05
_PROJECTION_COSINE_MIN = 0.995
_WORST_FAMILY_MODAL_RELATIVE_ERROR_MAX = 0.35
_WORST_FAMILY_MODAL_COSINE_MIN = 0.90
_WORST_FAMILY_PROJECTION_RELATIVE_ERROR_MAX = 0.10
_WORST_FAMILY_PROJECTION_COSINE_MIN = 0.99
_MINIMUM_FAMILY_SIGNAL_L2_NORM = 1e-12

_CALIBRATION_B_EXAMPLE_SHA256S = (
    "02ef6e47edc2ee9925ec58436fa43c51b9debf88d8544b8f1de4fd59d45887a0",
    "01a30c7e2e23933aa82c498de10acf5a59ae8e150c4c247d36b3a03ccf29b351",
    "7e0d8b7105e2e5d768e5685f1048d37ade2820b2a811d8330be98d16aafe4bfd",
    "053211b93b62d33ae744074622b478a84b7ef856f61a7ec47d07cf679f9787df",
    "637f02eaf22cbb1ad8ccbd09b91b240923ca935dda36aa156741cbab6f176255",
    "dc46cff6019572f0f09a5a6c0ae463f5f9ebc4527746d2b0a8828704add4e1e0",
    "ad200950116e633e1e30a87be11136f11dbfa8b4ae44e493782f14722fa523ac",
    "34c53b1c5f6ea3f89c989b6d83302872dbd19ec6a86483a2e2dd16d20b619c01",
    "90ab0258b6f420e4cecee72079b2e51baf896ef0bedda0d1ce03c60d7d2b7ff0",
    "78242ff6ebfcd5fb77b249d27d497a42e6f49db379628721e5e022fc248b2338",
    "53c8a7b4a4c8dd8c97faebe3e4191873d9add324ec4322477222e9509b7ae4ad",
    "369a2e84e3f649fbc18fcfef20ccc8acdb1d0373e474dae85c9732eddc92e59c",
    "e3f6a8f291b14c6aa3b65262b2d73bd4404c970a2566f6e842f60b93037e8fc7",
    "1d94b08c0f62535d9891f261100279597e9d576751f18da51c528e030b48c68e",
    "b72a078891399946be2c92809ec1721d32b720b4d9f1287fbd1f06e39bf23adb",
    "dbb60006157dfa2191468f3ee2222c14757fa9347a61071481130e26f7beaa91",
    "c418315a5960cd0404541f55ec24e6f0c6ad1efc4c5784b5d58fcb6cd2a4d7d0",
    "9be1c4ddbb8236a1ee27f6a0b3354b24e1301ff9d9127c9ec129e3b678f1df99",
    "b3683e772818e7cd59f38d42ff7df5b31e16646f981495a76c6613b2e1545367",
    "2ae1259543ba5a85cd15659dd8525599de6d21bbb866d753bbe332df9939ba4e",
    "a66d9c9d60c7e1edc4baccfecbd19b3836c9501f0388b307b4b92b5afb47f77a",
    "b682edf5de7b09500e0b1b0c9c0c5275e242197bfb412aa3102d8029459e13d5",
    "8f701b900db2d58109554dba1a073d51f3d90abae57b9346ddabe61deed1f770",
    "91b239ff1001a4b749d37a60ca4f8210dcee1d14b237b02771b4c76343d2176f",
    "ffca0c90a6732b22d35459cb724f1cf386258060a28d5b351f533607365851bf",
    "27764f147ab4cd0604a9d7813cfe76712a99bea92b482486848f75b92640a4c0",
    "0eb8ac0cc933a14c6244f226f1e49649afc97c428e0416ccb0b3d96a1a7a4764",
    "edc472d79fa5ee4382f247520daa20d297ed8d8fa482250f7c5caf17d295ea22",
    "ed66fade2103c76c86655e452d8aa5ce1c20b2df027a0c089ada31617b00336b",
    "a45a78011c8dbd65b9c6a411ce262ab30c1241ca47a7dd321258728329a4acd0",
    "3bf8807a5f357a60c3514ad77d900a4ed68885781af8e6fe72417e5a48dd7908",
    "f98adbd2c1a8ba22f27c789f13e51fa2b4a773520f810d7457919d767c154611",
    "abe1506d353599af04936917c18eb2a4151b3d76091c2e303f28815fc298f1a7",
    "3227f4e5238f074e9b2ef89e7d8bfa3b8f43d43d3d41e706736c5472d5bc828c",
    "dd5201fa5f5426a87b5808bf4040669c91961e6501fa64396d33d8f46dda20ba",
    "cf23ecb8b94168366faf80166c498010e218007fd2b75f5f0fc639346e11796a",
    "337ba5d12d86aec1a80b5a024eadf017e46a9bb9ea6a0525d4c0919748fd20a0",
    "4ec45350294567ae9f68bbceec5c97cdc9779fb5945cddecf6e8333e6f9770d6",
    "04e179c0c72f932719c3eb96e269b215ae4e23342a82c5cb755d2ba6b497a0e7",
    "190ccbdb65b7ff08a9d429297abc97a187326de07195ca439f69653b0932df7d",
    "3a8b4982fb74b222cc9bddb734aaa222726b47143d3682a11a2d7cc7d6a2a5ae",
    "744b68e41262357f3ec908029ea29ae5e8fa832c1bc152f9316d3d149a76ccf8",
    "6d91f96541f7e379ee346b244ec49c06407525e011149857e9eee293418ab580",
    "9f7dbc5151934f7378ee357626ddd1ddbf1214771abe242eadc2dd5dca395e22",
    "8414dd1a5da67bdd04e55e0f7ce846078a5b200f3a3f08181faf68156ad443dd",
    "4366c46a9a0c96ab6694601dcfe458cfe9364edf75e0ac80bf73872ef99f22a3",
    "fc2121e180b3743a8e16b36c4c543c202aded47a7663e3ffb12e56aa4a8f7b22",
    "1516e33519aeb56796fa48a2fcb984dbbad76d3b2b793d991560468e72ba5be5",
    "380ef66874db25b6e1c7ad3dcdcb2d366bd11b2937755167674d0df4cc104cc0",
    "09ecf783ba4b46a0b717556cac89a320593aaefc667aaa834ad41f39ae409f01",
    "788ecad3c9f1000d57f515d2ddad4e2ddf0cb0a9664b115840c24cef78eba146",
    "0d81c88af70c35da2486e8d836addbbe364a477d6d595d7b151f4acc7668e59f",
    "ea517519df4bfbd763797bd8804e525d88c47325a40064ecd7a24411922926ce",
    "a5e237bfe85f0c15e3eeb6290a1d3b9abaf177e594b3352e499908948349af29",
    "98be62a86d8cb3b99a95dc14e6806add62d230544f7efe613037cef15c40ed53",
    "802f3ca04a568ca105d50f63868781c9a871632a2d593bda558e80450bc55a4d",
    "d09c8e8cbcf7c4b7ed8cea0f045bc1b41627fcc9ef1f7ab19f45d884de6ea736",
    "f5beffcabe13442619b83db720ae97f5e6b0f06d1060e1c0f5a843dc77065182",
    "f12433e1961cf629f47efb90de6ef4339f2b508bc2e7d6d05931df5813ff001b",
    "509998bdebee1ab0ea42d4d2f4ee378af5d6487968938610fdfd237c9dfd3be6",
    "919fa77ce6f481da7b0d64797c72b0a4061a692d334acb7758ce48884448e683",
    "c13fd3164d2e998572b46fbc127c5eda44e5afbb3e93d1ec843857e029e9961d",
    "7058d1183410ec66b8b8d1214bc6cfc49d8ef0e1cbf3acf26f6d250177a18464",
    "8aae2490516be8fb85af38ee702dd839b26bfb34d2b4e9630def801825cc6555",
    "f83f58009ca3baa97182fe3c18f47c148543297ad536d3090da6a8330ab4b823",
    "108b36d3b0f182097ee92e3ef457ba7271256a20ebbf55afee81039af6077d88",
    "6dec5c225ed5d7364ecffcf16d27b58f55d90e3a49b0929608d3b9f4c8c974c0",
    "88686a284cb31b84490867a1d94271f6ffba41d01ac805c3c3742c26c2afda49",
    "47c872a895f3f87deda9583228a2f1742003ba3ddd975f5dcc8b30611da4da23",
    "026cd6a58445869bc25b95d7fe30e1b4936d0292b8a0198035ef09f4880c7fe0",
    "1c496b32def8d17c904b113716b76b0c3fcb421360891faf21dc55efc47f2857",
    "9297faa08da7c6587d1d18a9f39b7bf88df2a5451696ecc67dd31da21f70b709",
    "1017b5540fa0ca2af60dbd9aaff0d7a3823569e2d4996935c32c6d3683d19143",
    "8c5390d78f4ce28476887370fd7ef86b76995af95e6da083c4059185e3214831",
    "0643ebc7132d87a51a51dcead9e36a43462cf841ca3b073b4298469df64751b2",
    "eb701b38675843f80b2cd6e8b34a1316eba4ddd2d8cd2e262ae10a9e43e78e5f",
    "d7853f0ce95f479ab6aa851b663c95b28ca0394dd136d3f97da68067205d9a18",
    "cebdd5c34c032e0ee30154f7df98fe43394b9136df4246c5343669ea67ad5aab",
    "d8d5752c729186667b8999bea19d8681ed7eb112d5ccdb19f38b90ce0d439b98",
    "3c8a415e3fb224daaaeb6fc3072d57480bbd7c369d5345ed300172672d11bd66",
    "d60f57b530d6565f6a22ffb628b9dcadc8ed3af31b2fc3c51a0eaee017adb479",
    "f7fee37f1c69267f4f543557cec92fffc2fbf19e853dd828b6e19e8334b99e66",
    "35b001ad8574b7bdce871723503b7534325a4d7eb2d7cf8c4805163b34e5d110",
    "5237da4eed42e85f7aff0ef7d9de529cb72fd0bb9e18bccc9cc3daeb67f70d51",
    "5ab95a9bfd9c158820ae9715f0e346cc29c7eeb707de094265c10e5cf2f9a28a",
    "f1a78fbb7c5a014e0abdcd5af6ba25961da00f08016e9e2a57c0147711016738",
    "90b989cf3aa978bb431960869f41f54e2cbff29c1b838d3dcc4989e0310d95bd",
    "af97a2a97cc561a4cf99e48f2956a325eb44434108574fb291080cfc48749d09",
    "aa366842836a1157b79f2d32aad0ec826feef0126aada935027d5b7130792c2c",
    "b5db4cd825c83db87d3e86ec21f952dea48de99360df024f90fbe24866f82667",
    "18db0ce03c20230f7bfedcedce7d37292cbcc2bf0d98760a9d59758839c42f26",
    "518aadfce9face07807536effb88d9217861f4c2b4f4805bde0ce1bef6f01cc6",
    "9a4b1c8a3063bffdea9462f5687cee5b1b54e4623cae74ce86cd3bdd52cf1125",
    "f2dc0cac6aadc8fb089ac18382ec475bb12864cead339aa4e945cdaefb94244d",
    "8363684566104acacaaf6c2f7e4d66534661223d126595d3a8cebc969bddd7ba",
    "0b1050722a495081ebdf7dc84931af8947757261871cee55cbe54a4089a7c0b9",
)

_CALIBRATION_B_FAMILY_CYCLE = (
    "structured-strong-v9-calibration_b-diatom-valve-outline-v9",
    "structured-strong-v9-calibration_b-harp-string-creep-v9",
    "structured-strong-v9-calibration_b-salt-pan-crust-growth-v9",
    "structured-strong-v9-calibration_b-orchid-pollinium-transfer-v9",
    "structured-strong-v9-calibration_b-river-ice-frazil-v9",
    "structured-strong-v9-calibration_b-vellum-cockling-map-v9",
    "structured-strong-v9-calibration_b-lava-tube-airflow-v9",
    "structured-strong-v9-calibration_b-seed-bank-germination-v9",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _arm(value: object) -> ShadowArm:
    if value not in _ARMS:
        raise ValueError("locked one-shot shadow arm must be all_on")
    return value  # type: ignore[return-value]


def _float_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
    require_finite: bool,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if (
        result.ndim != ndim
        or any(int(width) <= 0 for width in result.shape)
        or (require_finite and not bool(torch.isfinite(result).all()))
    ):
        raise ValueError(f"{label} has invalid geometry or values")
    return result


def _int_tensor(value: object, *, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"{label} must be an integer Tensor")
    result = value.detach().to(device="cpu", dtype=torch.int64)
    result = result.contiguous().clone()
    if result.ndim != 1 or result.numel() <= 0:
        raise ValueError(f"{label} must be nonempty rank-1 data")
    return result


def _bool_tensor(value: object, *, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{label} must be a boolean Tensor")
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.ndim != 1 or result.numel() <= 0:
        raise ValueError(f"{label} must be nonempty rank-1 data")
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        str(tuple(int(width) for width in tensor.shape)).encode("ascii")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def gemma3_l3_l4_graph_organized_svd_prompt_sha256(
    prompt: str | bytes,
) -> str:
    """Derive the frozen prompt identity without exposing it to evaluation.

    The corpus precedent hashes the canonical JSON encoding of a one-element
    prompt list, rather than hashing raw UTF-8 bytes directly.
    """

    if isinstance(prompt, bytes):
        try:
            text = prompt.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("prompt bytes must be strict UTF-8") from error
    elif isinstance(prompt, str):
        text = prompt
    else:
        raise TypeError("prompt must be a string or strict UTF-8 bytes")
    serialized = json.dumps(
        [text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def gemma3_l3_l4_graph_organized_svd_model_inputs_sha256(
    model_inputs: Mapping[str, Tensor],
) -> str:
    """Delegate to the runtime's single model-input identity ABI."""

    return gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)


def derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt(
    *,
    protocol_sha256: str,
    assessment_claim_sha256: str,
    runtime_binding_sha256: str,
    example_id: str,
    family_id: str,
    prompt_identity_sha256: str,
    model_inputs_sha256: str,
    shadow_result_artifact_sha256: str,
    execution_grid_sha256: str,
    projection_oracle_artifact_sha256: str,
    projection_injected_x4_sha256: str,
    carrier_oracle_artifact_sha256: str,
    carrier_injected_x4_sha256: str,
    evidence_payload_sha256: str,
    shadow_model_forward_count: int,
    projection_oracle_model_forward_count: int,
    carrier_oracle_model_forward_count: int,
    projection_oracle_role: str,
    carrier_oracle_role: str,
) -> dict[str, str]:
    """Bind one prompt, one runtime input, and the exact ``3 + 1 + 1`` run."""

    identities = {
        "protocol_sha256": _require_sha256(
            protocol_sha256,
            label="receipt protocol",
        ),
        "assessment_claim_sha256": _require_sha256(
            assessment_claim_sha256,
            label="receipt assessment claim",
        ),
        "runtime_binding_sha256": _require_sha256(
            runtime_binding_sha256,
            label="receipt runtime binding",
        ),
        "example_id": _require_sha256(
            example_id,
            label="receipt example_id",
        ),
        "prompt_identity_sha256": _require_sha256(
            prompt_identity_sha256,
            label="receipt prompt identity",
        ),
        "model_inputs_sha256": _require_sha256(
            model_inputs_sha256,
            label="receipt model inputs",
        ),
        "shadow_result_artifact_sha256": _require_sha256(
            shadow_result_artifact_sha256,
            label="receipt shadow result",
        ),
        "execution_grid_sha256": _require_sha256(
            execution_grid_sha256,
            label="receipt execution grid",
        ),
        "projection_oracle_artifact_sha256": _require_sha256(
            projection_oracle_artifact_sha256,
            label="receipt projection oracle",
        ),
        "projection_injected_x4_sha256": _require_sha256(
            projection_injected_x4_sha256,
            label="receipt projection injection",
        ),
        "carrier_oracle_artifact_sha256": _require_sha256(
            carrier_oracle_artifact_sha256,
            label="receipt carrier oracle",
        ),
        "carrier_injected_x4_sha256": _require_sha256(
            carrier_injected_x4_sha256,
            label="receipt carrier injection",
        ),
        "evidence_payload_sha256": _require_sha256(
            evidence_payload_sha256,
            label="receipt evidence payload",
        ),
    }
    family = _identifier(family_id, label="receipt family_id")
    if identities["prompt_identity_sha256"] != identities["example_id"]:
        raise ValueError(
            "prompt identity SHA-256 must equal the frozen example identity"
        )
    if (
        type(shadow_model_forward_count) is not int
        or shadow_model_forward_count != 3
        or type(projection_oracle_model_forward_count) is not int
        or projection_oracle_model_forward_count != 1
        or type(carrier_oracle_model_forward_count) is not int
        or carrier_oracle_model_forward_count != 1
        or projection_oracle_role != "projection_64"
        or carrier_oracle_role != "exact_x4_carrier"
    ):
        raise ValueError(
            "five-pass receipt requires all_on 3 + projection 1 + carrier 1"
        )
    input_payload = {
        "schema": f"{_SCHEMA}.input_provenance",
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": identities["protocol_sha256"],
        "assessment_claim_sha256": identities["assessment_claim_sha256"],
        "runtime_binding_sha256": identities["runtime_binding_sha256"],
        "example_id": identities["example_id"],
        "family_id": family,
        "prompt_identity": (
            "sha256_canonical_json_single_prompt_utf8"
        ),
        "prompt_identity_sha256": identities["prompt_identity_sha256"],
        "model_inputs_sha256": identities["model_inputs_sha256"],
        "execution_grid_sha256": identities["execution_grid_sha256"],
    }
    input_provenance_sha256 = _json_sha256(
        input_payload,
        domain=_INPUT_PROVENANCE_DOMAIN,
    )
    five_pass_payload = {
        "schema": f"{_SCHEMA}.five_pass_receipt",
        "format_version": _FORMAT_VERSION,
        **identities,
        "family_id": family,
        "input_provenance_sha256": input_provenance_sha256,
        "arm": "all_on",
        "shadow_model_forward_count": shadow_model_forward_count,
        "projection_oracle_model_forward_count": (
            projection_oracle_model_forward_count
        ),
        "carrier_oracle_model_forward_count": (
            carrier_oracle_model_forward_count
        ),
        "total_model_forward_count": (
            shadow_model_forward_count
            + projection_oracle_model_forward_count
            + carrier_oracle_model_forward_count
        ),
        "projection_oracle_role": projection_oracle_role,
        "carrier_oracle_role": carrier_oracle_role,
        "candidate_outputs_metrics_only": True,
        "routing_enabled": False,
    }
    return {
        "input_provenance_sha256": input_provenance_sha256,
        "five_pass_receipt_sha256": _json_sha256(
            five_pass_payload,
            domain=_FIVE_PASS_RECEIPT_DOMAIN,
        ),
    }


def frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest(
) -> dict[str, str]:
    """Return the prompt-blind frozen Calibration-B family manifest."""

    if (
        len(_CALIBRATION_B_EXAMPLE_SHA256S)
        != _CALIBRATION_B_EXAMPLE_COUNT
        or len(set(_CALIBRATION_B_EXAMPLE_SHA256S))
        != _CALIBRATION_B_EXAMPLE_COUNT
        or len(_CALIBRATION_B_FAMILY_CYCLE)
        != _CALIBRATION_B_FAMILY_COUNT
    ):
        raise RuntimeError("frozen Calibration-B identity is malformed")
    result = {
        example_id: _CALIBRATION_B_FAMILY_CYCLE[
            index % _CALIBRATION_B_FAMILY_COUNT
        ]
        for index, example_id in enumerate(
            _CALIBRATION_B_EXAMPLE_SHA256S
        )
    }
    logical = tuple(sorted(result.items()))
    if (
        _json_sha256(logical, domain=_MANIFEST_DOMAIN)
        != _CALIBRATION_B_MANIFEST_SHA256
    ):
        raise RuntimeError("frozen Calibration-B manifest hash drifted")
    return result


def derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
    logical_positions: Tensor,
    valid_mask: Tensor,
    supervised_boundary_indices: Tensor,
) -> dict[str, Tensor]:
    """Derive source, target, and supervised masks from frozen causality."""

    positions = _int_tensor(
        logical_positions,
        label="logical_positions",
    )
    valid = _bool_tensor(valid_mask, label="valid_mask")
    boundaries = _int_tensor(
        supervised_boundary_indices,
        label="supervised_boundary_indices",
    )
    if positions.numel() != valid.numel():
        raise ValueError("logical positions and valid mask must align")
    valid_positions = positions[valid]
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if (
        valid_positions.numel() == 0
        or bool((valid_positions < 0).any())
        or (
            valid_positions.numel() > 1
            and not bool(
                torch.all(
                    valid_positions[1:] == valid_positions[:-1] + 1
                )
            )
        )
    ):
        raise ValueError(
            "valid logical positions must be nonnegative and contiguous"
        )
    if (
        int(valid_indices[-1]) - int(valid_indices[0]) + 1
        != valid_indices.numel()
    ):
        raise ValueError("valid mask must contain one contiguous span")
    expected_boundaries = valid_indices[:-1]
    if not torch.equal(boundaries, expected_boundaries):
        raise ValueError(
            "supervised boundary indices must equal every adjacent valid "
            "next-token boundary"
        )
    source_eligible = (
        valid
        & (positions >= _SOURCE_ORIGIN_MIN)
        & (positions <= _SOURCE_ORIGIN_MAX)
    )
    target_affected = torch.zeros_like(valid)
    source_positions = positions[source_eligible]
    if source_positions.numel():
        target_indices = torch.nonzero(valid, as_tuple=False).flatten()
        target_positions = positions.index_select(0, target_indices)
        lags = target_positions.unsqueeze(1) - source_positions.unsqueeze(0)
        target_affected[target_indices] = (
            (lags >= 0) & (lags < _LAG_COUNT)
        ).any(dim=1)
    affected_supervised = target_affected.index_select(0, boundaries)
    if not bool(affected_supervised.any()):
        raise ValueError(
            "each observation must contain an affected supervised token"
        )
    return {
        "source_eligible_mask": source_eligible,
        "target_affected_mask": target_affected,
        "affected_supervised_mask": affected_supervised,
    }


def _assessment_claim_payload(protocol_sha256: str) -> dict[str, object]:
    return {
        "schema": f"{_SCHEMA}.calibration_b_assessment_claim",
        "format_version": _FORMAT_VERSION,
        "role": "calibration_b_one_shot",
        "protocol_sha256": _require_sha256(
            protocol_sha256,
            label="assessment protocol",
        ),
        "manifest_sha256": _CALIBRATION_B_MANIFEST_SHA256,
        "example_count": _CALIBRATION_B_EXAMPLE_COUNT,
        "family_count": _CALIBRATION_B_FAMILY_COUNT,
        "source_model_sha256": _SOURCE_MODEL_SHA256,
        "tokenizer": _tokenizer_contract_payload(),
        "graph_candidate_tensor_file_sha256": (
            _GRAPH_CANDIDATE_FILE_SHA256
        ),
        "graph_candidate_artifact_sha256": (
            _GRAPH_CANDIDATE_ARTIFACT_SHA256
        ),
        "signed_plan_sha256": _SIGNED_GFA_PLAN_SHA256,
        "basis_tensor_file_sha256": _BASIS_FILE_SHA256,
        "basis_payload_sha256": _BASIS_PAYLOAD_SHA256,
        "graph_basis_artifact_sha256": _GRAPH_BASIS_ARTIFACT_SHA256,
        "refit_scientific_payload_sha256": (
            _REFIT_SCIENTIFIC_PAYLOAD_SHA256
        ),
        "factorized_live_execution_sha256": (
            _FACTORIZED_LIVE_EXECUTION_SHA256
        ),
        "factorized_refit_execution_sha256": (
            _FACTORIZED_REFIT_EXECUTION_SHA256
        ),
        "runtime_binding_sha256": _json_sha256(
            _runtime_binding_payload(),
            domain=_RUNTIME_BINDING_DOMAIN,
        ),
        "candidate_independent_manifest": True,
        "subset_independent": True,
    }


def _runtime_binding_payload() -> dict[str, object]:
    return {
        "candidate_logical_artifact_sha256": (
            _GRAPH_CANDIDATE_ARTIFACT_SHA256
        ),
        "basis_payload_sha256": _BASIS_PAYLOAD_SHA256,
        "signed_plan_sha256": _SIGNED_GFA_PLAN_SHA256,
        "source_model_sha256": _SOURCE_MODEL_SHA256,
        "factorized_live_execution_sha256": (
            _FACTORIZED_LIVE_EXECUTION_SHA256
        ),
        "adapter_execution_fingerprint": (
            _FACTORIZED_REFIT_EXECUTION_SHA256
        ),
        "binding_scope": (
            "all_on_partial_edge_reference_oracle_shadow_metrics_only"
        ),
        "routing_enabled": False,
    }


def _tokenizer_contract_payload() -> dict[str, object]:
    """Return the frozen local tokenizer and one-prompt tokenization ABI."""

    return {
        "tokenizer_class": _TOKENIZER_CLASS,
        "name_or_path": _MODEL_ID,
        "configuration_sha256": _TOKENIZER_CONFIGURATION_SHA256,
        "backend_serialized_bytes": _TOKENIZER_BACKEND_SERIALIZED_BYTES,
        "backend_serialized_sha256": (
            _TOKENIZER_BACKEND_SERIALIZED_SHA256
        ),
        "post_tokenization_backend_serialized_bytes": (
            _TOKENIZER_POST_TOKENIZATION_BACKEND_SERIALIZED_BYTES
        ),
        "post_tokenization_backend_serialized_sha256": (
            _TOKENIZER_POST_TOKENIZATION_BACKEND_SERIALIZED_SHA256
        ),
        "canonical_vocab_count": _TOKENIZER_CANONICAL_VOCAB_COUNT,
        "canonical_vocab_sha256": _TOKENIZER_CANONICAL_VOCAB_SHA256,
        "added_token_count": _TOKENIZER_ADDED_TOKEN_COUNT,
        "added_tokens_sha256": _TOKENIZER_ADDED_TOKENS_SHA256,
        "special_tokens_map_sha256": (
            _TOKENIZER_SPECIAL_TOKENS_MAP_SHA256
        ),
        "transformers_version": _TOKENIZER_TRANSFORMERS_VERSION,
        "tokenizers_version": _TOKENIZER_TOKENIZERS_VERSION,
        "sentencepiece_version": _TOKENIZER_SENTENCEPIECE_VERSION,
        "vocab_size": _GEMMA_VOCAB_SIZE,
        "model_revision": _REVISION,
        "local_files_only": True,
        "max_length": 256,
        "tokenization_batch_size": 1,
        "device": "cpu",
        "padding_side": "right",
        "padding": True,
        "truncation": True,
        "add_special_tokens": True,
        "return_attention_mask": True,
    }


def _protocol_payload() -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "model": {
            "model_id": _MODEL_ID,
            "requested_revision": _REVISION,
            "resolved_commit": _REVISION,
            "source_model_sha256": _SOURCE_MODEL_SHA256,
            "vocab_size": _GEMMA_VOCAB_SIZE,
            "local_files_only": True,
        },
        "tokenizer": _tokenizer_contract_payload(),
        "graph_candidate": {
            "tensor_file_sha256": _GRAPH_CANDIDATE_FILE_SHA256,
            "logical_artifact_sha256": (
                _GRAPH_CANDIDATE_ARTIFACT_SHA256
            ),
            "factorized_live_execution_sha256": (
                _FACTORIZED_LIVE_EXECUTION_SHA256
            ),
            "factorized_refit_execution_sha256": (
                _FACTORIZED_REFIT_EXECUTION_SHA256
            ),
            "global_svd_base_plan_sha256": _GLOBAL_SVD_BASE_PLAN_SHA256,
            "graph_basis_artifact_sha256": (
                _GRAPH_BASIS_ARTIFACT_SHA256
            ),
            "deployment_plan_key": "signed_gfa",
            "deployment_plan_sha256": _SIGNED_GFA_PLAN_SHA256,
        },
        "prompt_blind_basis": {
            "tensor_file_sha256": _BASIS_FILE_SHA256,
            "logical_payload_sha256": _BASIS_PAYLOAD_SHA256,
            "target_modal_width": _TARGET_MODAL_WIDTH,
            "target_full_width": _TARGET_FULL_WIDTH,
        },
        "upstream_generator_lineage": {
            "hierarchy_artifact_sha256": _HIERARCHY_ARTIFACT_SHA256,
            "base_artifact_file_sha256": _BASE_ARTIFACT_FILE_SHA256,
            "base_scientific_payload_sha256": (
                _BASE_SCIENTIFIC_PAYLOAD_SHA256
            ),
            "refit_artifact_file_sha256": _REFIT_ARTIFACT_FILE_SHA256,
            "refit_scientific_payload_sha256": (
                _REFIT_SCIENTIFIC_PAYLOAD_SHA256
            ),
            "authoritative_logits": "sequential_refit_source_model",
        },
        "corpus": {
            "corpus_id": _CORPUS_ID,
            "prompt_file_sha256": _PROMPT_FILE_SHA256,
            "family_file_sha256": _FAMILY_FILE_SHA256,
            "audit_file_sha256": _AUDIT_FILE_SHA256,
            "example_identity": "prompt_sha256_only",
            "prompt_text_loaded_by_protocol_or_evaluator": False,
            "tokenizer_loaded_by_protocol_or_evaluator": False,
            "calibration_a_fit": "development_only",
            "calibration_a_guard": "development_only",
            "calibration_b": "unopened_one_shot",
            "calibration_b_policy": (
                "one_shot_frozen_candidate_selection"
            ),
            "calibration_b_manifest": {
                "role": "calibration_b_one_shot",
                "artifact_sha256": _CALIBRATION_B_MANIFEST_SHA256,
                "example_count": _CALIBRATION_B_EXAMPLE_COUNT,
                "family_count": _CALIBRATION_B_FAMILY_COUNT,
                "derivation": (
                    "canonical_sorted_zip_of_audit_prompt_sha256_by_role_"
                    "calibration_b_and_family_file_calibration_b"
                ),
                "prompt_file_opened_for_derivation": False,
            },
            "validation": "unopened",
            "test": "unopened",
        },
        "arms": {
            "primary": "all_on",
            "routed": "locked_disabled_and_rejected",
        },
        "runtime_binding_contract": {
            **_runtime_binding_payload(),
            "artifact_sha256": _json_sha256(
                _runtime_binding_payload(),
                domain=_RUNTIME_BINDING_DOMAIN,
            ),
        },
        "causal_geometry": {
            "source_origin_min_inclusive": _SOURCE_ORIGIN_MIN,
            "source_origin_max_inclusive": _SOURCE_ORIGIN_MAX,
            "lag_count": _LAG_COUNT,
            "producer_target_affected_mask_trusted": False,
            "producer_affected_supervised_mask_trusted": False,
            "valid_mask_scope": "one_contiguous_span",
            "valid_logical_position_stride": 1,
            "supervised_boundary_index_scope": (
                "all_adjacent_valid_next_token_rows"
            ),
        },
        "behavioral_gates": (
            ESTABLISHED_SHADOW_FIDELITY_GATES.metadata()
        ),
        "boundary_gates": {
            "pooled_target_modal_relative_error_max": (
                _BOUNDARY_RELATIVE_ERROR_MAX
            ),
            "pooled_target_modal_cosine_min": _BOUNDARY_COSINE_MIN,
            "valid_target_coverage_min": _VALID_TARGET_COVERAGE_MIN,
            "worst_family_target_modal_relative_error_max": (
                _WORST_FAMILY_MODAL_RELATIVE_ERROR_MAX
            ),
            "worst_family_target_modal_cosine_min": (
                _WORST_FAMILY_MODAL_COSINE_MIN
            ),
            "minimum_family_source_modal_signal_l2_norm": (
                _MINIMUM_FAMILY_SIGNAL_L2_NORM
            ),
        },
        "projection_capacity_gates": {
            "pooled_full_width_delta_relative_error_max": (
                _PROJECTION_RELATIVE_ERROR_MAX
            ),
            "pooled_full_width_delta_cosine_min": _PROJECTION_COSINE_MIN,
            "projection_oracle_behavioral_gates": (
                ESTABLISHED_SHADOW_FIDELITY_GATES.metadata()
            ),
            "worst_family_full_width_delta_relative_error_max": (
                _WORST_FAMILY_PROJECTION_RELATIVE_ERROR_MAX
            ),
            "worst_family_full_width_delta_cosine_min": (
                _WORST_FAMILY_PROJECTION_COSINE_MIN
            ),
            "minimum_family_source_full_width_signal_l2_norm": (
                _MINIMUM_FAMILY_SIGNAL_L2_NORM
            ),
        },
        "carrier_completeness_gates": {
            "exact_full_width_x4_on_clamped_reference_carrier": (
                ESTABLISHED_SHADOW_FIDELITY_GATES.metadata()
            ),
            "interpretation": (
                "incomplete_replacement_not_isolated_boundary_fidelity"
            ),
        },
        "calibration_a_development_evidence": {
            "selection_or_assessment_eligible": False,
            "token_count": 44,
            "source_row_count": 33,
            "affected_target_row_count": 36,
            "valid_target_coverage": 36 / 44,
            "corrected_all_on": {
                "pooled_target_modal_relative_error": 4.8208,
                "pooled_target_modal_cosine": 0.5404,
                "predicted_to_actual_norm_ratio": 5.287,
                "delta_nll_per_token": 3.0853,
                "source_to_candidate_kl_per_token": 3.3077,
                "top1_agreement_to_source": 0.3953,
                "passed": False,
            },
            "projection_capacity": {
                "target_modal_width": 64,
                "pooled_full_width_delta_relative_error": 0.9741,
                "pooled_full_width_delta_cosine": 0.2261,
                "delta_nll_per_token": 2.2416,
                "top1_agreement_to_source": 0.4651,
                "largest_reduced_rank_tested": 512,
                "rank_512_full_width_delta_relative_error": 0.5765,
                "rank_512_full_width_delta_cosine": 0.8171,
                "first_reconstructing_rank": 640,
                "passed": False,
            },
            "carrier_completeness": {
                "carrier": "clamped_y3_source_model_reference",
                "boundary": "exact_full_width_x4",
                "delta_nll_per_token": 2.0121,
                "top1_agreement_to_source": 0.4651,
                "interpretation": (
                    "incomplete_replacement_not_isolated_boundary_fidelity"
                ),
                "passed": False,
            },
            "deployment_authorized": False,
            "routing_authorized": False,
        },
        "scope": {
            "execution_mode": "source_authoritative_shadow",
            "partial_edge_only": True,
            "reference_provider": "clamped_y3_source_model_oracle",
            "reference_pass_oracle_fallback_required": True,
            "candidate_logits_metrics_only": True,
            "candidate_outputs_must_not_be_served": True,
            "candidate_logits_interpretation": (
                "incomplete_replacement_on_reference_carrier"
            ),
            "standalone_deployment_claim": False,
            "full_model_claim": False,
            "parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
            "required_model_forward_count_per_example": 5,
            "required_model_forward_shape": "3_plus_1_plus_1",
            "five_pass_receipt_required": True,
            "five_pass_receipt_binds_evidence_payload": True,
            "prompt_identity_must_equal_example_id": True,
            "model_inputs_sha256_abi": (
                "gemma3_l3_l4_shadow_model_inputs_sha256"
            ),
            "unique_model_inputs_per_manifest_example": True,
            "unique_input_provenance_per_manifest_example": True,
            "unique_five_pass_receipt_per_manifest_example": True,
            "behavioral_token_scope": (
                "causally_affected_supervised_tokens_only"
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphOrganizedSVDShadowProtocol:
    """Canonical hash-bound pre-assessment protocol."""

    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        computed = _json_sha256(
            _protocol_payload(),
            domain=_PROTOCOL_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="protocol artifact",
                )
                != computed
            ):
                raise ValueError("shadow protocol hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def behavioral_gates(self) -> ShadowFidelityGates:
        return ESTABLISHED_SHADOW_FIDELITY_GATES

    @property
    def target_modal_width(self) -> int:
        return _TARGET_MODAL_WIDTH

    def calibration_b_assessment_claim_sha256(self) -> str:
        self.validate_integrity()
        return _json_sha256(
            _assessment_claim_payload(self.artifact_sha256),
            domain=_ASSESSMENT_CLAIM_DOMAIN,
        )

    def validate_runtime_binding(
        self,
        metadata: Mapping[str, object],
    ) -> str:
        """Authenticate the exact locked all-on five-pass runtime."""

        self.validate_integrity()
        if not isinstance(metadata, Mapping):
            raise TypeError("runtime binding metadata must be a mapping")
        expected = _runtime_binding_payload()
        if set(metadata) != set(expected):
            raise ValueError("runtime binding metadata fields differ")
        if _canonical_json_bytes(metadata) != _canonical_json_bytes(expected):
            raise ValueError("runtime binding metadata differs from freeze")
        return _json_sha256(expected, domain=_RUNTIME_BINDING_DOMAIN)

    def validate_integrity(self) -> None:
        if (
            _json_sha256(
                _protocol_payload(),
                domain=_PROTOCOL_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("shadow protocol hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **_protocol_payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "Gemma3L3L4GraphOrganizedSVDShadowProtocol":
        if not isinstance(raw, Mapping):
            raise TypeError("shadow protocol state must be a mapping")
        expected = {*_protocol_payload(), "artifact_sha256"}
        if set(raw) != expected:
            raise ValueError("shadow protocol state fields differ")
        result = cls(
            artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
        )
        if _canonical_json_bytes(raw) != _canonical_json_bytes(
            result.state_dict()
        ):
            raise ValueError("shadow protocol state differs from the freeze")
        return result


def default_gemma3_l3_l4_graph_organized_svd_shadow_protocol(
) -> Gemma3L3L4GraphOrganizedSVDShadowProtocol:
    return Gemma3L3L4GraphOrganizedSVDShadowProtocol()


_OBSERVATION_TENSORS = (
    "source_logits",
    "candidate_logits",
    "projection_oracle_logits",
    "carrier_oracle_logits",
    "targets",
    "source_target_modes",
    "candidate_target_modes",
    "source_target_full_width_delta",
    "projection_target_full_width_delta",
    "logical_positions",
    "supervised_boundary_indices",
    "valid_target_mask",
    "source_eligible_mask",
)


def _evidence_payload_sha256_from_canonical(
    tensors: Mapping[str, Tensor],
) -> str:
    if set(tensors) != set(_OBSERVATION_TENSORS):
        raise ValueError("evidence payload tensor fields differ")
    payload = {
        "schema": f"{_SCHEMA}.evidence_payload",
        "format_version": _FORMAT_VERSION,
        "tensor_sha256s": {
            name: _tensor_sha256(tensors[name])
            for name in _OBSERVATION_TENSORS
        },
        "tensor_shapes": {
            name: tuple(int(width) for width in tensors[name].shape)
            for name in _OBSERVATION_TENSORS
        },
    }
    return _json_sha256(payload, domain=_EVIDENCE_PAYLOAD_DOMAIN)


def gemma3_l3_l4_graph_organized_svd_evidence_payload_sha256(
    tensors: Mapping[str, Tensor],
) -> str:
    """Hash the exact observation evidence copied from authenticated passes."""

    if not isinstance(tensors, Mapping):
        raise TypeError("evidence payload tensors must be a mapping")
    if set(tensors) != set(_OBSERVATION_TENSORS):
        raise ValueError("evidence payload tensor fields differ")
    canonical: dict[str, Tensor] = {}
    for name in _OBSERVATION_TENSORS:
        value = tensors[name]
        if name in {
            "targets",
            "logical_positions",
            "supervised_boundary_indices",
        }:
            canonical[name] = _int_tensor(value, label=name)
        elif name.endswith("_mask"):
            canonical[name] = _bool_tensor(value, label=name)
        else:
            canonical[name] = _float_tensor(
                value,
                label=name,
                ndim=2,
                require_finite=name.endswith("_logits"),
            )
    return _evidence_payload_sha256_from_canonical(canonical)


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphOrganizedSVDShadowObservation:
    """One hash-identified one-shot behavioral and boundary observation."""

    protocol_sha256: str
    assessment_claim_sha256: str
    runtime_binding_sha256: str
    role: str
    arm: ShadowArm
    example_id: str
    family_id: str
    prompt_identity_sha256: str
    model_inputs_sha256: str
    input_provenance_sha256: str
    shadow_result_artifact_sha256: str
    execution_grid_sha256: str
    projection_oracle_artifact_sha256: str
    projection_injected_x4_sha256: str
    carrier_oracle_artifact_sha256: str
    carrier_injected_x4_sha256: str
    evidence_payload_sha256: str
    five_pass_receipt_sha256: str
    source_logits: Tensor
    candidate_logits: Tensor
    projection_oracle_logits: Tensor
    carrier_oracle_logits: Tensor
    targets: Tensor
    source_target_modes: Tensor
    candidate_target_modes: Tensor
    source_target_full_width_delta: Tensor
    projection_target_full_width_delta: Tensor
    logical_positions: Tensor
    supervised_boundary_indices: Tensor
    valid_target_mask: Tensor
    source_eligible_mask: Tensor
    reference_provider: str = "clamped_y3_source_model_oracle"
    source_logits_role: str = "sequential_refit_authoritative"
    candidate_logits_role: str = (
        "reference_carrier_incomplete_replacement_metrics_only"
    )
    projection_oracle_logits_role: str = (
        "true_64_mode_delta_dual_decode_on_reference_carrier_metrics_only"
    )
    carrier_oracle_logits_role: str = (
        "exact_full_width_x4_on_clamped_reference_carrier_metrics_only"
    )
    carrier_interpretation: str = (
        "incomplete_replacement_not_isolated_boundary_fidelity"
    )
    behavioral_token_scope: str = (
        "causally_affected_supervised_tokens_only"
    )
    reference_pass_used: bool = True
    partial_edge_only: bool = True
    candidate_served: bool = False
    shadow_model_forward_count: int = 3
    projection_oracle_model_forward_count: int = 1
    carrier_oracle_model_forward_count: int = 1
    projection_oracle_receipt_role: str = "projection_64"
    carrier_oracle_receipt_role: str = "exact_x4_carrier"
    model_forward_count: int = 5
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="observation protocol")
        _require_sha256(
            self.assessment_claim_sha256,
            label="observation assessment claim",
        )
        _require_sha256(
            self.runtime_binding_sha256,
            label="observation runtime binding",
        )
        if self.role != "calibration_b_one_shot":
            raise ValueError(
                "shadow observation role must be calibration_b_one_shot"
            )
        object.__setattr__(self, "arm", _arm(self.arm))
        object.__setattr__(
            self,
            "example_id",
            _require_sha256(self.example_id, label="example_id"),
        )
        object.__setattr__(
            self,
            "family_id",
            _identifier(self.family_id, label="family_id"),
        )
        for name, label in (
            ("prompt_identity_sha256", "prompt identity"),
            ("model_inputs_sha256", "model inputs"),
            ("input_provenance_sha256", "input provenance"),
            ("shadow_result_artifact_sha256", "shadow result artifact"),
            ("execution_grid_sha256", "execution grid"),
            (
                "projection_oracle_artifact_sha256",
                "projection oracle artifact",
            ),
            (
                "projection_injected_x4_sha256",
                "projection injected X4",
            ),
            (
                "carrier_oracle_artifact_sha256",
                "carrier oracle artifact",
            ),
            ("carrier_injected_x4_sha256", "carrier injected X4"),
            ("evidence_payload_sha256", "evidence payload"),
            ("five_pass_receipt_sha256", "five-pass receipt"),
        ):
            _require_sha256(getattr(self, name), label=label)
        if self.prompt_identity_sha256 != self.example_id:
            raise ValueError(
                "prompt identity SHA-256 must equal example_id"
            )
        if (
            self.projection_oracle_artifact_sha256
            == self.carrier_oracle_artifact_sha256
        ):
            raise ValueError(
                "projection and carrier oracle artifacts must be distinct"
            )
        for name in (
            "source_logits",
            "candidate_logits",
            "projection_oracle_logits",
            "carrier_oracle_logits",
        ):
            object.__setattr__(
                self,
                name,
                _float_tensor(
                    getattr(self, name),
                    label=name,
                    ndim=2,
                    require_finite=True,
                ),
            )
        object.__setattr__(
            self,
            "targets",
            _int_tensor(self.targets, label="targets"),
        )
        for name in (
            "source_target_modes",
            "candidate_target_modes",
            "source_target_full_width_delta",
            "projection_target_full_width_delta",
        ):
            object.__setattr__(
                self,
                name,
                _float_tensor(
                    getattr(self, name),
                    label=name,
                    ndim=2,
                    require_finite=False,
                ),
            )
        for name in ("logical_positions", "supervised_boundary_indices"):
            object.__setattr__(
                self,
                name,
                _int_tensor(getattr(self, name), label=name),
            )
        for name in ("valid_target_mask", "source_eligible_mask"):
            object.__setattr__(
                self,
                name,
                _bool_tensor(getattr(self, name), label=name),
            )
        if (
            self.source_logits.shape != self.candidate_logits.shape
            or self.source_logits.shape
            != self.projection_oracle_logits.shape
            or self.source_logits.shape != self.carrier_oracle_logits.shape
            or self.source_logits.shape[0] != self.targets.numel()
            or self.supervised_boundary_indices.numel()
            != self.targets.numel()
            or self.source_logits.shape[1] != _GEMMA_VOCAB_SIZE
            or self.source_target_modes.shape
            != self.candidate_target_modes.shape
            or self.source_target_modes.shape[1] != _TARGET_MODAL_WIDTH
            or self.source_target_full_width_delta.shape
            != self.projection_target_full_width_delta.shape
            or self.source_target_full_width_delta.shape[0]
            != self.source_target_modes.shape[0]
            or self.source_target_full_width_delta.shape[1]
            != _TARGET_FULL_WIDTH
            or self.valid_target_mask.numel()
            != self.source_target_modes.shape[0]
            or self.logical_positions.numel()
            != self.source_target_modes.shape[0]
            or self.source_eligible_mask.shape
            != self.valid_target_mask.shape
            or not bool(self.valid_target_mask.any())
        ):
            raise ValueError("shadow observation tensor geometry differs")
        derived_masks = (
            derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
                self.logical_positions,
                self.valid_target_mask,
                self.supervised_boundary_indices,
            )
        )
        if not torch.equal(
            self.source_eligible_mask,
            derived_masks["source_eligible_mask"],
        ):
            raise ValueError("producer source-eligible mask differs")
        target_affected = derived_masks["target_affected_mask"]
        if bool(target_affected.any()):
            affected_source = self.source_target_modes[
                target_affected
            ]
            affected_candidate = self.candidate_target_modes[
                target_affected
            ]
            affected_source_full = self.source_target_full_width_delta[
                target_affected
            ]
            affected_projection_full = (
                self.projection_target_full_width_delta[
                    target_affected
                ]
            )
            if not (
                bool(torch.isfinite(affected_source).all())
                and bool(torch.isfinite(affected_candidate).all())
                and bool(torch.isfinite(affected_source_full).all())
                and bool(torch.isfinite(affected_projection_full).all())
            ):
                raise ValueError("affected target rows must be finite")
        if (
            self.reference_provider
            != "clamped_y3_source_model_oracle"
            or self.source_logits_role
            != "sequential_refit_authoritative"
            or self.candidate_logits_role
            != "reference_carrier_incomplete_replacement_metrics_only"
            or self.projection_oracle_logits_role
            != (
                "true_64_mode_delta_dual_decode_on_reference_carrier_"
                "metrics_only"
            )
            or self.carrier_oracle_logits_role
            != (
                "exact_full_width_x4_on_clamped_reference_carrier_"
                "metrics_only"
            )
            or self.carrier_interpretation
            != "incomplete_replacement_not_isolated_boundary_fidelity"
            or self.behavioral_token_scope
            != "causally_affected_supervised_tokens_only"
            or self.reference_pass_used is not True
            or self.partial_edge_only is not True
            or self.candidate_served is not False
            or self.shadow_model_forward_count != 3
            or self.projection_oracle_model_forward_count != 1
            or self.carrier_oracle_model_forward_count != 1
            or self.projection_oracle_receipt_role != "projection_64"
            or self.carrier_oracle_receipt_role != "exact_x4_carrier"
            or self.model_forward_count != 5
        ):
            raise ValueError("shadow observation provenance drifted")
        evidence_payload_sha256 = (
            _evidence_payload_sha256_from_canonical(
                {
                    name: getattr(self, name)
                    for name in _OBSERVATION_TENSORS
                }
            )
        )
        if self.evidence_payload_sha256 != evidence_payload_sha256:
            raise ValueError("shadow observation evidence payload differs")
        receipt = (
            derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt(
                protocol_sha256=self.protocol_sha256,
                assessment_claim_sha256=self.assessment_claim_sha256,
                runtime_binding_sha256=self.runtime_binding_sha256,
                example_id=self.example_id,
                family_id=self.family_id,
                prompt_identity_sha256=self.prompt_identity_sha256,
                model_inputs_sha256=self.model_inputs_sha256,
                shadow_result_artifact_sha256=(
                    self.shadow_result_artifact_sha256
                ),
                execution_grid_sha256=self.execution_grid_sha256,
                projection_oracle_artifact_sha256=(
                    self.projection_oracle_artifact_sha256
                ),
                projection_injected_x4_sha256=(
                    self.projection_injected_x4_sha256
                ),
                carrier_oracle_artifact_sha256=(
                    self.carrier_oracle_artifact_sha256
                ),
                carrier_injected_x4_sha256=(
                    self.carrier_injected_x4_sha256
                ),
                evidence_payload_sha256=self.evidence_payload_sha256,
                shadow_model_forward_count=(
                    self.shadow_model_forward_count
                ),
                projection_oracle_model_forward_count=(
                    self.projection_oracle_model_forward_count
                ),
                carrier_oracle_model_forward_count=(
                    self.carrier_oracle_model_forward_count
                ),
                projection_oracle_role=(
                    self.projection_oracle_receipt_role
                ),
                carrier_oracle_role=self.carrier_oracle_receipt_role,
            )
        )
        if (
            self.input_provenance_sha256
            != receipt["input_provenance_sha256"]
            or self.five_pass_receipt_sha256
            != receipt["five_pass_receipt_sha256"]
        ):
            raise ValueError("shadow observation five-pass receipt differs")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="observation artifact",
                )
                != computed
            ):
                raise ValueError("shadow observation hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "assessment_claim_sha256": self.assessment_claim_sha256,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "role": self.role,
            "arm": self.arm,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "prompt_identity_sha256": self.prompt_identity_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "input_provenance_sha256": self.input_provenance_sha256,
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "execution_grid_sha256": self.execution_grid_sha256,
            "projection_oracle_artifact_sha256": (
                self.projection_oracle_artifact_sha256
            ),
            "projection_injected_x4_sha256": (
                self.projection_injected_x4_sha256
            ),
            "carrier_oracle_artifact_sha256": (
                self.carrier_oracle_artifact_sha256
            ),
            "carrier_injected_x4_sha256": (
                self.carrier_injected_x4_sha256
            ),
            "evidence_payload_sha256": self.evidence_payload_sha256,
            "five_pass_receipt_sha256": self.five_pass_receipt_sha256,
            "reference_provider": self.reference_provider,
            "source_logits_role": self.source_logits_role,
            "candidate_logits_role": self.candidate_logits_role,
            "projection_oracle_logits_role": (
                self.projection_oracle_logits_role
            ),
            "carrier_oracle_logits_role": self.carrier_oracle_logits_role,
            "carrier_interpretation": self.carrier_interpretation,
            "behavioral_token_scope": self.behavioral_token_scope,
            "reference_pass_used": self.reference_pass_used,
            "partial_edge_only": self.partial_edge_only,
            "candidate_served": self.candidate_served,
            "shadow_model_forward_count": (
                self.shadow_model_forward_count
            ),
            "projection_oracle_model_forward_count": (
                self.projection_oracle_model_forward_count
            ),
            "carrier_oracle_model_forward_count": (
                self.carrier_oracle_model_forward_count
            ),
            "projection_oracle_receipt_role": (
                self.projection_oracle_receipt_role
            ),
            "carrier_oracle_receipt_role": (
                self.carrier_oracle_receipt_role
            ),
            "model_forward_count": self.model_forward_count,
            "tensor_sha256s": {
                name: _tensor_sha256(getattr(self, name))
                for name in _OBSERVATION_TENSORS
            },
            "tensor_shapes": {
                name: tuple(
                    int(width) for width in getattr(self, name).shape
                )
                for name in _OBSERVATION_TENSORS
            },
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._payload(),
            domain=_OBSERVATION_DOMAIN,
        )

    def validate_integrity(self) -> None:
        for name in _OBSERVATION_TENSORS:
            value = getattr(self, name)
            expected_dtype = (
                torch.int64
                if name
                in {
                    "targets",
                    "logical_positions",
                    "supervised_boundary_indices",
                }
                else torch.bool
                if name.endswith("_mask")
                else torch.float64
            )
            if (
                value.device.type != "cpu"
                or not value.is_contiguous()
                or value.dtype != expected_dtype
            ):
                raise ValueError(f"observation {name} drifted")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("shadow observation hash mismatch")

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            **{
                name: getattr(self, name).clone()
                for name in _OBSERVATION_TENSORS
            },
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "Gemma3L3L4GraphOrganizedSVDShadowObservation":
        if not isinstance(raw, Mapping):
            raise TypeError("shadow observation state must be a mapping")
        scalar = {
            "protocol_sha256",
            "assessment_claim_sha256",
            "runtime_binding_sha256",
            "role",
            "arm",
            "example_id",
            "family_id",
            "prompt_identity_sha256",
            "model_inputs_sha256",
            "input_provenance_sha256",
            "shadow_result_artifact_sha256",
            "execution_grid_sha256",
            "projection_oracle_artifact_sha256",
            "projection_injected_x4_sha256",
            "carrier_oracle_artifact_sha256",
            "carrier_injected_x4_sha256",
            "evidence_payload_sha256",
            "five_pass_receipt_sha256",
            "reference_provider",
            "source_logits_role",
            "candidate_logits_role",
            "projection_oracle_logits_role",
            "carrier_oracle_logits_role",
            "carrier_interpretation",
            "behavioral_token_scope",
            "reference_pass_used",
            "partial_edge_only",
            "candidate_served",
            "shadow_model_forward_count",
            "projection_oracle_model_forward_count",
            "carrier_oracle_model_forward_count",
            "projection_oracle_receipt_role",
            "carrier_oracle_receipt_role",
            "model_forward_count",
            "tensor_sha256s",
            "tensor_shapes",
            "artifact_sha256",
        }
        if set(raw) != {*scalar, *_OBSERVATION_TENSORS}:
            raise ValueError("shadow observation state fields differ")
        result = cls(
            protocol_sha256=raw["protocol_sha256"],  # type: ignore[arg-type]
            assessment_claim_sha256=raw[
                "assessment_claim_sha256"
            ],  # type: ignore[arg-type]
            runtime_binding_sha256=raw[
                "runtime_binding_sha256"
            ],  # type: ignore[arg-type]
            role=raw["role"],  # type: ignore[arg-type]
            arm=raw["arm"],  # type: ignore[arg-type]
            example_id=raw["example_id"],  # type: ignore[arg-type]
            family_id=raw["family_id"],  # type: ignore[arg-type]
            prompt_identity_sha256=raw[
                "prompt_identity_sha256"
            ],  # type: ignore[arg-type]
            model_inputs_sha256=raw[
                "model_inputs_sha256"
            ],  # type: ignore[arg-type]
            input_provenance_sha256=raw[
                "input_provenance_sha256"
            ],  # type: ignore[arg-type]
            shadow_result_artifact_sha256=raw[
                "shadow_result_artifact_sha256"
            ],  # type: ignore[arg-type]
            execution_grid_sha256=raw[
                "execution_grid_sha256"
            ],  # type: ignore[arg-type]
            projection_oracle_artifact_sha256=raw[
                "projection_oracle_artifact_sha256"
            ],  # type: ignore[arg-type]
            projection_injected_x4_sha256=raw[
                "projection_injected_x4_sha256"
            ],  # type: ignore[arg-type]
            carrier_oracle_artifact_sha256=raw[
                "carrier_oracle_artifact_sha256"
            ],  # type: ignore[arg-type]
            carrier_injected_x4_sha256=raw[
                "carrier_injected_x4_sha256"
            ],  # type: ignore[arg-type]
            evidence_payload_sha256=raw[
                "evidence_payload_sha256"
            ],  # type: ignore[arg-type]
            five_pass_receipt_sha256=raw[
                "five_pass_receipt_sha256"
            ],  # type: ignore[arg-type]
            source_logits=raw["source_logits"],  # type: ignore[arg-type]
            candidate_logits=raw["candidate_logits"],  # type: ignore[arg-type]
            projection_oracle_logits=raw[
                "projection_oracle_logits"
            ],  # type: ignore[arg-type]
            carrier_oracle_logits=raw[
                "carrier_oracle_logits"
            ],  # type: ignore[arg-type]
            targets=raw["targets"],  # type: ignore[arg-type]
            source_target_modes=raw[
                "source_target_modes"
            ],  # type: ignore[arg-type]
            candidate_target_modes=raw[
                "candidate_target_modes"
            ],  # type: ignore[arg-type]
            source_target_full_width_delta=raw[
                "source_target_full_width_delta"
            ],  # type: ignore[arg-type]
            projection_target_full_width_delta=raw[
                "projection_target_full_width_delta"
            ],  # type: ignore[arg-type]
            logical_positions=raw[
                "logical_positions"
            ],  # type: ignore[arg-type]
            supervised_boundary_indices=raw[
                "supervised_boundary_indices"
            ],  # type: ignore[arg-type]
            valid_target_mask=raw[
                "valid_target_mask"
            ],  # type: ignore[arg-type]
            source_eligible_mask=raw[
                "source_eligible_mask"
            ],  # type: ignore[arg-type]
            reference_provider=raw[
                "reference_provider"
            ],  # type: ignore[arg-type]
            source_logits_role=raw[
                "source_logits_role"
            ],  # type: ignore[arg-type]
            candidate_logits_role=raw[
                "candidate_logits_role"
            ],  # type: ignore[arg-type]
            projection_oracle_logits_role=raw[
                "projection_oracle_logits_role"
            ],  # type: ignore[arg-type]
            carrier_oracle_logits_role=raw[
                "carrier_oracle_logits_role"
            ],  # type: ignore[arg-type]
            carrier_interpretation=raw[
                "carrier_interpretation"
            ],  # type: ignore[arg-type]
            behavioral_token_scope=raw[
                "behavioral_token_scope"
            ],  # type: ignore[arg-type]
            reference_pass_used=raw[
                "reference_pass_used"
            ],  # type: ignore[arg-type]
            partial_edge_only=raw[
                "partial_edge_only"
            ],  # type: ignore[arg-type]
            candidate_served=raw["candidate_served"],  # type: ignore[arg-type]
            shadow_model_forward_count=raw[
                "shadow_model_forward_count"
            ],  # type: ignore[arg-type]
            projection_oracle_model_forward_count=raw[
                "projection_oracle_model_forward_count"
            ],  # type: ignore[arg-type]
            carrier_oracle_model_forward_count=raw[
                "carrier_oracle_model_forward_count"
            ],  # type: ignore[arg-type]
            projection_oracle_receipt_role=raw[
                "projection_oracle_receipt_role"
            ],  # type: ignore[arg-type]
            carrier_oracle_receipt_role=raw[
                "carrier_oracle_receipt_role"
            ],  # type: ignore[arg-type]
            model_forward_count=raw[
                "model_forward_count"
            ],  # type: ignore[arg-type]
            artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
        )
        payload = result._payload()
        if (
            raw["tensor_sha256s"] != payload["tensor_sha256s"]
            or raw["tensor_shapes"] != payload["tensor_shapes"]
        ):
            raise ValueError("serialized observation tensors differ")
        return result


def _manifest(
    value: Mapping[str, str],
) -> tuple[dict[str, str], str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("expected family manifest must be nonempty")
    result: dict[str, str] = {}
    for example_id, family_id in value.items():
        digest = _require_sha256(example_id, label="manifest example_id")
        result[digest] = _identifier(family_id, label="manifest family_id")
    logical = tuple(sorted(result.items()))
    return result, _json_sha256(logical, domain=_MANIFEST_DOMAIN)


def _evaluate_arm(
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    observations: Iterable[
        Gemma3L3L4GraphOrganizedSVDShadowObservation
    ],
    *,
    arm: ShadowArm,
    assessment_claim_sha256: str,
    expected_family_by_example: Mapping[str, str],
) -> dict[str, object]:
    selected_arm = _arm(arm)
    expected_claim = _require_sha256(
        assessment_claim_sha256,
        label="assessment claim",
    )
    expected_runtime_binding = _json_sha256(
        _runtime_binding_payload(),
        domain=_RUNTIME_BINDING_DOMAIN,
    )
    behavioral_accumulator = SourceAuthoritativeShadowFidelityAccumulator(
        expected_family_by_example,
        gates=protocol.behavioral_gates,
    )
    projection_accumulator = SourceAuthoritativeShadowFidelityAccumulator(
        expected_family_by_example,
        gates=protocol.behavioral_gates,
    )
    carrier_accumulator = SourceAuthoritativeShadowFidelityAccumulator(
        expected_family_by_example,
        gates=protocol.behavioral_gates,
    )
    seen: set[str] = set()
    seen_model_inputs: set[str] = set()
    seen_input_provenance: set[str] = set()
    seen_shadow_results: set[str] = set()
    seen_projection_oracles: set[str] = set()
    seen_carrier_oracles: set[str] = set()
    seen_five_pass_receipts: set[str] = set()
    observation_count = 0
    residual_square = 0.0
    source_square = 0.0
    candidate_square = 0.0
    dot = 0.0
    projection_residual_square = 0.0
    projection_source_square = 0.0
    projection_candidate_square = 0.0
    projection_dot = 0.0
    valid_rows = 0
    affected_rows = 0
    supervised_tokens = 0
    affected_supervised_tokens = 0
    family_sums: dict[str, dict[str, float]] = {}
    per_example: list[dict[str, object]] = []
    for observation in observations:
        observation_count += 1
        if not isinstance(
            observation,
            Gemma3L3L4GraphOrganizedSVDShadowObservation,
        ):
            raise TypeError("shadow observations must use the strict type")
        observation.validate_integrity()
        if (
            observation.protocol_sha256 != protocol.artifact_sha256
            or observation.assessment_claim_sha256 != expected_claim
            or observation.runtime_binding_sha256
            != expected_runtime_binding
            or observation.arm != selected_arm
        ):
            raise ValueError("shadow observation binding or arm differs")
        if observation.example_id in seen:
            raise ValueError(
                f"duplicate shadow observation: {observation.example_id}"
            )
        seen.add(observation.example_id)
        for value, identities, label in (
            (
                observation.model_inputs_sha256,
                seen_model_inputs,
                "model inputs",
            ),
            (
                observation.input_provenance_sha256,
                seen_input_provenance,
                "input provenance",
            ),
            (
                observation.shadow_result_artifact_sha256,
                seen_shadow_results,
                "shadow result",
            ),
            (
                observation.projection_oracle_artifact_sha256,
                seen_projection_oracles,
                "projection oracle",
            ),
            (
                observation.carrier_oracle_artifact_sha256,
                seen_carrier_oracles,
                "carrier oracle",
            ),
            (
                observation.five_pass_receipt_sha256,
                seen_five_pass_receipts,
                "five-pass receipt",
            ),
        ):
            if value in identities:
                raise ValueError(
                    f"replayed shadow observation {label}: {value}"
                )
            identities.add(value)
        derived_masks = (
            derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
                observation.logical_positions,
                observation.valid_target_mask,
                observation.supervised_boundary_indices,
            )
        )
        if not torch.equal(
            observation.source_eligible_mask,
            derived_masks["source_eligible_mask"],
        ):
            raise ValueError("producer source-eligible mask differs")
        target_affected_mask = derived_masks["target_affected_mask"]
        behavioral_mask = derived_masks["affected_supervised_mask"]
        supervised = observation.targets.numel()
        affected_supervised = int(behavioral_mask.sum())
        supervised_tokens += supervised
        affected_supervised_tokens += affected_supervised
        behavioral_accumulator.add(
            ShadowFidelityExample(
                example_id=observation.example_id,
                family_id=observation.family_id,
                source_logits=observation.source_logits[behavioral_mask],
                candidate_logits=observation.candidate_logits[
                    behavioral_mask
                ],
                targets=observation.targets[behavioral_mask],
            )
        )
        projection_accumulator.add(
            ShadowFidelityExample(
                example_id=observation.example_id,
                family_id=observation.family_id,
                source_logits=observation.source_logits[behavioral_mask],
                candidate_logits=observation.projection_oracle_logits[
                    behavioral_mask
                ],
                targets=observation.targets[behavioral_mask],
            )
        )
        carrier_accumulator.add(
            ShadowFidelityExample(
                example_id=observation.example_id,
                family_id=observation.family_id,
                source_logits=observation.source_logits[behavioral_mask],
                candidate_logits=observation.carrier_oracle_logits[
                    behavioral_mask
                ],
                targets=observation.targets[behavioral_mask],
            )
        )
        valid = int(observation.valid_target_mask.sum())
        affected = int(target_affected_mask.sum())
        valid_rows += valid
        affected_rows += affected
        if affected:
            source = observation.source_target_modes[
                target_affected_mask
            ]
            candidate = observation.candidate_target_modes[
                target_affected_mask
            ]
            residual = candidate - source
            row_residual = float(residual.square().sum())
            row_source = float(source.square().sum())
            row_candidate = float(candidate.square().sum())
            row_dot = float((source * candidate).sum())
            residual_square += row_residual
            source_square += row_source
            candidate_square += row_candidate
            dot += row_dot
            source_full = observation.source_target_full_width_delta[
                target_affected_mask
            ]
            projection_full = (
                observation.projection_target_full_width_delta[
                    target_affected_mask
                ]
            )
            projection_residual = projection_full - source_full
            row_projection_residual = float(
                projection_residual.square().sum()
            )
            row_projection_source = float(source_full.square().sum())
            row_projection_candidate = float(projection_full.square().sum())
            row_projection_dot = float((source_full * projection_full).sum())
            projection_residual_square += row_projection_residual
            projection_source_square += row_projection_source
            projection_candidate_square += row_projection_candidate
            projection_dot += row_projection_dot
            family = family_sums.setdefault(
                observation.family_id,
                {
                    "modal_residual_square": 0.0,
                    "modal_source_square": 0.0,
                    "modal_candidate_square": 0.0,
                    "modal_dot": 0.0,
                    "projection_residual_square": 0.0,
                    "projection_source_square": 0.0,
                    "projection_candidate_square": 0.0,
                    "projection_dot": 0.0,
                },
            )
            family["modal_residual_square"] += row_residual
            family["modal_source_square"] += row_source
            family["modal_candidate_square"] += row_candidate
            family["modal_dot"] += row_dot
            family["projection_residual_square"] += (
                row_projection_residual
            )
            family["projection_source_square"] += row_projection_source
            family["projection_candidate_square"] += (
                row_projection_candidate
            )
            family["projection_dot"] += row_projection_dot
            row_relative = (
                math.sqrt(row_residual / row_source)
                if row_source > 0.0
                else 0.0
                if row_residual == 0.0
                else math.inf
            )
            denominator = math.sqrt(row_source * row_candidate)
            row_cosine = (
                row_dot / denominator
                if denominator > 0.0
                else 1.0
                if row_source == 0.0 and row_candidate == 0.0
                else 0.0
            )
            row_projection_relative = (
                math.sqrt(
                    row_projection_residual / row_projection_source
                )
                if row_projection_source > 0.0
                else 0.0
                if row_projection_residual == 0.0
                else math.inf
            )
            projection_denominator = math.sqrt(
                row_projection_source * row_projection_candidate
            )
            row_projection_cosine = (
                row_projection_dot / projection_denominator
                if projection_denominator > 0.0
                else 1.0
                if (
                    row_projection_source == 0.0
                    and row_projection_candidate == 0.0
                )
                else 0.0
            )
        else:
            row_relative = 0.0
            row_cosine = 1.0
            row_projection_relative = 0.0
            row_projection_cosine = 1.0
        per_example.append(
            {
                "example_id": observation.example_id,
                "family_id": observation.family_id,
                "valid_target_rows": valid,
                "affected_target_rows": affected,
                "valid_target_coverage": affected / valid,
                "supervised_tokens": supervised,
                "affected_supervised_tokens": affected_supervised,
                "affected_supervised_coverage": (
                    affected_supervised / supervised
                ),
                "target_modal_relative_error": row_relative,
                "target_modal_cosine": row_cosine,
                "projection_full_width_delta_relative_error": (
                    row_projection_relative
                ),
                "projection_full_width_delta_cosine": (
                    row_projection_cosine
                ),
                "observation_sha256": observation.artifact_sha256,
            }
        )
    if observation_count == 0:
        raise ValueError(f"{selected_arm} observations cannot be empty")
    behavioral = behavioral_accumulator.finalize()
    projection_oracle_behavioral = projection_accumulator.finalize()
    carrier_oracle_behavioral = carrier_accumulator.finalize()
    pooled_relative = (
        math.sqrt(residual_square / source_square)
        if source_square > 0.0
        else 0.0
        if residual_square == 0.0
        else math.inf
    )
    pooled_denominator = math.sqrt(source_square * candidate_square)
    pooled_cosine = (
        dot / pooled_denominator
        if pooled_denominator > 0.0
        else 1.0
        if source_square == 0.0 and candidate_square == 0.0
        else 0.0
    )
    pooled_projection_relative = (
        math.sqrt(
            projection_residual_square / projection_source_square
        )
        if projection_source_square > 0.0
        else 0.0
        if projection_residual_square == 0.0
        else math.inf
    )
    pooled_projection_denominator = math.sqrt(
        projection_source_square * projection_candidate_square
    )
    pooled_projection_cosine = (
        projection_dot / pooled_projection_denominator
        if pooled_projection_denominator > 0.0
        else 1.0
        if (
            projection_source_square == 0.0
            and projection_candidate_square == 0.0
        )
        else 0.0
    )
    family_metrics: list[dict[str, object]] = []
    for family_id, values in sorted(family_sums.items()):
        modal_source = values["modal_source_square"]
        modal_candidate = values["modal_candidate_square"]
        modal_residual = values["modal_residual_square"]
        modal_denominator = math.sqrt(modal_source * modal_candidate)
        projection_source = values["projection_source_square"]
        projection_candidate = values["projection_candidate_square"]
        projection_residual = values["projection_residual_square"]
        projection_denominator = math.sqrt(
            projection_source * projection_candidate
        )
        family_metrics.append(
            {
                "family_id": family_id,
                "source_modal_signal_l2_norm": math.sqrt(modal_source),
                "target_modal_relative_error": (
                    math.sqrt(modal_residual / modal_source)
                    if modal_source > 0.0
                    else math.inf
                ),
                "target_modal_cosine": (
                    values["modal_dot"] / modal_denominator
                    if modal_denominator > 0.0
                    else 0.0
                ),
                "source_full_width_signal_l2_norm": (
                    math.sqrt(projection_source)
                ),
                "projection_full_width_delta_relative_error": (
                    math.sqrt(projection_residual / projection_source)
                    if projection_source > 0.0
                    else math.inf
                ),
                "projection_full_width_delta_cosine": (
                    values["projection_dot"] / projection_denominator
                    if projection_denominator > 0.0
                    else 0.0
                ),
            }
        )
    if len(family_metrics) != _CALIBRATION_B_FAMILY_COUNT:
        raise ValueError("shadow observations do not cover all frozen families")
    worst_family_modal_relative = max(
        float(row["target_modal_relative_error"])
        for row in family_metrics
    )
    worst_family_modal_cosine = min(
        float(row["target_modal_cosine"]) for row in family_metrics
    )
    minimum_family_modal_signal = min(
        float(row["source_modal_signal_l2_norm"])
        for row in family_metrics
    )
    worst_family_projection_relative = max(
        float(row["projection_full_width_delta_relative_error"])
        for row in family_metrics
    )
    worst_family_projection_cosine = min(
        float(row["projection_full_width_delta_cosine"])
        for row in family_metrics
    )
    minimum_family_projection_signal = min(
        float(row["source_full_width_signal_l2_norm"])
        for row in family_metrics
    )
    coverage = affected_rows / valid_rows
    boundary_gates = {
        "pooled_target_modal_relative_error": (
            math.isfinite(pooled_relative)
            and pooled_relative <= _BOUNDARY_RELATIVE_ERROR_MAX
        ),
        "pooled_target_modal_cosine": (
            math.isfinite(pooled_cosine)
            and pooled_cosine >= _BOUNDARY_COSINE_MIN
        ),
        "valid_target_coverage": (
            coverage >= _VALID_TARGET_COVERAGE_MIN
        ),
        "worst_family_target_modal_relative_error": (
            math.isfinite(worst_family_modal_relative)
            and worst_family_modal_relative
            <= _WORST_FAMILY_MODAL_RELATIVE_ERROR_MAX
        ),
        "worst_family_target_modal_cosine": (
            math.isfinite(worst_family_modal_cosine)
            and worst_family_modal_cosine
            >= _WORST_FAMILY_MODAL_COSINE_MIN
        ),
        "nondegenerate_pooled_source_modal_signal": (
            math.sqrt(source_square) >= _MINIMUM_FAMILY_SIGNAL_L2_NORM
        ),
        "nondegenerate_every_family_source_modal_signal": (
            minimum_family_modal_signal
            >= _MINIMUM_FAMILY_SIGNAL_L2_NORM
        ),
    }
    boundary_gates["passed"] = all(boundary_gates.values())
    behavioral_passed = behavioral["gates"]["passed"] is True  # type: ignore[index]
    projection_capacity_gates = {
        "pooled_full_width_delta_relative_error": (
            math.isfinite(pooled_projection_relative)
            and pooled_projection_relative <= _PROJECTION_RELATIVE_ERROR_MAX
        ),
        "pooled_full_width_delta_cosine": (
            math.isfinite(pooled_projection_cosine)
            and pooled_projection_cosine >= _PROJECTION_COSINE_MIN
        ),
        "projection_oracle_behavioral": (
            projection_oracle_behavioral["gates"]["passed"]  # type: ignore[index]
            is True
        ),
        "worst_family_full_width_delta_relative_error": (
            math.isfinite(worst_family_projection_relative)
            and worst_family_projection_relative
            <= _WORST_FAMILY_PROJECTION_RELATIVE_ERROR_MAX
        ),
        "worst_family_full_width_delta_cosine": (
            math.isfinite(worst_family_projection_cosine)
            and worst_family_projection_cosine
            >= _WORST_FAMILY_PROJECTION_COSINE_MIN
        ),
        "nondegenerate_pooled_source_full_width_signal": (
            math.sqrt(projection_source_square)
            >= _MINIMUM_FAMILY_SIGNAL_L2_NORM
        ),
        "nondegenerate_every_family_source_full_width_signal": (
            minimum_family_projection_signal
            >= _MINIMUM_FAMILY_SIGNAL_L2_NORM
        ),
    }
    projection_capacity_gates["passed"] = all(
        projection_capacity_gates.values()
    )
    carrier_completeness_gates = {
        "exact_full_width_x4_behavioral": (
            carrier_oracle_behavioral["gates"]["passed"] is True  # type: ignore[index]
        ),
    }
    carrier_completeness_gates["passed"] = all(
        carrier_completeness_gates.values()
    )
    return {
        "arm": selected_arm,
        "observation_count": observation_count,
        "behavioral": behavioral,
        "behavioral_scope": {
            "token_scope": "causally_affected_supervised_tokens_only",
            "total_supervised_tokens": supervised_tokens,
            "affected_supervised_tokens": affected_supervised_tokens,
            "affected_supervised_coverage": (
                affected_supervised_tokens / supervised_tokens
            ),
            "unaffected_prefix_tokens_excluded": True,
        },
        "boundary": {
            "target_modal_width": _TARGET_MODAL_WIDTH,
            "valid_target_rows": valid_rows,
            "affected_target_rows": affected_rows,
            "valid_target_coverage": coverage,
            "pooled_target_modal_relative_error": pooled_relative,
            "pooled_target_modal_cosine": pooled_cosine,
            "worst_family_target_modal_relative_error": (
                worst_family_modal_relative
            ),
            "worst_family_target_modal_cosine": (
                worst_family_modal_cosine
            ),
            "minimum_family_source_modal_signal_l2_norm": (
                minimum_family_modal_signal
            ),
            "thresholds": {
                "pooled_target_modal_relative_error_max": (
                    _BOUNDARY_RELATIVE_ERROR_MAX
                ),
                "pooled_target_modal_cosine_min": (
                    _BOUNDARY_COSINE_MIN
                ),
                "valid_target_coverage_min": (
                    _VALID_TARGET_COVERAGE_MIN
                ),
                "worst_family_target_modal_relative_error_max": (
                    _WORST_FAMILY_MODAL_RELATIVE_ERROR_MAX
                ),
                "worst_family_target_modal_cosine_min": (
                    _WORST_FAMILY_MODAL_COSINE_MIN
                ),
                "minimum_family_source_modal_signal_l2_norm": (
                    _MINIMUM_FAMILY_SIGNAL_L2_NORM
                ),
            },
            "gates": boundary_gates,
            "family_metrics": tuple(family_metrics),
            "per_example": tuple(per_example),
        },
        "projection_capacity": {
            "target_modal_width": _TARGET_MODAL_WIDTH,
            "target_full_width": _TARGET_FULL_WIDTH,
            "pooled_full_width_delta_relative_error": (
                pooled_projection_relative
            ),
            "pooled_full_width_delta_cosine": pooled_projection_cosine,
            "worst_family_full_width_delta_relative_error": (
                worst_family_projection_relative
            ),
            "worst_family_full_width_delta_cosine": (
                worst_family_projection_cosine
            ),
            "minimum_family_source_full_width_signal_l2_norm": (
                minimum_family_projection_signal
            ),
            "thresholds": {
                "pooled_full_width_delta_relative_error_max": (
                    _PROJECTION_RELATIVE_ERROR_MAX
                ),
                "pooled_full_width_delta_cosine_min": (
                    _PROJECTION_COSINE_MIN
                ),
                "worst_family_full_width_delta_relative_error_max": (
                    _WORST_FAMILY_PROJECTION_RELATIVE_ERROR_MAX
                ),
                "worst_family_full_width_delta_cosine_min": (
                    _WORST_FAMILY_PROJECTION_COSINE_MIN
                ),
                "minimum_family_source_full_width_signal_l2_norm": (
                    _MINIMUM_FAMILY_SIGNAL_L2_NORM
                ),
            },
            "behavioral": projection_oracle_behavioral,
            "gates": projection_capacity_gates,
        },
        "carrier_completeness": {
            "carrier": "clamped_y3_source_model_reference",
            "boundary": "exact_full_width_x4",
            "interpretation": (
                "incomplete_replacement_not_isolated_boundary_fidelity"
            ),
            "behavioral": carrier_oracle_behavioral,
            "gates": carrier_completeness_gates,
        },
        "passed": (
            behavioral_passed
            and boundary_gates["passed"]
            and projection_capacity_gates["passed"]
            and carrier_completeness_gates["passed"]
        ),
    }


def _evaluate_gemma3_l3_l4_graph_organized_svd_shadow(
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    all_on_observations: Iterable[
        Gemma3L3L4GraphOrganizedSVDShadowObservation
    ],
    *,
    assessment_claim_sha256: str,
    expected_family_by_example: Mapping[str, str],
    routed_observations: Iterable[
        Gemma3L3L4GraphOrganizedSVDShadowObservation
    ]
    | None = None,
) -> dict[str, object]:
    """Evaluate a complete one-shot manifest without opening corpus files."""

    if not isinstance(
        protocol,
        Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    ):
        raise TypeError("protocol must use the strict shadow protocol type")
    protocol.validate_integrity()
    if routed_observations is not None:
        raise ValueError("routed observations are disabled in locked protocol")
    manifest, manifest_sha256 = _manifest(expected_family_by_example)
    frozen_manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    if (
        manifest != frozen_manifest
        or manifest_sha256 != _CALIBRATION_B_MANIFEST_SHA256
        or len(manifest) != _CALIBRATION_B_EXAMPLE_COUNT
        or len(set(manifest.values())) != _CALIBRATION_B_FAMILY_COUNT
    ):
        raise ValueError(
            "evaluation requires the exact full frozen Calibration-B manifest"
        )
    claim = _require_sha256(
        assessment_claim_sha256,
        label="assessment claim",
    )
    derived_claim = protocol.calibration_b_assessment_claim_sha256()
    if claim != derived_claim:
        raise ValueError("assessment claim differs from frozen derivation")
    all_on = _evaluate_arm(
        protocol,
        all_on_observations,
        arm="all_on",
        assessment_claim_sha256=claim,
        expected_family_by_example=manifest,
    )
    all_on_passed = all_on["passed"] is True
    partial_shadow_qualified = all_on_passed
    return {
        "schema": f"{_SCHEMA}.evaluation",
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.artifact_sha256,
        "assessment_claim_sha256": claim,
        "assessment_claim_identity": _assessment_claim_payload(
            protocol.artifact_sha256
        ),
        "manifest": {
            "role": "calibration_b_one_shot",
            "example_identity": "prompt_sha256_only",
            "artifact_sha256": manifest_sha256,
            "example_count": len(manifest),
            "family_count": len(set(manifest.values())),
            "complete": True,
            "matches_frozen_role": True,
            "derivation": (
                "canonical_sorted_zip_of_audit_prompt_sha256_by_role_"
                "calibration_b_and_family_file_calibration_b"
            ),
            "prompt_file_opened_by_evaluator": False,
        },
        "scope": {
            "source_path": "sequential_refit_authoritative",
            "candidate_path": (
                "reference_carrier_incomplete_replacement_metrics_only"
            ),
            "reference_provider": "clamped_y3_source_model_oracle",
            "reference_pass_oracle_fallback_required": True,
            "candidate_outputs_must_not_be_served": True,
            "candidate_logits_interpretation": (
                "incomplete_replacement_not_isolated_boundary_fidelity"
            ),
            "behavioral_token_scope": (
                "causally_affected_supervised_tokens_only"
            ),
            "prompt_text_loaded": False,
            "tokenizer_loaded": False,
            "parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
            "full_model_claim": False,
        },
        "calibration_a_development_evidence": {
            "selection_or_assessment_eligible": False,
            "deployment_authorized": False,
            "routing_authorized": False,
            "corrected_all_on_passed": False,
            "projection_capacity_passed": False,
            "carrier_completeness_passed": False,
        },
        "all_on": all_on,
        "routed": {
            "allowed": False,
            "evaluated": False,
            "reason": "locked_protocol_all_on_only",
        },
        "authorization": {
            "partial_shadow_qualified": partial_shadow_qualified,
            "partial_shadow_scope": (
                "partial_edge_reference_oracle_shadow"
                if partial_shadow_qualified
                else "none"
            ),
            "all_on_passed": all_on_passed,
            "deployment_authorized": False,
            "deployment_scope": "none",
            "routing_authorized": False,
            "routing_qualification_available": False,
            "non_authorization_reason": (
                "reference_oracle_required_and_candidate_outputs_metrics_only"
            ),
            "standalone_deployment_authorized": False,
            "full_model_deployment_authorized": False,
        },
    }
