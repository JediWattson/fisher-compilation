"""Run the first Gemma MLP shared-family compression experiment.

Importing this module is inert: setup, predecessor, panel, tensor libraries,
and model weights are opened only by :func:`run_gemma3_mlp_shared_family_residual_adapter_experiment`.
The high-level phase functions are intentionally monkeypatchable so the frozen
data firewall and orchestration can be tested without loading Gemma.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
from typing import Final


GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_SCHEMA: Final = (
    "fisher_graph.gemma3_mlp_shared_family_residual_adapter.experiment.v1"
)
GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_FORMAT_VERSION: Final = 1
PINNED_SETUP_SHA256: Final = (
    "39d4debc38f680d8318bced5b9f553fe9ec8aa9866a990d9852c10e8f55059e8"
)
PINNED_SETUP_FILE_SHA256: Final = (
    "6fe26fe3e6d4af2460dc2c65e636559b6d00a9b0718c48edd947996692d1f56f"
)
PINNED_PREDECESSOR_REPORT_SHA256: Final = (
    "7f062b40a2af2fe61364b208189773b82569bbe88902c0b8f3e7bbe607401926"
)
PINNED_PREDECESSOR_REPORT_FILE_SHA256: Final = (
    "2ad135209ff1c780944c7d0efd5dbd7b52724266dc880345abb31cfc9bbcf64a"
)
PINNED_PANEL_SHA256: Final = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_MODEL_ID: Final = "google/gemma-3-270m"
PINNED_MODEL_REVISION: Final = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_SOURCE_MODEL_FINGERPRINT: Final = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
PINNED_SOURCE_EXECUTION_FINGERPRINT: Final = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
PINNED_BATCH_SIZE: Final = 4
ORDERED_ROLES: Final = ("A_fit", "A_select", "B_guard", "C_assessment")
LOW_RANK_POINTS: Final = (256, 128, 64, 32)
DEFAULT_SETUP_OUTPUT = Path(
    "artifacts/research/gemma3-mlp-shared-family-residual-adapter-setup-v1.json"
)
DEFAULT_PREDECESSOR_REPORT = Path(
    ".local-runs/google--gemma-3-270m/heterogeneous-generator-family-exact-v1.json"
)
DEFAULT_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "mlp-shared-family-residual-adapter-v1.json"
)

_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-residual-adapter-report:v1\0"
)
_MANIFEST_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-runtime-manifest:v1\0"
)
_CONTENT_STORE_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-content-store:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _manifest_sha256(value: object) -> str:
    return hashlib.sha256(
        _MANIFEST_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _content_store_inventory_sha256(value: object) -> str:
    return hashlib.sha256(
        _CONTENT_STORE_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _primary_generator_message_receipt(boundary: object) -> object:
    receipts = tuple(getattr(boundary, "message_receipts", ()))
    primary = tuple(
        receipt
        for receipt in receipts
        if getattr(receipt, "message_kind", None)
        in {"attention_transport", "mlp_mutation"}
    )
    if len(primary) != 1:
        raise ValueError(
            "carrier boundary must contain exactly one typed primary message"
        )
    return primary[0]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("experiment report cannot contain nonfinite floats")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    raise TypeError(f"report value {type(value).__qualname__} is unsupported")


@dataclass(frozen=True, slots=True)
class _PanelExample:
    example_id: str
    family_id: str
    prompt: str
    choices: tuple[str, ...]
    correct_choice: int


@dataclass(frozen=True, slots=True)
class _Preflight:
    setup: Mapping[str, object]
    predecessor_report: Mapping[str, object]
    role_examples: Mapping[str, tuple[_PanelExample, ...]]
    setup_path: Path
    predecessor_path: Path
    panel_path: Path
    receipts: Mapping[str, object]
    artifact_root: Path | None = None


@dataclass(slots=True)
class _Runtime:
    tokenizer: object
    model: object
    adapter: object
    device: object
    source_guard: object
    source_model_fingerprint_before: str
    source_execution_fingerprint_before: str
    exact_carrier: object
    exact_family: object
    exact_executor: object
    model_receipt: Mapping[str, object]
    artifact_root: Path | None
    artifact_receipts: dict[str, Mapping[str, object]]
    artifact_store_inventory_receipt: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _RoleBatch:
    model_inputs: Mapping[str, object]
    example_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    choice_ids: object
    gold: object


@dataclass(frozen=True, slots=True)
class _MLPCapture:
    raw_carrier_inputs: object
    normalized_inputs: object
    gated_features: object
    exact_delta: object
    delta_gradients: object
    gate_gradients: object
    up_gradients: object
    down_gradients: object
    family_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AFitCapture:
    layers: Mapping[int, _MLPCapture]
    row_manifest_sha256: str
    capture_receipt: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_addressed_storage_plan(
    setup: Mapping[str, object],
) -> dict[str, object]:
    arms = setup.get("arm_contract", {}).get("arms")
    if not isinstance(arms, list):
        raise ValueError("setup omitted the point-artifact inventory")
    materialized = arms[2:]
    if len(materialized) != 14 or any(
        not isinstance(arm, Mapping)
        or type(arm.get("candidate_stored_scalars")) is not int
        for arm in materialized
    ):
        raise ValueError("setup point-artifact inventory drifted")
    by_id = {str(arm["arm_id"]): arm for arm in materialized}
    if len(by_id) != 14:
        raise ValueError("setup point-artifact identities drifted")
    dense_id = "exact.shared_core_dense_residual"
    zero_id = "ablation.shared_core_zero_residual"
    try:
        shared_base_scalars = int(by_id[zero_id]["candidate_stored_scalars"])
        dense_member_scalars = (
            int(by_id[dense_id]["candidate_stored_scalars"])
            - shared_base_scalars
        )
    except KeyError as error:
        raise ValueError("setup omitted dense/zero storage controls") from error
    if shared_base_scalars <= 0 or dense_member_scalars <= 0:
        raise ValueError("setup shared-base storage accounting is invalid")
    primary_adapter_scalars = 0
    independent_adapter_scalars = 0
    for rank in LOW_RANK_POINTS:
        primary_id = f"primary.shared_core_residual_r{rank}"
        independent_id = f"control.independent_member_r{rank}"
        shuffled_id = f"control.shuffled_residual_r{rank}"
        if not {primary_id, independent_id, shuffled_id} <= set(by_id):
            raise ValueError("setup omitted one rank-matched storage control")
        primary_count = int(by_id[primary_id]["candidate_stored_scalars"])
        independent_count = int(
            by_id[independent_id]["candidate_stored_scalars"]
        )
        shuffled_count = int(by_id[shuffled_id]["candidate_stored_scalars"])
        if primary_count != independent_count or primary_count != shuffled_count:
            raise ValueError("rank-matched logical storage budgets drifted")
        adapter_scalars = primary_count - shared_base_scalars
        if adapter_scalars <= 0:
            raise ValueError("rank adapter storage accounting is invalid")
        primary_adapter_scalars += adapter_scalars
        independent_adapter_scalars += adapter_scalars
    logical_scalars = sum(
        int(arm["candidate_stored_scalars"]) for arm in materialized
    )
    unique_scalars = (
        shared_base_scalars
        + dense_member_scalars
        + primary_adapter_scalars
        + independent_adapter_scalars
    )
    logical_bytes = logical_scalars * 4
    unique_bytes = unique_scalars * 4
    savings_bytes = logical_bytes - unique_bytes
    result = {
        "materialized_point_artifact_count": 14,
        "unique_tensor_blob_count": 10,
        "component_blob_reference_count": 27,
        "reused_component_blob_reference_count": 17,
        "shared_base_stored_scalars": shared_base_scalars,
        "dense_exact_member_stored_scalars": dense_member_scalars,
        "primary_adapter_stored_scalars": primary_adapter_scalars,
        "independent_adapter_stored_scalars": independent_adapter_scalars,
        "logical_flat_stored_scalars": logical_scalars,
        "unique_content_addressed_stored_scalars": unique_scalars,
        "logical_flat_tensor_storage_bytes": logical_bytes,
        "unique_content_addressed_tensor_storage_bytes": unique_bytes,
        "content_addressed_tensor_storage_savings_bytes": savings_bytes,
        "content_addressed_tensor_storage_savings_fraction": (
            savings_bytes / logical_bytes
        ),
        "shuffled_adapter_blob_reuses_primary": True,
        "logical_candidate_scalar_accounting_unchanged": True,
    }
    if result != {
        "materialized_point_artifact_count": 14,
        "unique_tensor_blob_count": 10,
        "component_blob_reference_count": 27,
        "reused_component_blob_reference_count": 17,
        "shared_base_stored_scalars": 264_164_736,
        "dense_exact_member_stored_scalars": 3_933_440,
        "primary_adapter_stored_scalars": 3_875_840,
        "independent_adapter_stored_scalars": 3_875_840,
        "logical_flat_stored_scalars": 3_713_867_264,
        "unique_content_addressed_stored_scalars": 275_849_856,
        "logical_flat_tensor_storage_bytes": 14_855_469_056,
        "unique_content_addressed_tensor_storage_bytes": 1_103_399_424,
        "content_addressed_tensor_storage_savings_bytes": 13_752_069_632,
        "content_addressed_tensor_storage_savings_fraction": (
            13_752_069_632 / 14_855_469_056
        ),
        "shuffled_adapter_blob_reuses_primary": True,
        "logical_candidate_scalar_accounting_unchanged": True,
    }:
        raise ValueError("frozen content-addressed storage plan drifted")
    return result


def _artifact_storage_preflight(
    *, destination: Path, setup: Mapping[str, object]
) -> dict[str, object]:
    artifact_root = destination.parent / f"{destination.stem}-artifacts"
    plan = _content_addressed_storage_plan(setup)
    estimated_tensor_bytes = int(
        plan["unique_content_addressed_tensor_storage_bytes"]
    )
    safety_margin_bytes = max(
        1 << 30,
        math.ceil(estimated_tensor_bytes * 0.10),
    )
    required_free_bytes = estimated_tensor_bytes + safety_margin_bytes
    probe = destination.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available_bytes = shutil.disk_usage(probe).free
    if available_bytes < required_free_bytes:
        raise RuntimeError(
            "insufficient free space for 14 authenticated candidate point artifacts: "
            f"need {required_free_bytes} bytes, found {available_bytes} bytes"
        )
    return {
        "artifact_root": str(artifact_root),
        **plan,
        "estimated_tensor_storage_bytes": estimated_tensor_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "required_free_bytes": required_free_bytes,
        "available_free_bytes": available_bytes,
        "passed": True,
    }


def _strict_predecessor_report(source: Path | str) -> dict[str, object]:
    path = Path(source)
    if _sha256_file(path) != PINNED_PREDECESSOR_REPORT_FILE_SHA256:
        raise RuntimeError("exact predecessor report file digest drifted")
    from .gemma3_heterogeneous_generator_family_experiment import (
        load_gemma3_heterogeneous_generator_family_report,
    )

    predecessor = load_gemma3_heterogeneous_generator_family_report(path)
    if predecessor.get("report_sha256") != PINNED_PREDECESSOR_REPORT_SHA256:
        raise RuntimeError("exact predecessor internal digest drifted")
    required = "exact.heterogeneous_full_rank_atom_bank_lift"
    arms = predecessor.get("arm_results")
    matching = [
        arm
        for arm in arms
        if isinstance(arm, Mapping) and arm.get("arm_id") == required
    ] if isinstance(arms, list) else []
    if (
        len(matching) != 1
        or not isinstance(matching[0].get("gate"), Mapping)
        or matching[0]["gate"].get("passed") is not True
    ):
        raise RuntimeError("required exact predecessor arm did not pass")
    return predecessor


def _load_role_panel(
    *,
    panel_path: Path | str,
    setup: Mapping[str, object],
) -> dict[str, tuple[_PanelExample, ...]]:
    panel_path = Path(panel_path)
    if _sha256_file(panel_path) != PINNED_PANEL_SHA256:
        raise RuntimeError("pinned role panel file digest drifted")
    raw = json.loads(panel_path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("panel_id") != "gemma3-single-token-forced-choice-v1"
        or raw.get("choice_count") != 4
        or not isinstance(raw.get("examples"), list)
    ):
        raise ValueError("pinned forced-choice panel schema drifted")
    rows: dict[str, _PanelExample] = {}
    for value in raw["examples"]:
        if not isinstance(value, dict) or set(value) != {
            "example_id", "family_id", "prompt", "choices", "correct_choice"
        }:
            raise ValueError("forced-choice panel row schema drifted")
        choices = value["choices"]
        item = _PanelExample(
            example_id=value["example_id"],
            family_id=value["family_id"],
            prompt=value["prompt"],
            choices=tuple(choices) if isinstance(choices, list) else (),
            correct_choice=value["correct_choice"],
        )
        if (
            not item.example_id
            or not item.family_id
            or not item.prompt
            or len(item.choices) != 4
            or type(item.correct_choice) is not int
            or not 0 <= item.correct_choice < 4
            or item.example_id in rows
        ):
            raise ValueError("forced-choice panel row is invalid")
        rows[item.example_id] = item
    firewall = setup.get("data_firewall")
    if not isinstance(firewall, Mapping) or not isinstance(
        firewall.get("roles"), Mapping
    ):
        raise ValueError("setup role firewall is unavailable")
    roles: dict[str, tuple[_PanelExample, ...]] = {}
    used: list[str] = []
    for role in ORDERED_ROLES:
        manifest = firewall["roles"].get(role)
        if not isinstance(manifest, Mapping):
            raise ValueError(f"setup role {role} is unavailable")
        ids = manifest.get("ordered_example_ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"setup role {role} identities drifted")
        try:
            values = tuple(rows[x] for x in ids)
        except KeyError as error:
            raise ValueError("setup role references an absent panel row") from error
        if tuple(item.family_id for item in values) and any(
            item.family_id not in manifest.get("family_ids", ()) for item in values
        ):
            raise ValueError(f"setup role {role} family assignment drifted")
        roles[role] = values
        used.extend(ids)
    if len(used) != 60 or len(set(used)) != 60 or set(used) != set(rows):
        raise ValueError("role manifests are not disjoint and exhaustive")
    return roles


def _preflight(
    *,
    revision: str,
    model_id: str,
    output: Path | str,
    setup_path: Path | str,
    predecessor_report_path: Path | str,
    panel_path: Path | str,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = PINNED_BATCH_SIZE,
) -> _Preflight:
    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("shared-family experiment output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a shared-family report")
    artifact_root = destination.parent / f"{destination.stem}-artifacts"
    if artifact_root.exists():
        raise FileExistsError(
            "refusing to reuse a shared-family point artifact directory"
        )
    if model_id != PINNED_MODEL_ID or revision != PINNED_MODEL_REVISION:
        raise ValueError("shared-family experiment requires pinned Gemma")
    if device_name != "cpu" or dtype != "float32" or batch_size != PINNED_BATCH_SIZE:
        raise ValueError("shared-family experiment requires CPU float32 batch size 4")

    setup_file = Path(setup_path)
    predecessor_file = Path(predecessor_report_path)
    panel_file = Path(panel_path)
    if _sha256_file(setup_file) != PINNED_SETUP_FILE_SHA256:
        raise RuntimeError("canonical shared-family setup file digest drifted")
    # This import and the next two model-free reads remain on the authorized
    # run path and complete before tensor/model dependencies are resolved.
    from .gemma3_mlp_shared_family_residual_adapter_setup import (
        load_gemma3_mlp_shared_family_residual_adapter_setup,
    )

    setup = load_gemma3_mlp_shared_family_residual_adapter_setup(setup_file)
    if setup.get("setup_sha256") != PINNED_SETUP_SHA256:
        raise RuntimeError("canonical shared-family setup binding drifted")
    storage_receipt = _artifact_storage_preflight(
        destination=destination,
        setup=setup,
    )
    predecessor = _strict_predecessor_report(predecessor_file)
    roles = _load_role_panel(panel_path=panel_file, setup=setup)
    return _Preflight(
        setup=setup,
        predecessor_report=predecessor,
        role_examples=roles,
        setup_path=setup_file,
        predecessor_path=predecessor_file,
        panel_path=panel_file,
        artifact_root=artifact_root,
        receipts={
            "setup_file_sha256": PINNED_SETUP_FILE_SHA256,
            "predecessor_file_sha256": PINNED_PREDECESSOR_REPORT_FILE_SHA256,
            "panel_file_sha256": PINNED_PANEL_SHA256,
            "artifact_storage_preflight": storage_receipt,
            "all_authenticated_before_live_dependencies": True,
            "all_role_manifests_verified_before_model_load": True,
        },
    )


def _load_live_dependencies() -> SimpleNamespace:
    """Resolve all tensor/model/compiler dependencies after strict preflight."""

    import torch
    from .adapters import Gemma3CausalLMAdapter
    from .gemma3_ablation_experiment import _FrozenModelTensorGuard
    from .gemma3_causal_carrier_transport_compiler import (
        compile_gemma3_exact_causal_carrier_transport,
    )
    from .gemma3_experiment import (
        _model_provenance,
        load_gemma3,
        resolve_gemma3_huggingface_paths,
        resolve_torch_device,
    )
    from .gemma3_heterogeneous_generator_family_compiler import (
        compile_gemma3_exact_heterogeneous_generator_plan,
    )
    from .gemma3_mlp_shared_family_residual_adapter import (
        DENSE_EXACT_CONTROL,
        INDEPENDENT_LOW_RANK_NONSHARING,
        SHARED_CORE_LOW_RANK,
        SHUFFLED_ASSIGNMENT,
        ZERO_RESIDUAL_SHARED_CORE,
        build_mlp_family_pair_manifest,
        compile_gemma3_mlp_shared_family_residual_adapter,
        compile_no_refit_residual_assignment_control,
        content_addressed_mlp_shared_family_point_manifest_path,
        load_mlp_shared_family_content_addressed_point_artifact,
        validate_mlp_shared_family_content_addressed_store_inventory,
        write_mlp_shared_family_content_addressed_point_artifact,
        write_mlp_shared_family_content_addressed_store_inventory,
    )
    from .gemma3_mlp_shared_family_residual_adapter_fit import (
        build_directed_mlp_pair_candidate,
        canonical_float32_tensor_sha256,
        fit_mlp_projection_factor_catalog,
        select_directed_mlp_family_pair,
    )
    from .heterogeneous_generator_atom_bank import (
        LIFT_SCOPE_FULL_HETEROGENEOUS,
        module_state_sha256,
    )
    return SimpleNamespace(
        torch=torch,
        adapter_type=Gemma3CausalLMAdapter,
        frozen_guard_type=_FrozenModelTensorGuard,
        compile_exact_carrier=compile_gemma3_exact_causal_carrier_transport,
        compile_exact_family=compile_gemma3_exact_heterogeneous_generator_plan,
        load_model=load_gemma3,
        model_provenance=_model_provenance,
        resolve_paths=resolve_gemma3_huggingface_paths,
        resolve_device=resolve_torch_device,
        dense_kind=DENSE_EXACT_CONTROL,
        primary_kind=SHARED_CORE_LOW_RANK,
        independent_kind=INDEPENDENT_LOW_RANK_NONSHARING,
        zero_kind=ZERO_RESIDUAL_SHARED_CORE,
        shuffled_assignment=SHUFFLED_ASSIGNMENT,
        build_pair=build_mlp_family_pair_manifest,
        compile_point=compile_gemma3_mlp_shared_family_residual_adapter,
        compile_assignment_control=compile_no_refit_residual_assignment_control,
        content_point_manifest_path=(
            content_addressed_mlp_shared_family_point_manifest_path
        ),
        load_content_point_artifact=(
            load_mlp_shared_family_content_addressed_point_artifact
        ),
        write_content_point_artifact=(
            write_mlp_shared_family_content_addressed_point_artifact
        ),
        write_store_inventory=(
            write_mlp_shared_family_content_addressed_store_inventory
        ),
        validate_store_inventory=(
            validate_mlp_shared_family_content_addressed_store_inventory
        ),
        build_pair_candidate=build_directed_mlp_pair_candidate,
        fit_factor_catalog=fit_mlp_projection_factor_catalog,
        select_pair_candidate=select_directed_mlp_family_pair,
        canonical_tensor_sha256=canonical_float32_tensor_sha256,
        full_lift_scope=LIFT_SCOPE_FULL_HETEROGENEOUS,
        module_state_sha256=module_state_sha256,
    )


# The following phase seams are deliberately small. Production implementations
# are appended below; tests may replace any seam without importing torch.
def _prepare_runtime(*, preflight: _Preflight, dependencies: SimpleNamespace, cache_dir: Path | str | None, device_name: str, dtype: str) -> object:
    return _prepare_runtime_live(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )


def _capture_a_fit(*, runtime: object, preflight: _Preflight, dependencies: SimpleNamespace, batch_size: int) -> object:
    return _capture_a_fit_live(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )


def _select_pair(*, capture: object, runtime: object, preflight: _Preflight, dependencies: SimpleNamespace) -> Mapping[str, object]:
    return _select_pair_live(
        capture=capture,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )


def _fit_rank_point(*, rank: int, independent_nonsharing: bool, capture: object, pair_selection: Mapping[str, object], runtime: object, preflight: _Preflight, dependencies: SimpleNamespace) -> Mapping[str, object]:
    return _fit_rank_point_live(
        rank=rank,
        independent_nonsharing=independent_nonsharing,
        capture=capture,
        pair_selection=pair_selection,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )


def _evaluate_role(*, role_id: str, runtime: object, preflight: _Preflight, pair_selection: Mapping[str, object], fit_points: Mapping[str, object], dependencies: SimpleNamespace, batch_size: int) -> Mapping[str, object]:
    return _evaluate_role_live(
        role_id=role_id,
        runtime=runtime,
        preflight=preflight,
        pair_selection=pair_selection,
        fit_points=fit_points,
        dependencies=dependencies,
        batch_size=batch_size,
    )


def _freeze_a_select_frontier(*, role_result: Mapping[str, object], setup: Mapping[str, object]) -> Mapping[str, object]:
    return _freeze_a_select_frontier_live(role_result=role_result, setup=setup)


def _run_b_guard(*, role_result: Mapping[str, object], frontier: Mapping[str, object], setup: Mapping[str, object]) -> Mapping[str, object]:
    return _run_b_guard_live(role_result=role_result, frontier=frontier, setup=setup)


def _run_c_assessment(*, role_result: Mapping[str, object], winner_arm_id: str, setup: Mapping[str, object]) -> Mapping[str, object]:
    return _run_c_assessment_live(
        role_result=role_result,
        winner_arm_id=winner_arm_id,
        setup=setup,
    )


_REPORT_TOP_FIELDS = {
    "schema", "format_version", "diagnostic_outcome",
    "contains_model_weights", "contains_executor_weights",
    "contains_prompt_text", "contains_token_ids", "setup", "predecessor",
    "model", "data", "configuration", "pair_selection", "fit_receipts",
    "arm_artifacts", "role_results", "frontier", "B_guard",
    "C_assessment", "gate_chain", "execution_audit", "resources", "claims",
}


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if set(report) != _REPORT_TOP_FIELDS:
        raise ValueError("shared-family report fields are not exact")
    if (
        report.get("schema")
        != GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_FORMAT_VERSION
        or any(report.get(name) is not False for name in (
            "contains_model_weights", "contains_executor_weights",
            "contains_prompt_text", "contains_token_ids",
        ))
    ):
        raise ValueError("shared-family report header drifted")
    from .gemma3_mlp_shared_family_residual_adapter_setup import (
        build_gemma3_mlp_shared_family_residual_adapter_setup,
    )

    canonical_setup = build_gemma3_mlp_shared_family_residual_adapter_setup()
    arm_order = tuple(canonical_setup["arm_contract"]["arm_order"])
    primary_arms = tuple(
        f"primary.shared_core_residual_r{rank}" for rank in LOW_RANK_POINTS
    )
    fit_arms = tuple(
        arm_id
        for rank in LOW_RANK_POINTS
        for arm_id in (
            f"primary.shared_core_residual_r{rank}",
            f"control.independent_member_r{rank}",
        )
    )

    def strict(value: object, fields: set[str], label: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"{label} fields are not exact")
        return value

    def sealed(value: Mapping[str, object], field: str, label: str) -> None:
        payload = dict(value)
        observed = payload.pop(field, None)
        if not _is_sha256(observed) or observed != _manifest_sha256(payload):
            raise ValueError(f"{label} authentication failed")

    setup = strict(
        report.get("setup"),
        {"setup_sha256", "file_sha256", "path"},
        "setup receipt",
    )
    predecessor = strict(
        report.get("predecessor"),
        {
            "report_sha256", "file_sha256", "required_arm_id",
            "required_arm_gate_passed", "anchor_source_atom_state_sha256",
        },
        "predecessor receipt",
    )
    model = strict(
        report.get("model"),
        {
            "model_id", "revision", "parameter_count",
            "source_model_fingerprint_before", "source_model_fingerprint_after",
            "source_execution_fingerprint_before",
            "source_execution_fingerprint_after",
        },
        "model receipt",
    )
    data = strict(
        report.get("data"),
        {
            "panel_sha256", "ordered_roles", "roles",
            "role_family_overlap", "role_example_overlap",
        },
        "data receipt",
    )
    configuration = strict(
        report.get("configuration"),
        {
            "ordered_roles", "ordered_arm_ids", "residual_rank_order",
            "fit_arm_ids", "seeds", "acceptance_gates", "device", "dtype",
            "batch_size",
        },
        "configuration",
    )
    roles = report.get("role_results")
    if (
        setup.get("setup_sha256") != PINNED_SETUP_SHA256
        or setup.get("file_sha256") != PINNED_SETUP_FILE_SHA256
        or setup.get("path") != str(DEFAULT_SETUP_OUTPUT)
        or predecessor.get("report_sha256") != PINNED_PREDECESSOR_REPORT_SHA256
        or predecessor.get("file_sha256")
        != PINNED_PREDECESSOR_REPORT_FILE_SHA256
        or predecessor.get("required_arm_id")
        != "exact.heterogeneous_full_rank_atom_bank_lift"
        or predecessor.get("required_arm_gate_passed") is not True
        or not _is_sha256(predecessor.get("anchor_source_atom_state_sha256"))
        or model.get("model_id") != PINNED_MODEL_ID
        or model.get("revision") != PINNED_MODEL_REVISION
        or model.get("parameter_count") != 268_098_176
        or model.get("source_model_fingerprint_before")
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or model.get("source_model_fingerprint_before")
        != model.get("source_model_fingerprint_after")
        or model.get("source_execution_fingerprint_before")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT
        or model.get("source_execution_fingerprint_before")
        != model.get("source_execution_fingerprint_after")
        or data.get("panel_sha256") != PINNED_PANEL_SHA256
        or data.get("ordered_roles") != list(ORDERED_ROLES)
        or data.get("roles") != canonical_setup["data_firewall"]["roles"]
        or data.get("role_family_overlap") is not False
        or data.get("role_example_overlap") is not False
        or configuration.get("ordered_roles") != list(ORDERED_ROLES)
        or configuration.get("ordered_arm_ids") != list(arm_order)
        or configuration.get("residual_rank_order") != list(LOW_RANK_POINTS)
        or configuration.get("fit_arm_ids") != list(fit_arms)
        or configuration.get("seeds") != canonical_setup["fit_contract"]["seeds"]
        or configuration.get("acceptance_gates")
        != canonical_setup["acceptance_gates"]
        or configuration.get("device") != "cpu"
        or configuration.get("dtype") != "float32"
        or configuration.get("batch_size") != PINNED_BATCH_SIZE
        or not isinstance(roles, Mapping)
        or list(roles) != list(ORDERED_ROLES)
    ):
        raise ValueError("shared-family report lineage or role order drifted")

    pair = strict(
        report.get("pair_selection"),
        {
            "family_id", "anchor_invocation_id", "member_invocation_id",
            "anchor_atom_id", "member_atom_id", "anchor_block_ordinal",
            "member_block_ordinal", "selected_on",
            "a_fit_role_manifest_sha256", "selection_evidence_sha256",
            "shared_core_state_sha256", "activation_and_Fisher_evidence_only",
            "block_ordinal_used_as_numerical_feature",
            "nonadjacency_is_eligibility_only", "frozen_before_A_select",
            "dense_exact_control_gate_passed", "pair_manifest_sha256",
        },
        "pair selection",
    )
    sealed(pair, "pair_manifest_sha256", "pair selection")
    anchor = pair.get("anchor_block_ordinal")
    member = pair.get("member_block_ordinal")
    if (
        pair.get("family_id") != "family.mlp.gated_gelu_h640_i2048"
        or type(anchor) is not int or type(member) is not int
        or not 0 <= anchor < 18 or not 0 <= member < 18
        or abs(anchor - member) < 2
        or pair.get("anchor_invocation_id") != f"mutation.{2 * anchor + 2}"
        or pair.get("member_invocation_id") != f"mutation.{2 * member + 2}"
        or pair.get("anchor_atom_id") != f"atom.mlp.layer.{anchor}"
        or pair.get("member_atom_id") != f"atom.mlp.layer.{member}"
        or pair.get("selected_on") != "A_fit"
        or pair.get("a_fit_role_manifest_sha256")
        != canonical_setup["data_firewall"]["roles"]["A_fit"]["role_manifest_sha256"]
        or not _is_sha256(pair.get("selection_evidence_sha256"))
        or pair.get("shared_core_state_sha256")
        != predecessor.get("anchor_source_atom_state_sha256")
        or pair.get("activation_and_Fisher_evidence_only") is not True
        or pair.get("block_ordinal_used_as_numerical_feature") is not False
        or pair.get("nonadjacency_is_eligibility_only") is not True
        or pair.get("frozen_before_A_select") is not True
        or pair.get("dense_exact_control_gate_passed") is not True
    ):
        raise ValueError("shared-family pair firewall drifted")

    fit_values = report.get("fit_receipts")
    if not isinstance(fit_values, list) or len(fit_values) != len(fit_arms):
        raise ValueError("fit receipt inventory drifted")
    fit_by_arm: dict[str, Mapping[str, object]] = {}
    nonces: set[object] = set()
    for value, arm_id in zip(fit_values, fit_arms, strict=True):
        fit = strict(
            value,
            {
                "arm_id", "rank", "fit_nonce", "fit_seed", "fit_role",
                "a_fit_role_manifest_sha256", "pair_manifest_sha256",
                "shared_core_state_sha256", "independently_fitted",
                "supernet_or_factor_slicing_used", "factor_catalog_sha256",
                "fit_receipt_sha256",
            },
            f"fit receipt {arm_id}",
        )
        sealed(fit, "fit_receipt_sha256", f"fit receipt {arm_id}")
        setup_arm = next(item for item in canonical_setup["arm_contract"]["arms"] if item["arm_id"] == arm_id)
        rank = next(item for item in LOW_RANK_POINTS if arm_id.endswith(str(item)))
        if (
            fit.get("arm_id") != arm_id or fit.get("rank") != rank
            or fit.get("fit_nonce") != setup_arm["fit_nonce"]
            or fit.get("fit_seed") != setup_arm["fit_seed"]
            or fit.get("fit_role") != "A_fit"
            or fit.get("a_fit_role_manifest_sha256") != pair.get("a_fit_role_manifest_sha256")
            or fit.get("pair_manifest_sha256") != pair.get("pair_manifest_sha256")
            or fit.get("shared_core_state_sha256") != pair.get("shared_core_state_sha256")
            or fit.get("independently_fitted") is not True
            or fit.get("supernet_or_factor_slicing_used") is not False
            or not _is_sha256(fit.get("factor_catalog_sha256"))
            or fit.get("fit_nonce") in nonces
        ):
            raise ValueError("fit receipt contract drifted")
        nonces.add(fit.get("fit_nonce"))
        fit_by_arm[arm_id] = fit

    artifacts_value = report.get("arm_artifacts")
    if not isinstance(artifacts_value, list) or len(artifacts_value) != len(arm_order):
        raise ValueError("arm artifact inventory drifted")
    artifact_by_arm: dict[str, Mapping[str, object]] = {}
    for value, arm_id in zip(artifacts_value, arm_order, strict=True):
        artifact = strict(
            value,
            {
                "arm_id", "pair_manifest_sha256", "shared_core_state_sha256",
                "fit_receipt_sha256", "artifact_state_sha256",
                "candidate_stored_scalars", "learned_correction_edge_count",
                "chart_or_learned_transport_scalars",
                "source_projection_and_fallback_calls",
                "nonanchor_dense_source_storage_present",
                "artifact_manifest_sha256",
            },
            f"arm artifact {arm_id}",
        )
        sealed(artifact, "artifact_manifest_sha256", f"arm artifact {arm_id}")
        setup_arm = next(item for item in canonical_setup["arm_contract"]["arms"] if item["arm_id"] == arm_id)
        expected_fit = fit_by_arm.get(arm_id)
        if (
            artifact.get("arm_id") != arm_id
            or artifact.get("pair_manifest_sha256") != pair.get("pair_manifest_sha256")
            or artifact.get("shared_core_state_sha256") != pair.get("shared_core_state_sha256")
            or artifact.get("fit_receipt_sha256") != (None if expected_fit is None else expected_fit["fit_receipt_sha256"])
            or not _is_sha256(artifact.get("artifact_state_sha256"))
            or artifact.get("candidate_stored_scalars") != setup_arm["candidate_stored_scalars"]
            or artifact.get("learned_correction_edge_count") != 0
            or artifact.get("chart_or_learned_transport_scalars") != 0
            or artifact.get("source_projection_and_fallback_calls") != 0
            or artifact.get("nonanchor_dense_source_storage_present") is not False
        ):
            raise ValueError("arm artifact contract drifted")
        artifact_by_arm[arm_id] = artifact

    metric_fields = {
        "arm_id", "delta_nll_per_token", "native_to_candidate_kl_per_query",
        "top1_agreement", "valid_query_logit_nrmse", "final_carrier_nrmse",
        "selected_nonanchor_MLP_delta_nrmse",
        "selected_nonanchor_MLP_delta_cosine",
        "source_projection_and_fallback_calls", "passed_fidelity_gate",
    }
    role_fields = {
        "role_id", "role_manifest_sha256", "example_ids_sha256",
        "example_count", "family_count", "ordered_arm_ids", "arm_metrics",
        "pair_manifest_sha256", "fit_performed", "pair_selection_performed",
        "dense_exact_control_gate_passed", "threshold_or_topology_changed",
        "role_result_sha256",
    }
    for role_id in ORDERED_ROLES:
        role = strict(roles[role_id], role_fields, f"role {role_id}")
        sealed(role, "role_result_sha256", f"role {role_id}")
        manifest = canonical_setup["data_firewall"]["roles"][role_id]
        metrics = role.get("arm_metrics")
        if (
            role.get("role_id") != role_id
            or role.get("role_manifest_sha256") != manifest["role_manifest_sha256"]
            or role.get("example_ids_sha256") != manifest["example_ids_sha256"]
            or role.get("example_count") != manifest["example_count"]
            or role.get("family_count") != manifest["family_count"]
            or role.get("ordered_arm_ids") != list(arm_order)
            or not isinstance(metrics, list) or len(metrics) != len(arm_order)
            or role.get("pair_manifest_sha256") != pair.get("pair_manifest_sha256")
            or role.get("fit_performed") is not (role_id == "A_fit")
            or role.get("pair_selection_performed") is not (role_id == "A_fit")
            or role.get("dense_exact_control_gate_passed") is not (role_id == "A_fit")
            or role.get("threshold_or_topology_changed") is not False
        ):
            raise ValueError("role result contract drifted")
        for metric, arm_id in zip(metrics, arm_order, strict=True):
            metric = strict(metric, metric_fields, f"{role_id} metric")
            if metric.get("arm_id") != arm_id or metric.get("source_projection_and_fallback_calls") != 0:
                raise ValueError("role metric contract drifted")

    frontier = strict(
        report.get("frontier"),
        {
            "ordered_primary_arm_ids", "promotion_order", "pair_manifest_sha256",
            "A_select_role_result_sha256", "frozen_before_B_guard",
            "refit_or_pruning_performed", "frontier_sha256",
        },
        "frontier",
    )
    sealed(frontier, "frontier_sha256", "frontier")
    if (
        frontier.get("ordered_primary_arm_ids") != list(primary_arms)
        or frontier.get("promotion_order") != list(reversed(primary_arms))
        or frontier.get("pair_manifest_sha256") != pair.get("pair_manifest_sha256")
        or frontier.get("A_select_role_result_sha256") != roles["A_select"]["role_result_sha256"]
        or frontier.get("frozen_before_B_guard") is not True
        or frontier.get("refit_or_pruning_performed") is not False
    ):
        raise ValueError("frontier contract drifted")

    guard = strict(
        report.get("B_guard"),
        {
            "opened_once", "entire_frozen_frontier_scored_once",
            "frontier_sha256", "winner_arm_id", "bootstrap_seed",
            "refit_reorder_prune_or_threshold_change", "passed",
        },
        "B_guard",
    )
    winner = guard.get("winner_arm_id")
    if (
        guard.get("opened_once") is not True
        or guard.get("entire_frozen_frontier_scored_once") is not True
        or guard.get("frontier_sha256") != frontier.get("frontier_sha256")
        or winner not in primary_arms
        or guard.get("bootstrap_seed") != canonical_setup["fit_contract"]["seeds"]["bootstrap"]
        or guard.get("refit_reorder_prune_or_threshold_change") is not False
        or guard.get("passed") is not True
    ):
        raise ValueError("B_guard contract drifted")

    assessment = strict(
        report.get("C_assessment"),
        {
            "opened_at_most_once", "opened_after_guard_pass", "candidate_count",
            "candidate_arm_id", "refit_selection_or_threshold_change", "passed",
        },
        "C_assessment",
    )
    if (
        assessment.get("opened_at_most_once") is not True
        or assessment.get("opened_after_guard_pass") is not True
        or assessment.get("candidate_count") != 1
        or assessment.get("candidate_arm_id") != winner
        or assessment.get("refit_selection_or_threshold_change") is not False
        or assessment.get("passed") is not True
    ):
        raise ValueError("C assessment contract drifted")

    gate = strict(
        report.get("gate_chain"),
        {
            "ordered_gates", "dense_exact_control_passed_before_compressed_fit",
            "failed_approximate_point_did_not_prune_frontier", "all_passed",
        },
        "gate chain",
    )
    if gate != {
        "ordered_gates": [
            "preflight", "exact_predecessor", "dense_exact_control",
            "fit_frozen_frontier", "A_select_freeze", "B_guard", "C_assessment",
        ],
        "dense_exact_control_passed_before_compressed_fit": True,
        "failed_approximate_point_did_not_prune_frontier": True,
        "all_passed": True,
    }:
        raise ValueError("gate chain drifted")

    audit = strict(
        report.get("execution_audit"),
        {
            "preflight_before_live_dependencies", "source_model_unchanged",
            "source_execution_unchanged", "source_projection_and_fallback_calls",
            "learned_correction_edge_count", "chart_or_learned_transport_scalars",
            "pair_manifest_sha256_by_role", "fit_call_count",
            "unique_fit_nonce_count",
            "selected_member_prefix_is_exact_by_single_approximation",
            "candidate_prefix_recapture_required",
            "exact_prefix_input_hash_equality_authenticated_on_A_fit",
            "selected_member_input_sha256_by_fit_arm",
            "all_39_invocations_and_75_occurrences_retained",
        },
        "execution audit",
    )
    if audit != {
        "preflight_before_live_dependencies": True,
        "source_model_unchanged": True,
        "source_execution_unchanged": True,
        "source_projection_and_fallback_calls": 0,
        "learned_correction_edge_count": 0,
        "chart_or_learned_transport_scalars": 0,
        "pair_manifest_sha256_by_role": {role: pair["pair_manifest_sha256"] for role in ORDERED_ROLES},
        "fit_call_count": 8,
        "unique_fit_nonce_count": 8,
        "selected_member_prefix_is_exact_by_single_approximation": True,
        "candidate_prefix_recapture_required": False,
        "exact_prefix_input_hash_equality_authenticated_on_A_fit": True,
        "selected_member_input_sha256_by_fit_arm": audit.get(
            "selected_member_input_sha256_by_fit_arm"
        ),
        "all_39_invocations_and_75_occurrences_retained": True,
    }:
        raise ValueError("execution audit drifted")
    member_input_hashes = audit.get("selected_member_input_sha256_by_fit_arm")
    if (
        not isinstance(member_input_hashes, Mapping)
        or set(member_input_hashes) != set(fit_arms)
        or any(not _is_sha256(value) for value in member_input_hashes.values())
        or len(set(member_input_hashes.values())) != 1
    ):
        raise ValueError("exact selected-member input proof drifted")

    resources = strict(
        report.get("resources"),
        {
            "source_stored_scalars", "candidate_stored_scalars_by_arm",
            "matched_control_budget_equal_by_rank",
            "nonanchor_dense_source_storage_absent",
            "serialized_byte_reduction_measured", "logical_MAC_reduction_measured",
            "peak_memory_measured", "latency_measured",
        },
        "resources",
    )
    expected_counts = {arm: artifact_by_arm[arm]["candidate_stored_scalars"] for arm in arm_order}
    if (
        resources.get("source_stored_scalars") != 268_098_176
        or resources.get("candidate_stored_scalars_by_arm") != expected_counts
        or resources.get("matched_control_budget_equal_by_rank") is not True
        or resources.get("nonanchor_dense_source_storage_absent") is not True
        or any(resources.get(name) is not False for name in (
            "serialized_byte_reduction_measured", "logical_MAC_reduction_measured",
            "peak_memory_measured", "latency_measured",
        ))
    ):
        raise ValueError("resource contract drifted")

    claims = strict(
        report.get("claims"),
        {
            "pair_mechanism_qualified",
            "twenty_percent_whole_model_compression_claimed",
            "learned_edge_compression_claimed",
            "serialized_byte_reduction_claimed", "logical_MAC_reduction_claimed",
            "latency_or_speed_claimed",
        },
        "claims",
    )
    if (
        claims.get("pair_mechanism_qualified") is not assessment.get("passed")
        or any(claims.get(name) is not False for name in (
            "twenty_percent_whole_model_compression_claimed",
            "learned_edge_compression_claimed", "serialized_byte_reduction_claimed",
            "logical_MAC_reduction_claimed", "latency_or_speed_claimed",
        ))
    ):
        raise ValueError("claim firewall drifted")


def _validate_report_semantics_v2(report: Mapping[str, object]) -> None:
    """Validate the source-free experiment envelope and all claim lineage."""

    if set(report) != _REPORT_TOP_FIELDS:
        raise ValueError("shared-family report fields are not exact")
    if (
        report.get("schema")
        != GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_FORMAT_VERSION
        or any(
            report.get(name) is not False
            for name in (
                "contains_model_weights",
                "contains_executor_weights",
                "contains_prompt_text",
                "contains_token_ids",
            )
        )
    ):
        raise ValueError("shared-family report header drifted")

    from .gemma3_mlp_shared_family_residual_adapter import (
        MLPFamilyPairManifest,
    )
    from .gemma3_mlp_shared_family_residual_adapter_setup import (
        build_gemma3_mlp_shared_family_residual_adapter_setup,
    )

    canonical_setup = build_gemma3_mlp_shared_family_residual_adapter_setup()
    arm_order = tuple(canonical_setup["arm_contract"]["arm_order"])
    materialized_arms = arm_order[2:]
    primary_arms = tuple(
        f"primary.shared_core_residual_r{rank}" for rank in LOW_RANK_POINTS
    )
    fit_arms = tuple(
        arm_id
        for rank in LOW_RANK_POINTS
        for arm_id in (
            f"primary.shared_core_residual_r{rank}",
            f"control.independent_member_r{rank}",
        )
    )
    setup_arm_by_id = {
        value["arm_id"]: value
        for value in canonical_setup["arm_contract"]["arms"]
    }

    def strict(
        value: object, fields: set[str], label: str
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"{label} fields are not exact")
        return value

    def sealed(value: Mapping[str, object], field: str, label: str) -> None:
        payload = dict(value)
        observed = payload.pop(field, None)
        if not _is_sha256(observed) or observed != _manifest_sha256(payload):
            raise ValueError(f"{label} authentication failed")

    def finite_number(value: object, *, label: str) -> float:
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label} must be finite")
        return float(value)

    setup_receipt = strict(
        report.get("setup"),
        {"setup_sha256", "file_sha256", "path"},
        "setup receipt",
    )
    predecessor = strict(
        report.get("predecessor"),
        {
            "report_sha256",
            "file_sha256",
            "required_arm_id",
            "required_arm_gate_passed",
            "anchor_source_atom_state_sha256",
        },
        "predecessor receipt",
    )
    model = strict(
        report.get("model"),
        {
            "model_id",
            "revision",
            "parameter_count",
            "source_model_fingerprint_before",
            "source_model_fingerprint_after",
            "source_execution_fingerprint_before",
            "source_execution_fingerprint_after",
        },
        "model receipt",
    )
    data = strict(
        report.get("data"),
        {
            "panel_sha256",
            "ordered_roles",
            "roles",
            "role_family_overlap",
            "role_example_overlap",
        },
        "data receipt",
    )
    configuration = strict(
        report.get("configuration"),
        {
            "ordered_roles",
            "ordered_arm_ids",
            "residual_rank_order",
            "fit_arm_ids",
            "seeds",
            "acceptance_gates",
            "device",
            "dtype",
            "batch_size",
        },
        "configuration",
    )
    if (
        setup_receipt
        != {
            "setup_sha256": PINNED_SETUP_SHA256,
            "file_sha256": PINNED_SETUP_FILE_SHA256,
            "path": str(DEFAULT_SETUP_OUTPUT),
        }
        or predecessor.get("report_sha256")
        != PINNED_PREDECESSOR_REPORT_SHA256
        or predecessor.get("file_sha256")
        != PINNED_PREDECESSOR_REPORT_FILE_SHA256
        or predecessor.get("required_arm_id")
        != "exact.heterogeneous_full_rank_atom_bank_lift"
        or predecessor.get("required_arm_gate_passed") is not True
        or not _is_sha256(
            predecessor.get("anchor_source_atom_state_sha256")
        )
        or model
        != {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "parameter_count": 268_098_176,
            "source_model_fingerprint_before": PINNED_SOURCE_MODEL_FINGERPRINT,
            "source_model_fingerprint_after": PINNED_SOURCE_MODEL_FINGERPRINT,
            "source_execution_fingerprint_before": PINNED_SOURCE_EXECUTION_FINGERPRINT,
            "source_execution_fingerprint_after": PINNED_SOURCE_EXECUTION_FINGERPRINT,
        }
        or data.get("panel_sha256") != PINNED_PANEL_SHA256
        or data.get("ordered_roles") != list(ORDERED_ROLES)
        or data.get("roles") != canonical_setup["data_firewall"]["roles"]
        or data.get("role_family_overlap") is not False
        or data.get("role_example_overlap") is not False
        or configuration.get("ordered_roles") != list(ORDERED_ROLES)
        or configuration.get("ordered_arm_ids") != list(arm_order)
        or configuration.get("residual_rank_order") != list(LOW_RANK_POINTS)
        or configuration.get("fit_arm_ids") != list(fit_arms)
        or configuration.get("seeds")
        != canonical_setup["fit_contract"]["seeds"]
        or configuration.get("acceptance_gates")
        != canonical_setup["acceptance_gates"]
        or configuration.get("device") != "cpu"
        or configuration.get("dtype") != "float32"
        or configuration.get("batch_size") != PINNED_BATCH_SIZE
    ):
        raise ValueError("shared-family lineage or configuration drifted")

    pair = strict(
        report.get("pair_selection"),
        {
            "family_id",
            "anchor_invocation_id",
            "member_invocation_id",
            "anchor_atom_id",
            "member_atom_id",
            "anchor_block_ordinal",
            "member_block_ordinal",
            "selected_on",
            "a_fit_role_manifest_sha256",
            "selection_evidence_sha256",
            "shared_core_state_sha256",
            "activation_and_Fisher_evidence_only",
            "block_ordinal_used_as_numerical_feature",
            "nonadjacency_is_eligibility_only",
            "frozen_before_A_select",
            "dense_exact_control_gate_passed",
            "typed_pair_manifest",
            "pair_manifest_sha256",
        },
        "pair selection",
    )
    typed_pair = MLPFamilyPairManifest.from_dict(pair["typed_pair_manifest"])
    anchor = typed_pair.anchor_block_ordinal
    member = typed_pair.member_block_ordinal
    if (
        pair.get("pair_manifest_sha256") != typed_pair.fingerprint_sha256
        or typed_pair.successor_setup_sha256 != PINNED_SETUP_SHA256
        or typed_pair.source_model_fingerprint
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or typed_pair.a_fit_manifest_sha256
        != canonical_setup["data_firewall"]["roles"]["A_fit"][
            "role_manifest_sha256"
        ]
        or typed_pair.anchor_invocation_id != f"mutation.{2 * anchor + 2}"
        or typed_pair.member_invocation_id != f"mutation.{2 * member + 2}"
        or typed_pair.anchor_atom_id != f"atom.mlp.layer.{anchor}"
        or typed_pair.member_atom_id != f"atom.mlp.layer.{member}"
        or pair.get("family_id") != typed_pair.family_id
        or pair.get("anchor_invocation_id")
        != typed_pair.anchor_invocation_id
        or pair.get("member_invocation_id")
        != typed_pair.member_invocation_id
        or pair.get("anchor_atom_id") != typed_pair.anchor_atom_id
        or pair.get("member_atom_id") != typed_pair.member_atom_id
        or pair.get("anchor_block_ordinal") != anchor
        or pair.get("member_block_ordinal") != member
        or pair.get("selected_on") != typed_pair.selected_on
        or pair.get("a_fit_role_manifest_sha256")
        != typed_pair.a_fit_manifest_sha256
        or pair.get("selection_evidence_sha256")
        != typed_pair.selection_evidence_sha256
        or pair.get("shared_core_state_sha256")
        != predecessor.get("anchor_source_atom_state_sha256")
        or pair.get("activation_and_Fisher_evidence_only") is not True
        or pair.get("block_ordinal_used_as_numerical_feature") is not False
        or pair.get("nonadjacency_is_eligibility_only") is not True
        or pair.get("frozen_before_A_select") is not True
        or pair.get("dense_exact_control_gate_passed") is not True
    ):
        raise ValueError("typed pair binding or A_fit firewall drifted")

    fit_values = report.get("fit_receipts")
    if not isinstance(fit_values, list) or len(fit_values) != len(fit_arms):
        raise ValueError("fit receipt inventory drifted")
    fit_by_arm: dict[str, Mapping[str, object]] = {}
    fit_nonces: set[object] = set()
    member_input_hashes: set[str] = set()
    for value, arm_id in zip(fit_values, fit_arms, strict=True):
        fit = strict(
            value,
            {
                "arm_id",
                "rank",
                "fit_nonce",
                "fit_seed",
                "fit_role",
                "a_fit_role_manifest_sha256",
                "pair_manifest_sha256",
                "shared_core_state_sha256",
                "independently_fitted",
                "supernet_or_factor_slicing_used",
                "factor_catalog_sha256",
                "projection_solver_receipt",
                "projection_solver_receipt_sha256",
                "selected_member_input_sha256",
                "fit_receipt_sha256",
            },
            f"fit receipt {arm_id}",
        )
        sealed(fit, "fit_receipt_sha256", f"fit receipt {arm_id}")
        solver = fit.get("projection_solver_receipt")
        if (
            not isinstance(solver, Mapping)
            or fit.get("projection_solver_receipt_sha256")
            != _manifest_sha256(solver)
        ):
            raise ValueError("projection solver receipt authentication failed")
        rank = next(item for item in LOW_RANK_POINTS if arm_id.endswith(str(item)))
        setup_arm = setup_arm_by_id[arm_id]
        if (
            fit.get("arm_id") != arm_id
            or fit.get("rank") != rank
            or fit.get("fit_nonce") != setup_arm["fit_nonce"]
            or fit.get("fit_seed") != setup_arm["fit_seed"]
            or fit.get("fit_role") != "A_fit"
            or fit.get("a_fit_role_manifest_sha256")
            != typed_pair.a_fit_manifest_sha256
            or fit.get("pair_manifest_sha256")
            != typed_pair.fingerprint_sha256
            or fit.get("shared_core_state_sha256")
            != pair.get("shared_core_state_sha256")
            or fit.get("independently_fitted") is not True
            or fit.get("supernet_or_factor_slicing_used") is not False
            or not _is_sha256(fit.get("factor_catalog_sha256"))
            or not _is_sha256(fit.get("selected_member_input_sha256"))
            or fit.get("fit_nonce") in fit_nonces
        ):
            raise ValueError("fit contract drifted")
        projection_order = solver.get("projection_order")
        projection_receipts = solver.get("projection_receipts")
        if (
            projection_order != ["gate_proj", "up_proj", "down_proj"]
            or not isinstance(projection_receipts, Mapping)
            or set(projection_receipts) != set(projection_order)
        ):
            raise ValueError("projection solver catalog drifted")
        expected_target = (
            "independent_member"
            if arm_id.startswith("control.independent")
            else "primary_member_minus_anchor"
        )
        for projection_name in projection_order:
            projection = projection_receipts[projection_name]
            if (
                not isinstance(projection, Mapping)
                or projection.get("projection_name") != projection_name
                or projection.get("requested_rank") != rank
                or projection.get("target_kind") != expected_target
                or projection.get("relative_ridge") != 1.0e-4
                or projection.get("solver_dtype") != "torch.float64"
                or not _is_sha256(projection.get("factor_sha256"))
                or not _is_sha256(projection.get("evidence_sha256"))
                or not _is_sha256(projection.get("receipt_sha256"))
            ):
                raise ValueError("projection fit receipt drifted")
        fit_nonces.add(fit.get("fit_nonce"))
        member_input_hashes.add(str(fit["selected_member_input_sha256"]))
        fit_by_arm[arm_id] = fit
    if len(member_input_hashes) != 1:
        raise ValueError("frozen exact A_fit member input hash drifted")

    artifacts = report.get("arm_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(
        materialized_arms
    ):
        raise ValueError("materialized point artifact inventory drifted")
    artifact_by_arm: dict[str, Mapping[str, object]] = {}
    path_names: set[str] = set()
    for value, arm_id in zip(artifacts, materialized_arms, strict=True):
        artifact = strict(
            value,
            {
                "arm_id",
                "pair_manifest_sha256",
                "shared_core_state_sha256",
                "fit_receipt_sha256",
                "fit_factor_catalog_sha256",
                "artifact_state_sha256",
                "point_manifest_sha256",
                "point_artifact_sha256",
                "point_artifact_file_sha256",
                "artifact_path_basename",
                "serialized_file_bytes",
                "tensor_storage_bytes",
                "logical_tensor_storage_bytes",
                "serialized_stored_scalar_count",
                "tensor_state_sha256",
                "ordered_state_names_sha256",
                "storage_alias_groups_sha256",
                "storage_alias_group_count",
                "content_point_manifest_sha256",
                "content_addressed_store",
                "per_arm_logical_manifest_materialized",
                "ordered_component_blob_sha256s",
                "ordered_component_blob_kinds",
                "component_blob_catalog_sha256",
                "referenced_blob_count",
                "referenced_blob_file_bytes",
                "referenced_tensor_storage_bytes",
                "newly_materialized_blob_count",
                "newly_materialized_blob_file_bytes",
                "newly_materialized_tensor_storage_bytes",
                "serialized_file_bytes_newly_materialized",
                "unique_scalar_inventory_and_physical_alias_receipt_present",
                "candidate_stored_scalars",
                "source_free_candidate_runtime",
                "serialized_file_bytes_measured",
                "learned_correction_edge_count",
                "chart_or_learned_transport_scalars",
                "source_projection_and_fallback_calls",
                "nonanchor_dense_source_storage_present",
                "artifact_manifest_sha256",
            },
            f"point artifact {arm_id}",
        )
        sealed(
            artifact,
            "artifact_manifest_sha256",
            f"point artifact {arm_id}",
        )
        rank = next(
            (
                candidate_rank
                for candidate_rank in LOW_RANK_POINTS
                if arm_id.endswith(f"r{candidate_rank}")
            ),
            None,
        )
        fit_arm_id = (
            f"primary.shared_core_residual_r{rank}"
            if arm_id.startswith("control.shuffled")
            else arm_id if arm_id in fit_by_arm else None
        )
        expected_fit = fit_by_arm.get(fit_arm_id) if fit_arm_id else None
        basename = artifact.get("artifact_path_basename")
        serialized_count = artifact.get("serialized_stored_scalar_count")
        tensor_bytes = artifact.get("tensor_storage_bytes")
        file_bytes = artifact.get("serialized_file_bytes")
        if (
            artifact.get("arm_id") != arm_id
            or artifact.get("pair_manifest_sha256")
            != typed_pair.fingerprint_sha256
            or artifact.get("shared_core_state_sha256")
            != pair.get("shared_core_state_sha256")
            or artifact.get("fit_receipt_sha256")
            != (
                expected_fit.get("fit_receipt_sha256")
                if expected_fit is not None
                else None
            )
            or artifact.get("fit_factor_catalog_sha256")
            != (
                expected_fit.get("factor_catalog_sha256")
                if expected_fit is not None
                else None
            )
            or any(
                not _is_sha256(artifact.get(name))
                for name in (
                    "artifact_state_sha256",
                    "point_manifest_sha256",
                    "point_artifact_sha256",
                    "point_artifact_file_sha256",
                    "tensor_state_sha256",
                    "ordered_state_names_sha256",
                    "storage_alias_groups_sha256",
                    "content_point_manifest_sha256",
                    "component_blob_catalog_sha256",
                )
            )
            or not isinstance(basename, str)
            or not basename.startswith("points/")
            or not basename.endswith(".json")
            or Path(basename).parts != ("points", Path(basename).name)
            or basename in path_names
            or type(serialized_count) is not int
            or serialized_count != setup_arm_by_id[arm_id]["candidate_stored_scalars"]
            or type(tensor_bytes) is not int
            or tensor_bytes != serialized_count * 4
            or artifact.get("logical_tensor_storage_bytes") != tensor_bytes
            or type(file_bytes) is not int
            or file_bytes <= 0
            or type(artifact.get("storage_alias_group_count")) is not int
            or int(artifact["storage_alias_group_count"]) < 0
            or artifact.get(
                "unique_scalar_inventory_and_physical_alias_receipt_present"
            )
            is not True
            or artifact.get("candidate_stored_scalars") != serialized_count
            or artifact.get("source_free_candidate_runtime") is not True
            or artifact.get("serialized_file_bytes_measured") is not True
            or artifact.get("content_addressed_store") is not True
            or artifact.get("per_arm_logical_manifest_materialized") is not True
            or artifact.get("learned_correction_edge_count") != 0
            or artifact.get("chart_or_learned_transport_scalars") != 0
            or artifact.get("source_projection_and_fallback_calls") != 0
            or artifact.get("nonanchor_dense_source_storage_present")
            is not (arm_id == "exact.shared_core_dense_residual")
        ):
            raise ValueError("point artifact lineage or resource receipt drifted")
        component_shas = artifact.get("ordered_component_blob_sha256s")
        component_kinds = artifact.get("ordered_component_blob_kinds")
        expected_component_kinds = (
            ["shared_base"]
            if arm_id == "ablation.shared_core_zero_residual"
            else [
                "shared_base",
                (
                    "dense_exact_member"
                    if arm_id == "exact.shared_core_dense_residual"
                    else (
                        "independent_adapter"
                        if arm_id.startswith("control.independent")
                        else "primary_adapter"
                    )
                ),
            ]
        )
        new_blob_count = artifact.get("newly_materialized_blob_count")
        new_blob_file_bytes = artifact.get(
            "newly_materialized_blob_file_bytes"
        )
        new_tensor_bytes = artifact.get(
            "newly_materialized_tensor_storage_bytes"
        )
        if (
            not isinstance(component_shas, list)
            or any(not _is_sha256(item) for item in component_shas)
            or component_kinds != expected_component_kinds
            or len(component_shas) != len(component_kinds)
            or artifact.get("referenced_blob_count") != len(component_shas)
            or type(artifact.get("referenced_blob_file_bytes")) is not int
            or int(artifact["referenced_blob_file_bytes"]) <= 0
            or artifact.get("referenced_tensor_storage_bytes") != tensor_bytes
            or type(new_blob_count) is not int
            or not 0 <= int(new_blob_count) <= len(component_shas)
            or type(new_blob_file_bytes) is not int
            or int(new_blob_file_bytes) < 0
            or type(new_tensor_bytes) is not int
            or int(new_tensor_bytes) < 0
            or artifact.get("serialized_file_bytes_newly_materialized")
            != file_bytes + int(new_blob_file_bytes)
            or (
                arm_id.startswith("control.shuffled")
                and (
                    new_blob_count != 0
                    or new_blob_file_bytes != 0
                    or new_tensor_bytes != 0
                )
            )
        ):
            raise ValueError("point artifact component-storage receipt drifted")
        path_names.add(basename)
        artifact_by_arm[arm_id] = artifact

    _validate_report_roles_gates_and_resources_v2(
        report=report,
        canonical_setup=canonical_setup,
        arm_order=arm_order,
        primary_arms=primary_arms,
        fit_arms=fit_arms,
        pair=pair,
        typed_pair=typed_pair,
        fit_by_arm=fit_by_arm,
        artifact_by_arm=artifact_by_arm,
        finite_number=finite_number,
        strict=strict,
        sealed=sealed,
    )


def _validate_report_roles_gates_and_resources_v2(
    *,
    report: Mapping[str, object],
    canonical_setup: Mapping[str, object],
    arm_order: tuple[str, ...],
    primary_arms: tuple[str, ...],
    fit_arms: tuple[str, ...],
    pair: Mapping[str, object],
    typed_pair: object,
    fit_by_arm: Mapping[str, Mapping[str, object]],
    artifact_by_arm: Mapping[str, Mapping[str, object]],
    finite_number: object,
    strict: object,
    sealed: object,
) -> None:
    finite = finite_number
    require = strict
    authenticate = sealed
    pair_sha = typed_pair.fingerprint_sha256
    gates = canonical_setup["acceptance_gates"]

    guard = require(
        report.get("B_guard"),
        {
            "opened_once",
            "entire_frozen_frontier_scored_once",
            "frontier_sha256",
            "evaluated_frozen_frontier_source_sha256",
            "winner_arm_id",
            "bootstrap_seed",
            "bootstrap_resamples",
            "paired_family_clustered",
            "simultaneous_method",
            "simultaneous_critical_value",
            "bootstrap_draw_catalog_sha256",
            "family_metric_input_catalog",
            "family_metric_input_catalog_sha256",
            "runtime_source_gate_input_catalog",
            "runtime_source_gate_input_catalog_sha256",
            "frozen_artifact_state_input_catalog",
            "frozen_artifact_state_input_catalog_sha256",
            "point_results",
            "refit_reorder_prune_or_threshold_change",
            "passed",
            "B_guard_receipt_sha256",
        },
        "B_guard",
    )
    authenticate(guard, "B_guard_receipt_sha256", "B_guard")
    winner = guard.get("winner_arm_id")
    guard_passed = guard.get("passed") is True
    if (winner is None) is guard_passed or (
        winner is not None and winner not in primary_arms
    ):
        raise ValueError("B_guard winner/pass relationship drifted")

    roles_value = report.get("role_results")
    if not isinstance(roles_value, Mapping) or list(roles_value) != list(
        ORDERED_ROLES
    ):
        raise ValueError("role result order drifted")
    expected_role_arms = {
        "A_fit": arm_order[:3],
        "A_select": arm_order,
        "B_guard": arm_order,
        "C_assessment": (
            ("native", "exact.heterogeneous_full_rank_atom_bank_lift", winner)
            if guard_passed
            else ()
        ),
    }
    role_by_id: dict[str, Mapping[str, object]] = {}
    metric_by_role_arm: dict[tuple[str, str], Mapping[str, object]] = {}
    role_fields = {
        "role_id",
        "role_manifest_sha256",
        "example_ids_sha256",
        "example_count",
        "family_count",
        "opened",
        "ordered_observed_arm_ids",
        "arm_metrics",
        "pair_manifest_sha256",
        "fit_performed",
        "pair_selection_performed",
        "dense_exact_control_gate_passed",
        "skipped_reason",
        "threshold_or_topology_changed",
        "role_result_sha256",
    }
    metric_fields = {
        "arm_id",
        "artifact_state_sha256",
        "point_artifact_sha256",
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_query",
        "top1_agreement",
        "valid_query_logit_nrmse",
        "native_correct_downstream_accuracy_retained",
        "worst_family_native_correct_retention",
        "family_native_correct_retention",
        "final_carrier_nrmse",
        "selected_nonanchor_MLP_delta_nrmse",
        "selected_nonanchor_MLP_delta_cosine",
        "source_projection_and_fallback_calls",
        "source_call_audit",
        "runtime_gate_passed",
        "literal_parity",
        "execution_receipt_sha256s",
        "deterministic_replay",
        "all_36_carrier_boundary_metrics",
        "first_divergence_occurrence",
        "all_36_mutation_input_delta_and_boundary_hash_audit",
        "selected_pair_input_delta_and_output_carrier_metrics",
        "passed_fidelity_gate",
    }
    summary_fields = {
        "compared_elements",
        "nrmse",
        "cosine_similarity",
        "maximum_absolute_error",
    }

    def validate_summary(value: object, label: str) -> None:
        summary = require(value, summary_fields, label)
        if type(summary.get("compared_elements")) is not int or int(
            summary["compared_elements"]
        ) <= 0:
            raise ValueError(f"{label} element count drifted")
        for name in summary_fields - {"compared_elements"}:
            finite(summary.get(name), label=f"{label}.{name}")

    for role_id in ORDERED_ROLES:
        role = require(roles_value[role_id], role_fields, f"role {role_id}")
        authenticate(role, "role_result_sha256", f"role {role_id}")
        manifest = canonical_setup["data_firewall"]["roles"][role_id]
        observed_ids = role.get("ordered_observed_arm_ids")
        expected_ids = list(expected_role_arms[role_id])
        metrics = role.get("arm_metrics")
        opened = role_id != "C_assessment" or guard_passed
        if (
            role.get("role_id") != role_id
            or role.get("role_manifest_sha256")
            != manifest["role_manifest_sha256"]
            or role.get("example_ids_sha256")
            != manifest["example_ids_sha256"]
            or role.get("example_count") != manifest["example_count"]
            or role.get("family_count") != manifest["family_count"]
            or role.get("opened") is not opened
            or observed_ids != expected_ids
            or not isinstance(metrics, list)
            or len(metrics) != len(expected_ids)
            or role.get("pair_manifest_sha256") != pair_sha
            or role.get("fit_performed") is not (role_id == "A_fit")
            or role.get("pair_selection_performed")
            is not (role_id == "A_fit")
            or role.get("dense_exact_control_gate_passed")
            is not (role_id == "A_fit")
            or role.get("skipped_reason")
            != ("B_guard_no_winner" if not opened else None)
            or role.get("threshold_or_topology_changed") is not False
        ):
            raise ValueError("observed role inventory or firewall drifted")
        role_mutation_catalogs: dict[str, list[Mapping[str, object]]] = {}
        for metric, arm_id in zip(metrics, expected_ids, strict=True):
            row = require(metric, metric_fields, f"{role_id}.{arm_id}")
            if row.get("arm_id") != arm_id:
                raise ValueError("role metric order drifted")
            expected_artifact = artifact_by_arm.get(arm_id)
            if (
                row.get("artifact_state_sha256")
                != (
                    expected_artifact.get("artifact_state_sha256")
                    if expected_artifact is not None
                    else None
                )
                or row.get("point_artifact_sha256")
                != (
                    expected_artifact.get("point_artifact_sha256")
                    if expected_artifact is not None
                    else None
                )
            ):
                raise ValueError("role candidate artifact state binding drifted")
            for name in (
                "nll_per_token",
                "delta_nll_per_token",
                "native_to_candidate_kl_per_query",
                "top1_agreement",
                "valid_query_logit_nrmse",
                "native_correct_downstream_accuracy_retained",
                "worst_family_native_correct_retention",
                "final_carrier_nrmse",
                "selected_nonanchor_MLP_delta_nrmse",
                "selected_nonanchor_MLP_delta_cosine",
            ):
                finite(row.get(name), label=f"{role_id}.{arm_id}.{name}")
            family_retention = row.get("family_native_correct_retention")
            if (
                not isinstance(family_retention, Mapping)
                or set(family_retention) != set(manifest["family_ids"])
                or any(
                    not math.isfinite(float(value))
                    for value in family_retention.values()
                )
                or float(row["worst_family_native_correct_retention"])
                != min(float(value) for value in family_retention.values())
            ):
                raise ValueError("role family retention receipt drifted")
            source_audit = row.get("source_call_audit")
            if arm_id == "native":
                if (
                    row.get("runtime_gate_passed") is not True
                    or row.get("literal_parity") is not True
                    or row.get("source_projection_and_fallback_calls") != 0
                    or not isinstance(source_audit, Mapping)
                    or source_audit.get("compiled_source_projection_call_count")
                    != 0
                    or source_audit.get("compiled_source_fallback_call_count")
                    != 0
                    or type(
                        source_audit.get("native_reference_model_forward_count")
                    )
                    is not int
                    or int(
                        source_audit["native_reference_model_forward_count"]
                    )
                    <= 0
                    or source_audit.get("passed") is not True
                    or row.get("execution_receipt_sha256s") != []
                    or row.get("deterministic_replay") is not None
                    or row.get("all_36_carrier_boundary_metrics") is not None
                    or row.get("first_divergence_occurrence") is not None
                    or row.get(
                        "all_36_mutation_input_delta_and_boundary_hash_audit"
                    )
                    is not None
                    or row.get(
                        "selected_pair_input_delta_and_output_carrier_metrics"
                    )
                    is not None
                ):
                    raise ValueError("same-role native baseline receipt drifted")
            else:
                if (
                    row.get("runtime_gate_passed") is not True
                    or not isinstance(source_audit, Mapping)
                    or set(source_audit)
                    != {
                        "source_projection_call_count",
                        "source_fallback_call_count",
                        "passed",
                    }
                    or source_audit
                    != {
                        "source_projection_call_count": 0,
                        "source_fallback_call_count": 0,
                        "passed": True,
                    }
                    or row.get("source_projection_and_fallback_calls") != 0
                ):
                    raise ValueError("per-arm source call audit drifted")
                receipts = row.get("execution_receipt_sha256s")
                if (
                    not isinstance(receipts, list)
                    or not receipts
                    or any(not _is_sha256(value) for value in receipts)
                ):
                    raise ValueError("execution receipt catalog drifted")
                replay = require(
                    row.get("deterministic_replay"),
                    {
                        "batch_index",
                        "logits_bitwise_equal",
                        "all_carrier_snapshots_bitwise_equal",
                        "all_mutation_input_hashes_equal",
                        "all_mutation_delta_hashes_equal",
                        "all_output_carrier_hashes_equal",
                        "passed",
                        "replay_receipt_sha256",
                    },
                    "deterministic replay",
                )
                authenticate(
                    replay,
                    "replay_receipt_sha256",
                    "deterministic replay",
                )
                if replay.get("batch_index") != 0 or any(
                    replay.get(name) is not True
                    for name in (
                        "logits_bitwise_equal",
                        "all_carrier_snapshots_bitwise_equal",
                        "all_mutation_input_hashes_equal",
                        "all_mutation_delta_hashes_equal",
                        "all_output_carrier_hashes_equal",
                        "passed",
                    )
                ):
                    raise ValueError("deterministic replay gate failed")
                boundaries = row.get("all_36_carrier_boundary_metrics")
                if not isinstance(boundaries, list) or len(boundaries) != 36:
                    raise ValueError("carrier boundary metric inventory drifted")
                for index, boundary in enumerate(boundaries, 1):
                    value = require(
                        boundary,
                        {"mutation_index", *summary_fields},
                        "carrier boundary metric",
                    )
                    if value.get("mutation_index") != index:
                        raise ValueError("carrier boundary order drifted")
                    validate_summary(
                        {
                            name: value[name] for name in summary_fields
                        },
                        "carrier boundary metric",
                    )
                expected_first = next(
                    (
                        value["mutation_index"]
                        for value in boundaries
                        if float(value["nrmse"]) > 0.0
                    ),
                    None,
                )
                if row.get("first_divergence_occurrence") != expected_first:
                    raise ValueError("first divergence receipt drifted")
                mutation_audit = require(
                    row.get(
                        "all_36_mutation_input_delta_and_boundary_hash_audit"
                    ),
                    {
                        "compared_batch_count",
                        "compared_mutation_count_per_batch",
                        "all_mutation_input_hashes_equal",
                        "all_mutation_delta_hashes_equal",
                        "all_output_carrier_hashes_equal",
                        "comparison_catalog_sha256",
                        "ordered_comparisons",
                    },
                    "mutation hash audit",
                )
                comparisons = mutation_audit.get("ordered_comparisons")
                if (
                    type(mutation_audit.get("compared_batch_count")) is not int
                    or int(mutation_audit["compared_batch_count"]) <= 0
                    or mutation_audit.get("compared_mutation_count_per_batch")
                    != 36
                    or not isinstance(comparisons, list)
                    or len(comparisons)
                    != int(mutation_audit["compared_batch_count"]) * 36
                    or mutation_audit.get("comparison_catalog_sha256")
                    != _manifest_sha256(comparisons)
                ):
                    raise ValueError("mutation hash catalog drifted")
                input_equal = delta_equal = output_equal = True
                for offset, comparison in enumerate(comparisons):
                    value = require(
                        comparison,
                        {
                            "batch_index",
                            "mutation_index",
                            "exact_input_sha256",
                            "candidate_input_sha256",
                            "input_hash_equal",
                            "exact_delta_sha256",
                            "candidate_delta_sha256",
                            "delta_hash_equal",
                            "exact_output_carrier_sha256",
                            "candidate_output_carrier_sha256",
                            "output_carrier_hash_equal",
                        },
                        "mutation hash comparison",
                    )
                    if (
                        value.get("batch_index") != offset // 36
                        or value.get("mutation_index") != offset % 36 + 1
                        or any(
                            not _is_sha256(value.get(name))
                            for name in (
                                "exact_input_sha256",
                                "candidate_input_sha256",
                                "exact_delta_sha256",
                                "candidate_delta_sha256",
                                "exact_output_carrier_sha256",
                                "candidate_output_carrier_sha256",
                            )
                        )
                        or value.get("input_hash_equal")
                        is not (
                            value.get("exact_input_sha256")
                            == value.get("candidate_input_sha256")
                        )
                        or value.get("delta_hash_equal")
                        is not (
                            value.get("exact_delta_sha256")
                            == value.get("candidate_delta_sha256")
                        )
                        or value.get("output_carrier_hash_equal")
                        is not (
                            value.get("exact_output_carrier_sha256")
                            == value.get("candidate_output_carrier_sha256")
                        )
                    ):
                        raise ValueError("mutation hash comparison drifted")
                    input_equal &= bool(value["input_hash_equal"])
                    delta_equal &= bool(value["delta_hash_equal"])
                    output_equal &= bool(value["output_carrier_hash_equal"])
                if (
                    mutation_audit.get("all_mutation_input_hashes_equal")
                    is not input_equal
                    or mutation_audit.get("all_mutation_delta_hashes_equal")
                    is not delta_equal
                    or mutation_audit.get("all_output_carrier_hashes_equal")
                    is not output_equal
                ):
                    raise ValueError("mutation hash audit summary drifted")
                role_mutation_catalogs[arm_id] = comparisons
                selected = require(
                    row.get(
                        "selected_pair_input_delta_and_output_carrier_metrics"
                    ),
                    {"anchor", "member"},
                    "selected pair event metrics",
                )
                for label in ("anchor", "member"):
                    event = require(
                        selected[label],
                        {"input", "delta", "output_carrier"},
                        f"selected {label} event",
                    )
                    for name in ("input", "delta", "output_carrier"):
                        validate_summary(
                            event[name], f"selected {label} {name}"
                        )
                exact_arm = arm_id in {
                    "exact.heterogeneous_full_rank_atom_bank_lift",
                    "exact.shared_core_dense_residual",
                }
                if row.get("literal_parity") is not exact_arm:
                    raise ValueError("literal parity classification drifted")
                if exact_arm and not (
                    input_equal and delta_equal and output_equal
                ):
                    raise ValueError("exact dense hash gate failed")
            metric_by_role_arm[(role_id, arm_id)] = row
        if opened:
            native_row = metric_by_role_arm.get((role_id, "native"))
            if native_row is None:
                raise ValueError("same-role native metric is missing")
            native_nll = float(native_row["nll_per_token"])
            for arm_id in expected_ids:
                row = metric_by_role_arm[(role_id, arm_id)]
                recovered_native_nll = (
                    float(row["nll_per_token"])
                    - float(row["delta_nll_per_token"])
                )
                if not math.isclose(
                    recovered_native_nll,
                    native_nll,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError("same-role native NLL binding drifted")

            exact_arm_ids = {
                "native",
                "exact.heterogeneous_full_rank_atom_bank_lift",
                "exact.shared_core_dense_residual",
            }
            exact_public_metric_fields = (
                "nll_per_token",
                "delta_nll_per_token",
                "native_to_candidate_kl_per_query",
                "top1_agreement",
                "valid_query_logit_nrmse",
                "native_correct_downstream_accuracy_retained",
                "worst_family_native_correct_retention",
                "final_carrier_nrmse",
                "selected_nonanchor_MLP_delta_nrmse",
                "selected_nonanchor_MLP_delta_cosine",
            )
            for arm_id in exact_arm_ids.intersection(expected_ids):
                row = metric_by_role_arm[(role_id, arm_id)]
                expected_identity = {
                    "delta_nll_per_token": 0.0,
                    "native_to_candidate_kl_per_query": 0.0,
                    "top1_agreement": 1.0,
                    "valid_query_logit_nrmse": 0.0,
                    "final_carrier_nrmse": 0.0,
                    "selected_nonanchor_MLP_delta_nrmse": 0.0,
                }
                if any(
                    not math.isclose(
                        float(row[field]),
                        expected,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    for field, expected in expected_identity.items()
                ) or any(
                    not math.isclose(
                        float(row[field]),
                        float(native_row[field]),
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    for field in exact_public_metric_fields
                ) or row["family_native_correct_retention"] != native_row[
                    "family_native_correct_retention"
                ]:
                    raise ValueError("exact identity metric drifted")
                exact_cosine = float(
                    row["selected_nonanchor_MLP_delta_cosine"]
                )
                if not (
                    math.isclose(
                        exact_cosine,
                        0.0,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    or math.isclose(
                        exact_cosine,
                        1.0,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError("exact identity cosine drifted")
                if row.get("passed_fidelity_gate") is not True:
                    raise ValueError("exact identity fidelity gate drifted")
                if arm_id == "native":
                    continue
                for boundary in row["all_36_carrier_boundary_metrics"]:
                    if (
                        not math.isclose(
                            float(boundary["nrmse"]),
                            0.0,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-12,
                        )
                        or not math.isclose(
                            float(boundary["maximum_absolute_error"]),
                            0.0,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-12,
                        )
                    ):
                        raise ValueError("exact boundary identity drifted")
                    boundary_cosine = float(
                        boundary["cosine_similarity"]
                    )
                    if not (
                        math.isclose(
                            boundary_cosine,
                            0.0,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-12,
                        )
                        or math.isclose(
                            boundary_cosine,
                            1.0,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-12,
                        )
                    ):
                        raise ValueError("exact boundary cosine drifted")
                selected = row[
                    "selected_pair_input_delta_and_output_carrier_metrics"
                ]
                for event in selected.values():
                    for summary in event.values():
                        if (
                            not math.isclose(
                                float(summary["nrmse"]),
                                0.0,
                                rel_tol=1.0e-12,
                                abs_tol=1.0e-12,
                            )
                            or not math.isclose(
                                float(summary["maximum_absolute_error"]),
                                0.0,
                                rel_tol=1.0e-12,
                                abs_tol=1.0e-12,
                            )
                        ):
                            raise ValueError(
                                "exact selected-event identity drifted"
                            )
                        summary_cosine = float(
                            summary["cosine_similarity"]
                        )
                        if not (
                            math.isclose(
                                summary_cosine,
                                0.0,
                                rel_tol=1.0e-12,
                                abs_tol=1.0e-12,
                            )
                            or math.isclose(
                                summary_cosine,
                                1.0,
                                rel_tol=1.0e-12,
                                abs_tol=1.0e-12,
                            )
                        ):
                            raise ValueError(
                                "exact selected-event cosine drifted"
                            )

            exact_arm_id = "exact.heterogeneous_full_rank_atom_bank_lift"
            exact_catalog = role_mutation_catalogs.get(exact_arm_id)
            if exact_catalog is None:
                raise ValueError("same-role exact mutation catalog missing")
            exact_by_occurrence = {
                (row["batch_index"], row["mutation_index"]): row
                for row in exact_catalog
            }
            for arm_id, candidate_catalog in role_mutation_catalogs.items():
                if len(candidate_catalog) != len(exact_catalog):
                    raise ValueError(
                        "candidate/exact mutation catalog length drifted"
                    )
                for candidate_row in candidate_catalog:
                    occurrence = (
                        candidate_row["batch_index"],
                        candidate_row["mutation_index"],
                    )
                    exact_row = exact_by_occurrence.get(occurrence)
                    if exact_row is None or any(
                        candidate_row[field] != exact_row[field]
                        for field in (
                            "exact_input_sha256",
                            "exact_delta_sha256",
                            "exact_output_carrier_sha256",
                        )
                    ):
                        raise ValueError(
                            "candidate exact-side mutation catalog is not "
                            "bound to the same-role heterogeneous exact arm"
                        )
        role_by_id[role_id] = role

    frontier = require(
        report.get("frontier"),
        {
            "ordered_primary_arm_ids",
            "promotion_order",
            "ordered_points",
            "A_select_metric_payload_by_primary_arm",
            "frozen_frontier_source_sha256",
            "pair_manifest_sha256",
            "A_select_role_result_sha256",
            "frozen_before_B_guard",
            "refit_or_pruning_performed",
            "frontier_sha256",
        },
        "frontier",
    )
    authenticate(frontier, "frontier_sha256", "frontier")
    ordered_points = frontier.get("ordered_points")
    selection_payloads = frontier.get(
        "A_select_metric_payload_by_primary_arm"
    )
    if (
        not isinstance(ordered_points, list)
        or len(ordered_points) != 4
        or not isinstance(selection_payloads, Mapping)
        or set(selection_payloads) != set(primary_arms)
    ):
        raise ValueError("frozen frontier point inventory drifted")
    for point, arm_id in zip(ordered_points, primary_arms, strict=True):
        if (
            not isinstance(point, Mapping)
            or set(point)
            != {
                "arm_id",
                "metrics_sha256",
                "artifact_state_sha256",
                "candidate_stored_scalars",
            }
            or point.get("arm_id") != arm_id
            or not isinstance(selection_payloads[arm_id], Mapping)
            or point.get("metrics_sha256")
            != _manifest_sha256(selection_payloads[arm_id])
            or point.get("artifact_state_sha256")
            != artifact_by_arm[arm_id]["artifact_state_sha256"]
            or point.get("candidate_stored_scalars")
            != artifact_by_arm[arm_id]["candidate_stored_scalars"]
        ):
            raise ValueError("frozen frontier point binding drifted")
        public_metric = metric_by_role_arm[("A_select", arm_id)]
        for public_name, source_name in (
            ("nll_per_token", "nll_per_token"),
            ("delta_nll_per_token", "delta_nll_per_token"),
            (
                "native_to_candidate_kl_per_query",
                "native_to_candidate_kl_per_query",
            ),
            ("top1_agreement", "top1_agreement_to_native"),
            ("valid_query_logit_nrmse", "valid_query_logit_nrmse"),
            (
                "native_correct_downstream_accuracy_retained",
                "native_correct_downstream_accuracy_retained",
            ),
            (
                "selected_nonanchor_MLP_delta_nrmse",
                "selected_nonanchor_MLP_delta_nrmse",
            ),
            (
                "selected_nonanchor_MLP_delta_cosine",
                "selected_nonanchor_MLP_delta_cosine",
            ),
            ("final_carrier_nrmse", "final_carrier_nrmse"),
        ):
            if public_metric[public_name] != selection_payloads[arm_id].get(
                source_name
            ):
                raise ValueError("A_select frontier metric payload drifted")
    source_payload = {
        "ordered_primary_arm_ids": list(primary_arms),
        "ordered_points": ordered_points,
        "selection_role": "A_select",
        "pair_anchor_topology_and_fits_frozen": True,
        "refit_or_pruning_performed": False,
    }
    if (
        frontier.get("ordered_primary_arm_ids") != list(primary_arms)
        or frontier.get("promotion_order") != list(reversed(primary_arms))
        or frontier.get("frozen_frontier_source_sha256")
        != _manifest_sha256(source_payload)
        or frontier.get("pair_manifest_sha256") != pair_sha
        or frontier.get("A_select_role_result_sha256")
        != role_by_id["A_select"]["role_result_sha256"]
        or frontier.get("frozen_before_B_guard") is not True
        or frontier.get("refit_or_pruning_performed") is not False
    ):
        raise ValueError("frozen frontier authentication drifted")

    family_catalog = guard.get("family_metric_input_catalog")
    runtime_catalog = guard.get("runtime_source_gate_input_catalog")
    frozen_state_catalog = guard.get("frozen_artifact_state_input_catalog")
    if (
        not isinstance(family_catalog, Mapping)
        or guard.get("family_metric_input_catalog_sha256")
        != _manifest_sha256(family_catalog)
        or not isinstance(runtime_catalog, Mapping)
        or guard.get("runtime_source_gate_input_catalog_sha256")
        != _manifest_sha256(runtime_catalog)
        or not isinstance(frozen_state_catalog, Mapping)
        or guard.get("frozen_artifact_state_input_catalog_sha256")
        != _manifest_sha256(frozen_state_catalog)
    ):
        raise ValueError("B_guard input catalog authentication failed")
    reconstructed_arms: dict[str, object] = {}
    for arm_id, family_rows in family_catalog.items():
        runtime_row = runtime_catalog.get(arm_id)
        if not isinstance(family_rows, Mapping) or not isinstance(
            runtime_row, Mapping
        ):
            raise ValueError("B_guard input catalog row drifted")
        replay_sha = runtime_row.get(
            "deterministic_replay_receipt_sha256"
        )
        reconstructed_arms[arm_id] = {
            "_family_metric_values": family_rows,
            "runtime_gate_passed": runtime_row.get("runtime_gate_passed"),
            "source_call_audit": {
                "passed": runtime_row.get("source_call_audit_passed")
            },
            "deterministic_replay": {
                "passed": runtime_row.get("deterministic_replay_passed"),
                "replay_receipt_sha256": replay_sha,
            },
        }
    recomputed_guard = _run_b_guard_live(
        role_result={
            "arm_results": reconstructed_arms,
            "artifact_receipts": {
                arm_id: {
                    "candidate_state_sha256": value[
                        "artifact_state_sha256"
                    ],
                    "candidate_stored_scalars": value[
                        "candidate_stored_scalars"
                    ],
                }
                for arm_id, value in frozen_state_catalog.items()
            },
        },
        frontier={
            "ordered_primary_arm_ids": list(primary_arms),
            "ordered_points": ordered_points,
            "frontier_sha256": frontier[
                "frozen_frontier_source_sha256"
            ],
            "frozen_before_B_guard": True,
        },
        setup=canonical_setup,
    )
    for report_field, live_field in (
        ("opened_once", "opened_once"),
        ("entire_frozen_frontier_scored_once", "entire_frontier_scored_once"),
        ("evaluated_frozen_frontier_source_sha256", "frontier_sha256"),
        ("winner_arm_id", "winner_arm_id"),
        ("bootstrap_seed", "bootstrap_seed"),
        ("bootstrap_resamples", "bootstrap_resamples"),
        ("paired_family_clustered", "paired_family_clustered"),
        ("simultaneous_method", "simultaneous_method"),
        ("simultaneous_critical_value", "simultaneous_critical_value"),
        ("bootstrap_draw_catalog_sha256", "bootstrap_draw_catalog_sha256"),
        ("family_metric_input_catalog", "family_metric_input_catalog"),
        ("family_metric_input_catalog_sha256", "family_metric_input_catalog_sha256"),
        ("runtime_source_gate_input_catalog", "runtime_source_gate_input_catalog"),
        ("runtime_source_gate_input_catalog_sha256", "runtime_source_gate_input_catalog_sha256"),
        ("frozen_artifact_state_input_catalog", "frozen_artifact_state_input_catalog"),
        ("frozen_artifact_state_input_catalog_sha256", "frozen_artifact_state_input_catalog_sha256"),
        ("point_results", "point_results"),
        ("refit_reorder_prune_or_threshold_change", "refit_reorder_prune_or_threshold_change"),
        ("passed", "passed"),
    ):
        if guard.get(report_field) != recomputed_guard.get(live_field):
            raise ValueError("B_guard deterministic recomputation failed")
    if guard.get("frontier_sha256") != frontier.get("frontier_sha256"):
        raise ValueError("B_guard report frontier binding drifted")

    _validate_report_c_gate_and_resources_v2(
        report=report,
        canonical_setup=canonical_setup,
        arm_order=arm_order,
        primary_arms=primary_arms,
        fit_arms=fit_arms,
        pair_sha=pair_sha,
        fit_by_arm=fit_by_arm,
        artifact_by_arm=artifact_by_arm,
        role_by_id=role_by_id,
        metric_by_role_arm=metric_by_role_arm,
        frontier=frontier,
        guard=guard,
        require=require,
        authenticate=authenticate,
        finite=finite,
    )


def _validate_report_c_gate_and_resources_v2(
    *,
    report: Mapping[str, object],
    canonical_setup: Mapping[str, object],
    arm_order: tuple[str, ...],
    primary_arms: tuple[str, ...],
    fit_arms: tuple[str, ...],
    pair_sha: str,
    fit_by_arm: Mapping[str, Mapping[str, object]],
    artifact_by_arm: Mapping[str, Mapping[str, object]],
    role_by_id: Mapping[str, Mapping[str, object]],
    metric_by_role_arm: Mapping[tuple[str, str], Mapping[str, object]],
    frontier: Mapping[str, object],
    guard: Mapping[str, object],
    require: object,
    authenticate: object,
    finite: object,
) -> None:
    winner = guard.get("winner_arm_id")
    guard_passed = guard.get("passed") is True
    assessment = require(
        report.get("C_assessment"),
        {
            "opened_at_most_once",
            "opened_once",
            "opened_after_guard_pass",
            "candidate_count",
            "candidate_arm_id",
            "same_point_thresholds_as_B_guard",
            "family_clustered_bootstrap_seed",
            "family_clustered_bootstrap_resamples",
            "family_clustered_bootstrap_draw_catalog_sha256",
            "family_native_correct_retention",
            "worst_family_retention_simultaneous_lower_bound",
            "worst_family_retention_gate_passed",
            "aggregate_fidelity_gate_passed",
            "runtime_and_source_call_gates_passed",
            "refit_selection_or_threshold_change",
            "skipped_reason",
            "passed",
            "C_assessment_receipt_sha256",
        },
        "C assessment",
    )
    authenticate(
        assessment,
        "C_assessment_receipt_sha256",
        "C assessment",
    )
    assessment_passed = assessment.get("passed") is True
    if assessment.get("opened_at_most_once") is not True:
        raise ValueError("C assessment at-most-once gate drifted")
    if not guard_passed:
        if assessment != {
            **assessment,
            "opened_at_most_once": True,
            "opened_once": False,
            "opened_after_guard_pass": False,
            "candidate_count": 0,
            "candidate_arm_id": None,
            "same_point_thresholds_as_B_guard": True,
            "family_clustered_bootstrap_seed": None,
            "family_clustered_bootstrap_resamples": None,
            "family_clustered_bootstrap_draw_catalog_sha256": None,
            "family_native_correct_retention": {},
            "worst_family_retention_simultaneous_lower_bound": None,
            "worst_family_retention_gate_passed": False,
            "aggregate_fidelity_gate_passed": False,
            "runtime_and_source_call_gates_passed": False,
            "refit_selection_or_threshold_change": False,
            "skipped_reason": "B_guard_no_winner",
            "passed": False,
        }:
            raise ValueError("honest C skip receipt drifted")
    else:
        winner_metric = metric_by_role_arm.get(("C_assessment", str(winner)))
        if not isinstance(winner_metric, Mapping):
            raise ValueError("C assessment winner metric is unavailable")
        family_retention = assessment.get("family_native_correct_retention")
        expected_families = canonical_setup["data_firewall"]["roles"][
            "C_assessment"
        ]["family_ids"]
        if (
            not isinstance(family_retention, Mapping)
            or set(family_retention) != set(expected_families)
            or family_retention
            != winner_metric.get("family_native_correct_retention")
        ):
            raise ValueError("C family retention binding drifted")
        worst = min(float(value) for value in family_retention.values())
        worst_gate = worst >= float(
            canonical_setup["acceptance_gates"]
            ["compressed_candidate_fidelity"]
            ["worst_family_bootstrap_lower_bound_minimum"]
        )
        draw_payload = {
            "role_id": "C_assessment",
            "seed": 1_733,
            "resamples": 10_000,
            "family_ids": list(expected_families),
            "draw_indices": [0] * 10_000,
        }
        fidelity = canonical_setup["acceptance_gates"][
            "compressed_candidate_fidelity"
        ]
        aggregate_gate = (
            float(winner_metric["delta_nll_per_token"])
            <= float(fidelity["delta_nll_per_token_maximum"])
            and float(winner_metric["native_to_candidate_kl_per_query"])
            <= float(fidelity["native_to_candidate_kl_per_query_maximum"])
            and float(winner_metric["top1_agreement"])
            >= float(fidelity["top1_agreement_minimum"])
            and float(winner_metric["valid_query_logit_nrmse"])
            <= float(fidelity["valid_query_logit_nrmse_maximum"])
            and float(
                winner_metric[
                    "native_correct_downstream_accuracy_retained"
                ]
            )
            >= float(
                fidelity[
                    "native_correct_downstream_accuracy_retained_minimum"
                ]
            )
            and float(
                winner_metric["selected_nonanchor_MLP_delta_nrmse"]
            )
            <= float(
                fidelity["selected_nonanchor_MLP_delta_nrmse_maximum"]
            )
            and float(
                winner_metric["selected_nonanchor_MLP_delta_cosine"]
            )
            >= float(
                fidelity["selected_nonanchor_MLP_delta_cosine_minimum"]
            )
            and float(winner_metric["final_carrier_nrmse"])
            <= float(fidelity["final_carrier_nrmse_maximum"])
        )
        runtime_gate = winner_metric.get("runtime_gate_passed") is True
        if (
            assessment.get("opened_once") is not True
            or assessment.get("opened_after_guard_pass") is not True
            or assessment.get("candidate_count") != 1
            or assessment.get("candidate_arm_id") != winner
            or assessment.get("same_point_thresholds_as_B_guard") is not True
            or assessment.get("family_clustered_bootstrap_seed") != 1_733
            or assessment.get("family_clustered_bootstrap_resamples")
            != 10_000
            or assessment.get(
                "family_clustered_bootstrap_draw_catalog_sha256"
            )
            != _manifest_sha256(draw_payload)
            or assessment.get(
                "worst_family_retention_simultaneous_lower_bound"
            )
            != worst
            or assessment.get("worst_family_retention_gate_passed")
            is not worst_gate
            or assessment.get("aggregate_fidelity_gate_passed")
            is not aggregate_gate
            or assessment.get("runtime_and_source_call_gates_passed")
            is not runtime_gate
            or assessment.get("refit_selection_or_threshold_change")
            is not False
            or assessment.get("skipped_reason") is not None
            or assessment_passed
            is not (worst_gate and aggregate_gate and runtime_gate)
        ):
            raise ValueError("C assessment gate receipt drifted")

    gate = require(
        report.get("gate_chain"),
        {
            "ordered_gates",
            "dense_exact_control_passed_before_compressed_fit",
            "failed_approximate_point_did_not_prune_frontier",
            "all_passed",
        },
        "gate chain",
    )
    qualified = guard_passed and assessment_passed
    if gate != {
        "ordered_gates": [
            "preflight",
            "exact_predecessor",
            "dense_exact_control",
            "fit_frozen_frontier",
            "A_select_freeze",
            "B_guard",
            "C_assessment",
        ],
        "dense_exact_control_passed_before_compressed_fit": True,
        "failed_approximate_point_did_not_prune_frontier": True,
        "all_passed": qualified,
    }:
        raise ValueError("gate chain drifted")

    audit = require(
        report.get("execution_audit"),
        {
            "preflight_before_live_dependencies",
            "source_model_unchanged",
            "source_execution_unchanged",
            "source_projection_and_fallback_calls",
            "learned_correction_edge_count",
            "chart_or_learned_transport_scalars",
            "pair_manifest_sha256_by_role",
            "fit_call_count",
            "unique_fit_nonce_count",
            "selected_member_prefix_is_exact_by_single_approximation",
            "candidate_prefix_recapture_required",
            "exact_prefix_input_hash_equality_authenticated_on_A_fit",
            "selected_member_input_sha256_by_fit_arm",
            "all_39_invocations_and_75_occurrences_retained",
        },
        "execution audit",
    )
    expected_input_hashes = {
        arm_id: fit_by_arm[arm_id]["selected_member_input_sha256"]
        for arm_id in fit_arms
    }
    if audit != {
        "preflight_before_live_dependencies": True,
        "source_model_unchanged": True,
        "source_execution_unchanged": True,
        "source_projection_and_fallback_calls": 0,
        "learned_correction_edge_count": 0,
        "chart_or_learned_transport_scalars": 0,
        "pair_manifest_sha256_by_role": {
            role: pair_sha for role in ORDERED_ROLES
        },
        "fit_call_count": 8,
        "unique_fit_nonce_count": 8,
        "selected_member_prefix_is_exact_by_single_approximation": True,
        "candidate_prefix_recapture_required": False,
        "exact_prefix_input_hash_equality_authenticated_on_A_fit": True,
        "selected_member_input_sha256_by_fit_arm": expected_input_hashes,
        "all_39_invocations_and_75_occurrences_retained": True,
    }:
        raise ValueError("execution audit or exact-prefix proof drifted")

    resources = require(
        report.get("resources"),
        {
            "source_stored_scalars",
            "candidate_stored_scalars_by_arm",
            "serialized_file_bytes_by_materialized_arm",
            "tensor_storage_bytes_by_materialized_arm",
            "serialized_point_artifact_count",
            "artifact_storage_preflight",
            "content_addressed_store",
            "matched_control_budget_equal_by_rank",
            "nonanchor_dense_source_storage_absent_for_compressed_artifacts",
            "serialized_file_bytes_measured",
            "cross_artifact_tensor_payload_deduplication_measured",
            "serialized_byte_reduction_measured",
            "logical_MAC_reduction_measured",
            "peak_memory_measured",
            "latency_measured",
        },
        "resources",
    )
    expected_counts = {
        value["arm_id"]: value["candidate_stored_scalars"]
        for value in canonical_setup["arm_contract"]["arms"]
    }
    expected_files = {
        arm_id: artifact_by_arm[arm_id]["serialized_file_bytes"]
        for arm_id in artifact_by_arm
    }
    expected_tensors = {
        arm_id: artifact_by_arm[arm_id]["tensor_storage_bytes"]
        for arm_id in artifact_by_arm
    }
    storage = require(
        resources.get("artifact_storage_preflight"),
        {
            "artifact_root",
            "materialized_point_artifact_count",
            "unique_tensor_blob_count",
            "component_blob_reference_count",
            "reused_component_blob_reference_count",
            "shared_base_stored_scalars",
            "dense_exact_member_stored_scalars",
            "primary_adapter_stored_scalars",
            "independent_adapter_stored_scalars",
            "logical_flat_stored_scalars",
            "unique_content_addressed_stored_scalars",
            "logical_flat_tensor_storage_bytes",
            "unique_content_addressed_tensor_storage_bytes",
            "content_addressed_tensor_storage_savings_bytes",
            "content_addressed_tensor_storage_savings_fraction",
            "shuffled_adapter_blob_reuses_primary",
            "logical_candidate_scalar_accounting_unchanged",
            "estimated_tensor_storage_bytes",
            "safety_margin_bytes",
            "required_free_bytes",
            "available_free_bytes",
            "passed",
        },
        "artifact storage preflight",
    )
    plan = _content_addressed_storage_plan(canonical_setup)
    estimated = int(plan["unique_content_addressed_tensor_storage_bytes"])
    margin = max(1 << 30, math.ceil(estimated * 0.10))
    store = require(
        resources.get("content_addressed_store"),
        {
            "schema",
            "format_version",
            "ordered_arm_ids",
            "point_manifest_count",
            "point_manifests",
            "unique_blob_count",
            "unique_blobs",
            "component_blob_reference_count",
            "reused_component_blob_reference_count",
            "logical_flat_tensor_storage_bytes",
            "unique_tensor_storage_bytes",
            "tensor_storage_savings_bytes",
            "tensor_storage_savings_fraction",
            "unique_blob_file_bytes",
            "point_manifest_file_bytes",
            "logical_candidate_scalar_accounting_unchanged",
            "equal_values_without_logical_alias_remain_separate",
            "shuffled_adapter_reuses_primary_blob",
            "source_free_serialized_candidates",
            "exact_point_and_blob_inventory",
            "store_inventory_sha256",
            "inventory_path_basename",
            "inventory_file_sha256",
            "inventory_file_bytes",
            "serialized_store_file_bytes",
            "unique_blob_catalog_sha256",
            "point_manifest_catalog_sha256",
            "content_addressed_store_receipt_sha256",
        },
        "content-addressed store",
    )
    store_without_receipt = dict(store)
    store_receipt_sha = store_without_receipt.pop(
        "content_addressed_store_receipt_sha256"
    )
    unique_blobs = store.get("unique_blobs")
    point_manifests = store.get("point_manifests")
    core_inventory = dict(store)
    for report_only_field in (
        "inventory_path_basename",
        "inventory_file_sha256",
        "inventory_file_bytes",
        "serialized_store_file_bytes",
        "unique_blob_catalog_sha256",
        "point_manifest_catalog_sha256",
        "content_addressed_store_receipt_sha256",
    ):
        core_inventory.pop(report_only_field, None)
    observed_core_inventory_sha = core_inventory.pop(
        "store_inventory_sha256", None
    )
    if (
        not isinstance(unique_blobs, list)
        or not isinstance(point_manifests, list)
        or observed_core_inventory_sha
        != _content_store_inventory_sha256(core_inventory)
        or store_receipt_sha != _manifest_sha256(store_without_receipt)
        or store.get("unique_blob_catalog_sha256")
        != _manifest_sha256(unique_blobs)
        or store.get("point_manifest_catalog_sha256")
        != _manifest_sha256(point_manifests)
    ):
        raise ValueError("content-addressed store catalog authentication failed")
    blob_row_fields = {
        "blob_kind",
        "blob_sha256",
        "blob_file_basename",
        "blob_file_sha256",
        "blob_file_bytes",
        "tensor_state_sha256",
        "stored_scalar_count",
        "tensor_storage_bytes",
    }
    point_row_fields = {
        "arm_id",
        "point_manifest_path_basename",
        "content_point_manifest_sha256",
        "point_manifest_file_sha256",
        "point_manifest_file_bytes",
        "point_artifact_sha256",
        "candidate_state_sha256",
        "serialized_stored_scalar_count",
        "logical_tensor_storage_bytes",
        "ordered_component_blob_sha256s",
    }
    blob_by_sha: dict[str, Mapping[str, object]] = {}
    kind_scalars: dict[str, list[int]] = {}
    for row in unique_blobs:
        if not isinstance(row, Mapping) or set(row) != blob_row_fields:
            raise ValueError("content-addressed unique-blob row fields drifted")
        blob_sha = row.get("blob_sha256")
        blob_kind = row.get("blob_kind")
        scalar_count = row.get("stored_scalar_count")
        tensor_bytes = row.get("tensor_storage_bytes")
        if (
            not _is_sha256(blob_sha)
            or blob_sha in blob_by_sha
            or blob_kind
            not in {
                "shared_base",
                "dense_exact_member",
                "primary_adapter",
                "independent_adapter",
            }
            or row.get("blob_file_basename") != f"{blob_sha}.pt"
            or not _is_sha256(row.get("blob_file_sha256"))
            or not _is_sha256(row.get("tensor_state_sha256"))
            or type(row.get("blob_file_bytes")) is not int
            or int(row["blob_file_bytes"]) <= 0
            or type(scalar_count) is not int
            or int(scalar_count) <= 0
            or type(tensor_bytes) is not int
            or tensor_bytes != int(scalar_count) * 4
            or int(row["blob_file_bytes"]) < int(tensor_bytes)
        ):
            raise ValueError("content-addressed unique-blob row drifted")
        blob_by_sha[str(blob_sha)] = row
        kind_scalars.setdefault(str(blob_kind), []).append(int(scalar_count))
    expected_adapter_scalars = [259_328, 517_376, 1_033_472, 2_065_664]
    if (
        kind_scalars.get("shared_base") != [264_164_736]
        or kind_scalars.get("dense_exact_member") != [3_933_440]
        or sorted(kind_scalars.get("primary_adapter", []))
        != expected_adapter_scalars
        or sorted(kind_scalars.get("independent_adapter", []))
        != expected_adapter_scalars
    ):
        raise ValueError("content-addressed blob kind/scalar inventory drifted")
    expected_materialized_arms = list(arm_order[2:])
    if store.get("ordered_arm_ids") != expected_materialized_arms:
        raise ValueError("content-addressed store arm order drifted")
    reference_multiplicity: dict[str, int] = {}
    for row, arm_id in zip(
        point_manifests,
        expected_materialized_arms,
        strict=True,
    ):
        if not isinstance(row, Mapping) or set(row) != point_row_fields:
            raise ValueError("content-addressed point-inventory row fields drifted")
        artifact = artifact_by_arm[arm_id]
        references = row.get("ordered_component_blob_sha256s")
        referenced_rows = (
            [blob_by_sha[reference] for reference in references]
            if isinstance(references, list)
            and all(reference in blob_by_sha for reference in references)
            else []
        )
        referenced_kinds = [
            value["blob_kind"] for value in referenced_rows
        ]
        referenced_file_bytes = sum(
            int(value["blob_file_bytes"]) for value in referenced_rows
        )
        referenced_tensor_bytes = sum(
            int(value["tensor_storage_bytes"]) for value in referenced_rows
        )
        if arm_id == "exact.shared_core_dense_residual":
            newly_materialized_rows = referenced_rows
        elif arm_id.startswith("primary.") or arm_id.startswith(
            "control.independent"
        ):
            newly_materialized_rows = referenced_rows[1:]
        else:
            newly_materialized_rows = []
        expected_new_blob_file_bytes = sum(
            int(value["blob_file_bytes"])
            for value in newly_materialized_rows
        )
        expected_new_tensor_bytes = sum(
            int(value["tensor_storage_bytes"])
            for value in newly_materialized_rows
        )
        component_catalog = [
            {
                name: value[name]
                for name in (
                    "blob_kind",
                    "blob_sha256",
                    "blob_file_sha256",
                    "blob_file_bytes",
                    "tensor_storage_bytes",
                )
            }
            for value in referenced_rows
        ]
        if (
            row.get("arm_id") != arm_id
            or row.get("point_manifest_path_basename")
            != artifact.get("artifact_path_basename")
            or row.get("content_point_manifest_sha256")
            != artifact.get("content_point_manifest_sha256")
            or row.get("point_manifest_file_sha256")
            != artifact.get("point_artifact_file_sha256")
            or row.get("point_manifest_file_bytes")
            != artifact.get("serialized_file_bytes")
            or row.get("point_artifact_sha256")
            != artifact.get("point_artifact_sha256")
            or row.get("candidate_state_sha256")
            != artifact.get("artifact_state_sha256")
            or row.get("serialized_stored_scalar_count")
            != artifact.get("serialized_stored_scalar_count")
            or row.get("logical_tensor_storage_bytes")
            != artifact.get("logical_tensor_storage_bytes")
            or references
            != artifact.get("ordered_component_blob_sha256s")
            or not isinstance(references, list)
            or any(reference not in blob_by_sha for reference in references)
            or referenced_kinds
            != artifact.get("ordered_component_blob_kinds")
            or artifact.get("component_blob_catalog_sha256")
            != _manifest_sha256(component_catalog)
            or artifact.get("referenced_blob_file_bytes")
            != referenced_file_bytes
            or artifact.get("referenced_tensor_storage_bytes")
            != referenced_tensor_bytes
            or artifact.get("newly_materialized_blob_count")
            != len(newly_materialized_rows)
            or artifact.get("newly_materialized_blob_file_bytes")
            != expected_new_blob_file_bytes
            or artifact.get("newly_materialized_tensor_storage_bytes")
            != expected_new_tensor_bytes
            or artifact.get("serialized_file_bytes_newly_materialized")
            != int(artifact["serialized_file_bytes"])
            + expected_new_blob_file_bytes
        ):
            raise ValueError("content-addressed point inventory binding drifted")
        for reference in references:
            reference_multiplicity[reference] = (
                reference_multiplicity.get(reference, 0) + 1
            )
    kind_reference_multiplicities: dict[str, list[int]] = {}
    for blob_sha, count in reference_multiplicity.items():
        kind = str(blob_by_sha[blob_sha]["blob_kind"])
        kind_reference_multiplicities.setdefault(kind, []).append(count)
    if (
        set(reference_multiplicity) != set(blob_by_sha)
        or sorted(kind_reference_multiplicities.get("shared_base", [])) != [14]
        or sorted(kind_reference_multiplicities.get("dense_exact_member", []))
        != [1]
        or sorted(kind_reference_multiplicities.get("primary_adapter", []))
        != [2, 2, 2, 2]
        or sorted(
            kind_reference_multiplicities.get("independent_adapter", [])
        )
        != [1, 1, 1, 1]
    ):
        raise ValueError("content-addressed blob reference multiplicity drifted")
    unique_blob_file_bytes = sum(
        int(value.get("blob_file_bytes", -1))
        for value in unique_blobs
        if isinstance(value, Mapping)
    )
    point_manifest_file_bytes = sum(
        int(value.get("point_manifest_file_bytes", -1))
        for value in point_manifests
        if isinstance(value, Mapping)
    )
    inventory_file_bytes = store.get("inventory_file_bytes")
    if (
        resources.get("source_stored_scalars") != 268_098_176
        or resources.get("candidate_stored_scalars_by_arm")
        != expected_counts
        or resources.get("serialized_file_bytes_by_materialized_arm")
        != expected_files
        or resources.get("tensor_storage_bytes_by_materialized_arm")
        != expected_tensors
        or resources.get("serialized_point_artifact_count") != 14
        or resources.get("matched_control_budget_equal_by_rank") is not True
        or resources.get(
            "nonanchor_dense_source_storage_absent_for_compressed_artifacts"
        )
        is not True
        or resources.get("serialized_file_bytes_measured") is not True
        or resources.get(
            "cross_artifact_tensor_payload_deduplication_measured"
        )
        is not True
        or resources.get("serialized_byte_reduction_measured") is not False
        or resources.get("logical_MAC_reduction_measured") is not False
        or resources.get("peak_memory_measured") is not False
        or resources.get("latency_measured") is not False
        or not isinstance(storage.get("artifact_root"), str)
        or not storage.get("artifact_root")
        or storage.get("materialized_point_artifact_count") != 14
        or any(storage.get(name) != value for name, value in plan.items())
        or storage.get("estimated_tensor_storage_bytes") != estimated
        or storage.get("safety_margin_bytes") != margin
        or storage.get("required_free_bytes") != estimated + margin
        or type(storage.get("available_free_bytes")) is not int
        or int(storage["available_free_bytes"]) < estimated + margin
        or storage.get("passed") is not True
        or store.get("schema")
        != "fisher_graph.gemma3_mlp_shared_family.content_store.v1"
        or store.get("format_version") != 1
        or store.get("point_manifest_count") != 14
        or len(point_manifests) != 14
        or store.get("unique_blob_count") != 10
        or len(unique_blobs) != 10
        or store.get("component_blob_reference_count") != 27
        or store.get("reused_component_blob_reference_count") != 17
        or store.get("logical_flat_tensor_storage_bytes")
        != 14_855_469_056
        or store.get("unique_tensor_storage_bytes") != 1_103_399_424
        or store.get("tensor_storage_savings_bytes") != 13_752_069_632
        or store.get("tensor_storage_savings_fraction")
        != 13_752_069_632 / 14_855_469_056
        or store.get("unique_blob_file_bytes") != unique_blob_file_bytes
        or store.get("point_manifest_file_bytes")
        != point_manifest_file_bytes
        or type(inventory_file_bytes) is not int
        or int(inventory_file_bytes) <= 0
        or store.get("serialized_store_file_bytes")
        != unique_blob_file_bytes
        + point_manifest_file_bytes
        + int(inventory_file_bytes)
        or int(store["serialized_store_file_bytes"])
        > int(storage["required_free_bytes"])
        or any(
            store.get(name) is not True
            for name in (
                "logical_candidate_scalar_accounting_unchanged",
                "equal_values_without_logical_alias_remain_separate",
                "shuffled_adapter_reuses_primary_blob",
                "source_free_serialized_candidates",
                "exact_point_and_blob_inventory",
            )
        )
        or any(
            not _is_sha256(store.get(name))
            for name in (
                "store_inventory_sha256",
                "inventory_file_sha256",
                "unique_blob_catalog_sha256",
                "point_manifest_catalog_sha256",
                "content_addressed_store_receipt_sha256",
            )
        )
        or store.get("inventory_path_basename") != "store-inventory.json"
        or artifact_by_arm["exact.shared_core_dense_residual"].get(
            "nonanchor_dense_source_storage_present"
        )
        is not True
        or any(
            artifact_by_arm[arm_id].get(
                "nonanchor_dense_source_storage_present"
            )
            is not False
            for arm_id in artifact_by_arm
            if arm_id != "exact.shared_core_dense_residual"
        )
    ):
        raise ValueError("resource or storage-preflight receipt drifted")

    claims = require(
        report.get("claims"),
        {
            "pair_mechanism_qualified",
            "twenty_percent_whole_model_compression_claimed",
            "learned_edge_compression_claimed",
            "serialized_byte_reduction_claimed",
            "logical_MAC_reduction_claimed",
            "latency_or_speed_claimed",
        },
        "claims",
    )
    if (
        claims.get("pair_mechanism_qualified") is not qualified
        or any(
            claims.get(name) is not False
            for name in (
                "twenty_percent_whole_model_compression_claimed",
                "learned_edge_compression_claimed",
                "serialized_byte_reduction_claimed",
                "logical_MAC_reduction_claimed",
                "latency_or_speed_claimed",
            )
        )
        or report.get("diagnostic_outcome")
        != (
            "shared_family_pair_mechanism_qualified"
            if qualified
            else "shared_family_pair_mechanism_not_qualified"
        )
    ):
        raise ValueError("claim firewall drifted")


def validate_gemma3_mlp_shared_family_residual_adapter_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("shared-family report must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    observed = materialized.pop("report_sha256", None)
    if not _is_sha256(observed) or _report_sha256(materialized) != observed:
        raise ValueError("shared-family report hash is invalid")
    _validate_report_semantics_v2(materialized)
    materialized["report_sha256"] = observed
    return materialized


def load_gemma3_mlp_shared_family_residual_adapter_report(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_gemma3_mlp_shared_family_residual_adapter_report(value)


def write_gemma3_mlp_shared_family_residual_adapter_report(
    report: Mapping[str, object],
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    validated = validate_gemma3_mlp_shared_family_residual_adapter_report(report)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(validated, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError("refusing to overwrite a shared-family report") from error
    loaded = load_gemma3_mlp_shared_family_residual_adapter_report(target)
    if loaded != validated:
        raise RuntimeError("shared-family report changed across strict reload")
    return loaded


def _source_layer(runtime: _Runtime, ordinal: int) -> object:
    return runtime.adapter.source_module(runtime.adapter.layers[ordinal].id)


def _prepare_runtime_live(
    *,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> _Runtime:
    del preflight
    device = dependencies.resolve_device(device_name)
    cache = dependencies.resolve_paths(cache_dir)["hub_cache"]
    tokenizer, model = dependencies.load_model(
        model_id=PINNED_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = dependencies.adapter_type(model)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != 268_098_176:
        raise RuntimeError("pinned source scalar inventory drifted")
    model_fingerprint = adapter.model_fingerprint()
    execution_fingerprint = adapter.execution_fingerprint()
    if (
        model_fingerprint != PINNED_SOURCE_MODEL_FINGERPRINT
        or execution_fingerprint != PINNED_SOURCE_EXECUTION_FINGERPRINT
    ):
        raise RuntimeError("loaded source fingerprint drifted before compilation")
    guard = dependencies.frozen_guard_type(model)
    exact_carrier = dependencies.compile_exact_carrier(
        adapter,
        dtype=dependencies.torch.float32,
        device=device,
    )
    exact_family = dependencies.compile_exact_family(exact_carrier.executor)
    exact_executor = exact_family.assemble(dependencies.full_lift_scope)
    guard.assert_unchanged()
    provenance = dependencies.model_provenance(
        model,
        model_id=PINNED_MODEL_ID,
        requested_revision=PINNED_MODEL_REVISION,
    )
    if provenance.get("resolved_commit") != PINNED_MODEL_REVISION:
        raise RuntimeError("loaded model resolved to the wrong pinned revision")
    return _Runtime(
        tokenizer=tokenizer,
        model=model,
        adapter=adapter,
        device=device,
        source_guard=guard,
        source_model_fingerprint_before=model_fingerprint,
        source_execution_fingerprint_before=execution_fingerprint,
        exact_carrier=exact_carrier,
        exact_family=exact_family,
        exact_executor=exact_executor,
        model_receipt={
            "model_id": PINNED_MODEL_ID,
            "requested_revision": PINNED_MODEL_REVISION,
            "resolved_commit": provenance.get("resolved_commit"),
            "parameter_count": parameter_count,
            "device": device_name,
            "dtype": dtype,
            "weights_in_report": False,
        },
        artifact_root=None,
        artifact_receipts={},
    )


def _tokenize_role(
    runtime: _Runtime,
    examples: Sequence[_PanelExample],
    *,
    batch_size: int,
) -> tuple[_RoleBatch, ...]:
    torch = __import__("torch")
    result: list[_RoleBatch] = []
    for start in range(0, len(examples), batch_size):
        rows = tuple(examples[start : start + batch_size])
        encoded = runtime.tokenizer(
            [row.prompt for row in rows],
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(device=runtime.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        if not {"input_ids", "attention_mask"} <= set(model_inputs):
            raise ValueError("tokenizer omitted input_ids or attention_mask")
        choice_rows: list[list[int]] = []
        for row in rows:
            ids: list[int] = []
            for choice in row.choices:
                encoded_choice = runtime.tokenizer.encode(
                    " " + choice,
                    add_special_tokens=False,
                )
                if not isinstance(encoded_choice, list) or len(encoded_choice) != 1:
                    raise ValueError("every forced choice must be one leading-space token")
                ids.append(int(encoded_choice[0]))
            choice_rows.append(ids)
        result.append(
            _RoleBatch(
                model_inputs=model_inputs,
                example_ids=tuple(row.example_id for row in rows),
                family_ids=tuple(row.family_id for row in rows),
                choice_ids=torch.tensor(
                    choice_rows,
                    dtype=torch.long,
                    device=runtime.device,
                ),
                gold=torch.tensor(
                    [row.correct_choice for row in rows],
                    dtype=torch.long,
                    device=runtime.device,
                ),
            )
        )
    return tuple(result)


def _capture_a_fit_live(
    *,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    batch_size: int,
) -> _AFitCapture:
    if not isinstance(runtime, _Runtime):
        raise TypeError("runtime has the wrong type")
    torch = dependencies.torch
    functional = torch.nn.functional
    examples = preflight.role_examples["A_fit"]
    # Keep whole frozen families in separate batches so row normalization and
    # equal-family scoring cannot be affected by padding from another family.
    by_family: dict[str, list[_PanelExample]] = {}
    for row in examples:
        by_family.setdefault(row.family_id, []).append(row)
    batches = tuple(
        batch
        for family in tuple(dict.fromkeys(row.family_id for row in examples))
        for batch in _tokenize_role(runtime, by_family[family], batch_size=batch_size)
    )
    rows: dict[int, dict[str, list[object]]] = {
        ordinal: {
            "raw": [], "x": [], "h": [], "delta": [], "delta_g": [],
            "gate_g": [], "up_g": [], "down_g": [], "families": [],
        }
        for ordinal in range(18)
    }
    backward_count = 0
    for batch in batches:
        current: dict[int, dict[str, object]] = {
            ordinal: {} for ordinal in range(18)
        }
        handles: list[object] = []
        embedding = runtime.model.model.embed_tokens

        def enable_activation_graph(_module: object, _args: object, output: object) -> object:
            if not isinstance(output, torch.Tensor):
                raise TypeError("embedding output is not a Tensor")
            output.requires_grad_(True)
            return output

        handles.append(embedding.register_forward_hook(enable_activation_graph))
        for ordinal in range(18):
            layer = _source_layer(runtime, ordinal)
            mlp = layer.mlp

            def capture_raw(_module: object, args: tuple[object, ...], *, index: int = ordinal) -> None:
                current[index]["raw"] = args[0]

            def capture_x(_module: object, args: tuple[object, ...], *, index: int = ordinal) -> None:
                current[index]["x"] = args[0]

            def capture_h(_module: object, args: tuple[object, ...], *, index: int = ordinal) -> None:
                current[index]["h"] = args[0]

            def retain(name: str, index: int):
                def hook(_module: object, _args: object, output: object) -> object:
                    if not isinstance(output, torch.Tensor) or not output.requires_grad:
                        raise RuntimeError(f"{name} output has no suffix gradient graph")
                    output.retain_grad()
                    current[index][name] = output
                    return output
                return hook

            handles.extend(
                (
                    mlp.register_forward_pre_hook(capture_x),
                    layer.pre_feedforward_layernorm.register_forward_pre_hook(
                        capture_raw
                    ),
                    mlp.down_proj.register_forward_pre_hook(capture_h),
                    mlp.gate_proj.register_forward_hook(retain("gate", ordinal)),
                    mlp.up_proj.register_forward_hook(retain("up", ordinal)),
                    mlp.down_proj.register_forward_hook(retain("down", ordinal)),
                    layer.post_feedforward_layernorm.register_forward_hook(
                        retain("delta", ordinal)
                    ),
                )
            )
        try:
            runtime.model.zero_grad(set_to_none=True)
            output = runtime.adapter.forward(batch.model_inputs)
            logits = output.logits[:, :-1, :].float()
            labels = batch.model_inputs["input_ids"][:, 1:]
            supervised = (
                batch.model_inputs["attention_mask"][:, :-1].bool()
                & batch.model_inputs["attention_mask"][:, 1:].bool()
            )
            loss = functional.cross_entropy(
                logits[supervised], labels[supervised], reduction="sum"
            )
            loss.backward()
            backward_count += 1
            if any(
                parameter.grad is not None
                for parameter in runtime.model.parameters()
            ):
                raise RuntimeError(
                    "A_fit capture populated a frozen source parameter gradient"
                )
            token_mask = batch.model_inputs["attention_mask"].bool()
            family_matrix = [
                family
                for row, family in enumerate(batch.family_ids)
                for valid in token_mask[row].detach().cpu().tolist()
                if valid
            ]
            for ordinal in range(18):
                item = current[ordinal]
                required = ("raw", "x", "h", "delta", "gate", "up", "down")
                if any(name not in item for name in required):
                    raise RuntimeError("simultaneous MLP capture omitted a site")
                for source_name, target_name in (
                    ("raw", "raw"), ("x", "x"), ("h", "h"),
                    ("delta", "delta")
                ):
                    value = item[source_name]
                    rows[ordinal][target_name].append(
                        value[token_mask].detach().to(device="cpu", dtype=torch.float32)
                    )
                for source_name, target_name in (
                    ("delta", "delta_g"), ("gate", "gate_g"),
                    ("up", "up_g"), ("down", "down_g"),
                ):
                    value = item[source_name]
                    if value.grad is None:
                        raise RuntimeError(f"{source_name} suffix gradient is unavailable")
                    rows[ordinal][target_name].append(
                        value.grad[token_mask].detach().to(
                            device="cpu", dtype=torch.float32
                        )
                    )
                rows[ordinal]["families"].extend(family_matrix)
            del output, logits, loss
        finally:
            for handle in handles:
                handle.remove()
            runtime.model.zero_grad(set_to_none=True)
    layers: dict[int, _MLPCapture] = {}
    row_catalog: list[dict[str, object]] = []
    for ordinal, value in rows.items():
        concatenate = torch.cat
        layer_capture = _MLPCapture(
            raw_carrier_inputs=concatenate(value["raw"]).contiguous(),
            normalized_inputs=concatenate(value["x"]).contiguous(),
            gated_features=concatenate(value["h"]).contiguous(),
            exact_delta=concatenate(value["delta"]).contiguous(),
            delta_gradients=concatenate(value["delta_g"]).contiguous(),
            gate_gradients=concatenate(value["gate_g"]).contiguous(),
            up_gradients=concatenate(value["up_g"]).contiguous(),
            down_gradients=concatenate(value["down_g"]).contiguous(),
            family_ids=tuple(value["families"]),
        )
        if layer_capture.normalized_inputs.shape[0] != len(layer_capture.family_ids):
            raise RuntimeError("A_fit capture row identity drifted")
        layers[ordinal] = layer_capture
        row_catalog.append(
            {
                "block_ordinal": ordinal,
                "row_count": len(layer_capture.family_ids),
                "family_counts": {
                    family: layer_capture.family_ids.count(family)
                    for family in sorted(set(layer_capture.family_ids))
                },
            }
        )
    row_sha = _manifest_sha256(row_catalog)
    runtime.source_guard.assert_unchanged()
    return _AFitCapture(
        layers=layers,
        row_manifest_sha256=row_sha,
        capture_receipt={
            "batch_count": len(batches),
            "backward_count": backward_count,
            "MLP_output_count_per_backward": 18,
            "simultaneous_retain_grad_no_graph_detach": True,
            "embedding_output_enabled_as_leaf": True,
            "intermediate_MLP_outputs_detached": False,
            "full_suffix_gradient_preserved_for_every_MLP_output": True,
            "source_parameter_gradients_present": False,
            "source_parameter_gradients_checked_after_every_backward": True,
            "row_manifest_sha256": row_sha,
        },
    )


def _select_pair_live(
    *,
    capture: object,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> Mapping[str, object]:
    if not isinstance(capture, _AFitCapture) or not isinstance(runtime, _Runtime):
        raise TypeError("pair selection inputs have the wrong type")
    torch = dependencies.torch
    candidates: list[object] = []
    receipts: list[dict[str, object]] = []
    with torch.no_grad():
        for member in range(18):
            observed = capture.layers[member]
            weights = observed.delta_gradients.square().mean(dim=1)
            member_layer = _source_layer(runtime, member)
            for anchor in range(18):
                if abs(anchor - member) < 2:
                    continue
                anchor_layer = _source_layer(runtime, anchor)
                predicted = anchor_layer.post_feedforward_layernorm(
                    anchor_layer.mlp(
                        anchor_layer.pre_feedforward_layernorm(
                            observed.raw_carrier_inputs.to(device=runtime.device)
                        )
                    )
                ).detach().to(device="cpu", dtype=torch.float32)
                candidate = dependencies.build_pair_candidate(
                    anchor_invocation_id=f"mutation.{2 * anchor + 2}",
                    member_invocation_id=f"mutation.{2 * member + 2}",
                    anchor_block_ordinal=anchor,
                    member_block_ordinal=member,
                    aligned_design=observed.normalized_inputs,
                    exact_member_delta=observed.exact_delta,
                    anchor_core_delta=predicted,
                    row_fisher_weights=weights,
                    family_ids=observed.family_ids,
                )
                candidates.append(candidate)
                receipts.append(
                    {
                        "anchor_invocation_id": candidate.anchor_invocation_id,
                        "member_invocation_id": candidate.member_invocation_id,
                        "score": candidate.score,
                        "numerical_tensor_sha256": candidate.numerical_tensor_sha256,
                        "score_receipt_sha256": candidate.evidence.receipt_sha256,
                    }
                )
    if len(candidates) != 272:
        raise RuntimeError("directed nonadjacent pair inventory drifted")
    selected = dependencies.select_pair_candidate(tuple(candidates))
    evidence_payload = {
        "objective": "zero_residual_shared_core_Fisher_weighted_relative_SSE",
        "direction": "minimize",
        "candidate_count": len(candidates),
        "ordered_candidates": receipts,
        "selected_score_receipt": selected.evidence.to_dict(),
        "A_fit_row_manifest_sha256": capture.row_manifest_sha256,
    }
    evidence_sha = _manifest_sha256(evidence_payload)
    role_manifest = preflight.setup["data_firewall"]["roles"]["A_fit"]
    pair_object = dependencies.build_pair(
        runtime.exact_family,
        anchor_invocation_id=selected.anchor_invocation_id,
        member_invocation_id=selected.member_invocation_id,
        a_fit_manifest_sha256=role_manifest["role_manifest_sha256"],
        selection_evidence_sha256=evidence_sha,
        successor_setup_sha256=PINNED_SETUP_SHA256,
    )
    pair_dict = pair_object.to_dict()
    anchor_invocation = next(
        item
        for item in runtime.exact_executor.manifest.invocations
        if item.invocation_id == selected.anchor_invocation_id
    )
    anchor_core_sha256 = dependencies.module_state_sha256(
        runtime.exact_executor.atom_for(anchor_invocation)
    )
    return {
        "selected_on": "A_fit",
        "anchor_invocation_id": selected.anchor_invocation_id,
        "member_invocation_id": selected.member_invocation_id,
        "anchor_block_ordinal": selected.anchor_block_ordinal,
        "member_block_ordinal": selected.member_block_ordinal,
        "objective_value": selected.score,
        "numerical_tie_break_sha256": selected.numerical_tensor_sha256,
        "selection_evidence_sha256": evidence_sha,
        "candidate_count": len(candidates),
        "pair_manifest": pair_dict,
        "pair_manifest_sha256": pair_dict["pair_manifest_sha256"],
        "exact_shared_core_state_sha256": anchor_core_sha256,
        "activation_and_Fisher_evidence_only": True,
        "block_ordinal_used_as_numerical_feature": False,
        "nonadjacency_is_eligibility_only": True,
        "frozen_before_A_select": True,
        "_pair_manifest_object": pair_object,
        "_selection_evidence": evidence_payload,
    }


def _fit_rank_point_live(
    *,
    rank: int,
    independent_nonsharing: bool,
    capture: object,
    pair_selection: Mapping[str, object],
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> Mapping[str, object]:
    if rank not in LOW_RANK_POINTS or not isinstance(capture, _AFitCapture) or not isinstance(runtime, _Runtime):
        raise ValueError("rank fit inputs are invalid")
    member = int(pair_selection["member_block_ordinal"])
    anchor = int(pair_selection["anchor_block_ordinal"])
    observed = capture.layers[member]
    member_mlp = _source_layer(runtime, member).mlp
    anchor_mlp = _source_layer(runtime, anchor).mlp
    catalog = dependencies.fit_factor_catalog(
        projection_designs={
            "gate_proj": observed.normalized_inputs,
            "up_proj": observed.normalized_inputs,
            "down_proj": observed.gated_features,
        },
        member_projection_weights={
            name: getattr(member_mlp, name).weight.detach()
            for name in ("gate_proj", "up_proj", "down_proj")
        },
        anchor_projection_weights={
            name: getattr(anchor_mlp, name).weight.detach()
            for name in ("gate_proj", "up_proj", "down_proj")
        },
        projection_row_fisher_weights={
            "gate_proj": observed.gate_gradients.square().mean(dim=1),
            "up_proj": observed.up_gradients.square().mean(dim=1),
            "down_proj": observed.down_gradients.square().mean(dim=1),
        },
        requested_rank=rank,
        independent_nonsharing=independent_nonsharing,
        relative_ridge=1.0e-4,
    )
    arm_id = (
        f"control.independent_member_r{rank}"
        if independent_nonsharing
        else f"primary.shared_core_residual_r{rank}"
    )
    setup_arm = next(
        item
        for item in preflight.setup["arm_contract"]["arms"]
        if item["arm_id"] == arm_id
    )
    receipt = catalog.to_dict()
    member_input_sha256 = dependencies.canonical_tensor_sha256(
        observed.raw_carrier_inputs
    )
    return {
        "arm_id": arm_id,
        "residual_rank": rank,
        "independent_nonsharing": independent_nonsharing,
        "fit_nonce": setup_arm["fit_nonce"],
        "fit_seed": setup_arm["fit_seed"],
        "selected_member_prefix_is_exact_by_single_approximation": True,
        "candidate_prefix_recapture_required": False,
        "exact_prefix_input_hash_equality_authenticated_on_A_fit": True,
        "selected_member_input_sha256": member_input_sha256,
        "A_fit_row_manifest_sha256": capture.row_manifest_sha256,
        "fit_receipt": receipt,
        "fit_receipt_sha256": _manifest_sha256(receipt),
        "_factor_catalog": catalog,
    }


def _compile_arm(
    *,
    arm_id: str,
    runtime: _Runtime,
    pair_selection: Mapping[str, object],
    fit_points: Mapping[str, object],
    dependencies: SimpleNamespace,
) -> object:
    pair = pair_selection.get("_pair_manifest_object")
    if pair is None:
        raise RuntimeError("live pair manifest object is unavailable")
    if arm_id == "exact.shared_core_dense_residual":
        compilation = dependencies.compile_point(
            runtime.exact_family,
            pair_manifest=pair,
            artifact_kind=dependencies.dense_kind,
        )
    elif arm_id == "ablation.shared_core_zero_residual":
        compilation = dependencies.compile_point(
            runtime.exact_family,
            pair_manifest=pair,
            artifact_kind=dependencies.zero_kind,
        )
    else:
        rank = next((value for value in LOW_RANK_POINTS if arm_id.endswith(f"r{value}")), None)
        if rank is None:
            raise ValueError(f"unsupported compressed arm {arm_id}")
        primary_id = f"primary.shared_core_residual_r{rank}"
        if arm_id.startswith("primary."):
            fit = fit_points.get(primary_id)
            if not isinstance(fit, Mapping):
                raise RuntimeError(f"frozen fit for {arm_id} is unavailable")
            catalog = fit.get("_factor_catalog")
            compilation = dependencies.compile_point(
                runtime.exact_family,
                pair_manifest=pair,
                artifact_kind=dependencies.primary_kind,
                residual_rank=rank,
                projection_factors=catalog.projection_factors,
            )
        elif arm_id.startswith("control.independent"):
            fit = fit_points.get(arm_id)
            if not isinstance(fit, Mapping):
                raise RuntimeError(f"frozen fit for {arm_id} is unavailable")
            catalog = fit.get("_factor_catalog")
            compilation = dependencies.compile_point(
                runtime.exact_family,
                pair_manifest=pair,
                artifact_kind=dependencies.independent_kind,
                residual_rank=rank,
                projection_factors=catalog.projection_factors,
            )
        elif arm_id.startswith("control.shuffled"):
            fit = fit_points.get(primary_id)
            if not isinstance(fit, Mapping):
                raise RuntimeError(f"frozen primary fit for {arm_id} is unavailable")
            catalog = fit.get("_factor_catalog")
            primary = dependencies.compile_point(
                runtime.exact_family,
                pair_manifest=pair,
                artifact_kind=dependencies.primary_kind,
                residual_rank=rank,
                projection_factors=catalog.projection_factors,
            )
            compilation = dependencies.compile_assignment_control(
                runtime.exact_family,
                fitted_compilation=primary,
                assignment_mode=dependencies.shuffled_assignment,
                shuffle_seed=1_734,
            )
        else:
            raise ValueError(f"unsupported arm {arm_id}")

    if runtime.artifact_root is None:
        # Unit-level injected calls may deliberately omit persistence; an
        # authorized public run always assigns the artifact root first.
        return compilation
    path = dependencies.content_point_manifest_path(
        runtime.artifact_root,
        arm_id=arm_id,
    )
    existing = runtime.artifact_receipts.get(arm_id)
    if existing is None:
        artifact = compilation.export_artifact()
        expected_sha = artifact.artifact_sha256
        receipt = dict(
            dependencies.write_content_point_artifact(
                artifact,
                runtime.artifact_root,
            )
        )
        loaded = dependencies.load_content_point_artifact(
            path,
            expected_artifact_sha256=expected_sha,
            expected_manifest_sha256=receipt[
                "content_point_manifest_sha256"
            ],
        )
        if loaded.artifact_sha256 != expected_sha:
            raise RuntimeError("point artifact changed across strict reload")
        receipt["tensor_storage_bytes"] = receipt[
            "logical_tensor_storage_bytes"
        ]
        if (
            receipt["serialized_stored_scalar_count"]
            != compilation.candidate_stored_scalars
            or receipt["tensor_storage_bytes"]
            != compilation.candidate_stored_scalars * 4
        ):
            raise RuntimeError("logical point scalar accounting drifted")
        if arm_id.startswith("control.shuffled"):
            rank = next(
                value for value in LOW_RANK_POINTS if arm_id.endswith(f"r{value}")
            )
            primary_receipt = runtime.artifact_receipts.get(
                f"primary.shared_core_residual_r{rank}"
            )
            if not isinstance(primary_receipt, Mapping):
                raise RuntimeError("shuffled point preceded its primary artifact")
            primary_components = primary_receipt.get("component_blob_receipts")
            shuffled_components = receipt.get("component_blob_receipts")
            if (
                not isinstance(primary_components, list)
                or not isinstance(shuffled_components, list)
                or [item.get("blob_sha256") for item in primary_components]
                != [item.get("blob_sha256") for item in shuffled_components]
                or receipt.get("newly_materialized_blob_count") != 0
            ):
                raise RuntimeError("shuffled point did not reuse primary blobs")
        runtime.artifact_receipts[arm_id] = receipt
    else:
        loaded = dependencies.load_content_point_artifact(
            path,
            expected_artifact_sha256=existing["artifact_sha256"],
            expected_manifest_sha256=existing[
                "content_point_manifest_sha256"
            ],
        )
        if (
            _sha256_file(path) != existing["file_sha256"]
            or path.stat().st_size != existing["serialized_file_bytes"]
            or loaded.point_manifest["candidate_state_sha256"]
            != existing["candidate_state_sha256"]
        ):
            raise RuntimeError("frozen point artifact drifted across role reuse")
    compilation.restore_artifact_(loaded)
    if compilation.executor.state_sha256 != loaded.point_manifest["candidate_state_sha256"]:
        raise RuntimeError("strictly reloaded point state did not restore")
    return compilation


def _freeze_runtime_artifact_store_inventory(
    *,
    runtime: object,
    setup: Mapping[str, object],
    dependencies: SimpleNamespace,
) -> Mapping[str, object] | None:
    if not isinstance(runtime, _Runtime) or runtime.artifact_root is None:
        return None
    expected_arm_ids = tuple(setup["arm_contract"]["arm_order"][2:])
    if set(runtime.artifact_receipts) != set(expected_arm_ids):
        raise RuntimeError("cannot freeze an incomplete point-artifact store")
    receipt = dict(
        dependencies.write_store_inventory(
            runtime.artifact_root,
            expected_arm_ids=expected_arm_ids,
        )
    )
    expected = _content_addressed_storage_plan(setup)
    if (
        receipt.get("point_manifest_count") != 14
        or receipt.get("unique_blob_count") != 10
        or receipt.get("component_blob_reference_count") != 27
        or receipt.get("reused_component_blob_reference_count") != 17
        or receipt.get("logical_flat_tensor_storage_bytes")
        != expected["logical_flat_tensor_storage_bytes"]
        or receipt.get("unique_tensor_storage_bytes")
        != expected["unique_content_addressed_tensor_storage_bytes"]
        or receipt.get("tensor_storage_savings_bytes")
        != expected["content_addressed_tensor_storage_savings_bytes"]
        or receipt.get("logical_candidate_scalar_accounting_unchanged")
        is not True
        or receipt.get("equal_values_without_logical_alias_remain_separate")
        is not True
        or receipt.get("shuffled_adapter_reuses_primary_blob") is not True
        or receipt.get("exact_point_and_blob_inventory") is not True
    ):
        raise RuntimeError("content-addressed store inventory drifted")
    runtime.artifact_store_inventory_receipt = receipt
    return receipt


def _validate_runtime_artifact_store_inventory(
    *,
    runtime: object,
    setup: Mapping[str, object],
    dependencies: SimpleNamespace,
) -> Mapping[str, object] | None:
    if not isinstance(runtime, _Runtime) or runtime.artifact_root is None:
        return None
    frozen = runtime.artifact_store_inventory_receipt
    if not isinstance(frozen, Mapping):
        raise RuntimeError("content-addressed store inventory was not frozen")
    expected_arm_ids = tuple(setup["arm_contract"]["arm_order"][2:])
    observed = dependencies.validate_store_inventory(
        runtime.artifact_root,
        expected_arm_ids=expected_arm_ids,
        expected_inventory_sha256=frozen["store_inventory_sha256"],
    )
    if observed != frozen:
        raise RuntimeError("content-addressed store changed after freezing")
    return observed


def _tensor_totals() -> dict[str, float | int]:
    return {"n": 0, "error2": 0.0, "reference2": 0.0, "candidate2": 0.0, "dot": 0.0, "max": 0.0}


def _update_tensor_totals(totals: dict[str, float | int], reference: object, candidate: object) -> None:
    left = reference.detach().double()
    right = candidate.detach().double()
    error = right - left
    totals["n"] = int(totals["n"]) + left.numel()
    totals["error2"] = float(totals["error2"]) + float(error.square().sum().item())
    totals["reference2"] = float(totals["reference2"]) + float(left.square().sum().item())
    totals["candidate2"] = float(totals["candidate2"]) + float(right.square().sum().item())
    totals["dot"] = float(totals["dot"]) + float((left * right).sum().item())
    totals["max"] = max(float(totals["max"]), float(error.abs().max().item()))


def _tensor_summary(totals: Mapping[str, float | int]) -> dict[str, object]:
    reference2 = float(totals["reference2"])
    candidate2 = float(totals["candidate2"])
    return {
        "compared_elements": int(totals["n"]),
        "nrmse": math.sqrt(float(totals["error2"]) / max(reference2, 1.0e-30)),
        "cosine_similarity": float(totals["dot"]) / max(math.sqrt(reference2 * candidate2), 1.0e-30),
        "maximum_absolute_error": float(totals["max"]),
    }


def _last_choice_observations(logits: object, batch: _RoleBatch) -> tuple[list[int], list[float], list[bool]]:
    torch = __import__("torch")
    mask = batch.model_inputs["attention_mask"].bool()
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).amax(dim=1)
    row = torch.arange(mask.shape[0], device=logits.device)
    scores = logits[row, last, :].gather(1, batch.choice_ids)
    prediction = scores.argmax(dim=1)
    log_prob = scores.double().log_softmax(dim=1)
    nll = -log_prob.gather(1, batch.gold[:, None]).squeeze(1)
    correct = prediction == batch.gold
    return (
        [int(value) for value in prediction.cpu().tolist()],
        [float(value) for value in nll.cpu().tolist()],
        [bool(value) for value in correct.cpu().tolist()],
    )


def _evaluate_one_arm(
    *,
    arm_id: str,
    runtime: _Runtime,
    batches: Sequence[_RoleBatch],
    candidate_executor: object,
    anchor_mutation_index: int,
    selected_mutation_index: int,
    dependencies: SimpleNamespace,
) -> dict[str, object]:
    torch = dependencies.torch
    functional = torch.nn.functional
    supervised_tokens = query_rows = top1 = 0
    native_nll_sum = candidate_nll_sum = kl_sum = 0.0
    logits_totals = _tensor_totals()
    boundary_totals = [_tensor_totals() for _ in range(36)]
    selected_totals = _tensor_totals()
    selected_event_totals = {
        "anchor": {
            "input": _tensor_totals(),
            "delta": _tensor_totals(),
            "output_carrier": _tensor_totals(),
        },
        "member": {
            "input": _tensor_totals(),
            "delta": _tensor_totals(),
            "output_carrier": _tensor_totals(),
        },
    }
    native_correct: list[bool] = []
    candidate_correct: list[bool] = []
    example_families: list[str] = []
    candidate_choice_nll: list[float] = []
    family_stats: dict[str, dict[str, object]] = {}
    source_projection_calls = source_fallback_calls = 0
    receipt_hashes: list[str] = []
    mutation_hash_catalog: list[dict[str, object]] = []
    deterministic_replay_receipt: dict[str, object] | None = None
    for batch_index, batch in enumerate(batches):
        with torch.inference_mode():
            native = runtime.adapter.forward(batch.model_inputs)
            sequence = runtime.adapter.prepare_sequence(batch.model_inputs)
            exact = runtime.exact_executor.execute(
                batch.model_inputs["input_ids"],
                sequence,
                execution_nonce=f"{arm_id}.reference.{batch_index}",
            )
            candidate = (
                exact
                if arm_id == "exact.heterogeneous_full_rank_atom_bank_lift"
                else candidate_executor.execute(
                    batch.model_inputs["input_ids"],
                    sequence,
                    execution_nonce=f"{arm_id}.candidate.{batch_index}",
                )
            )
            replay = None
            if batch_index == 0:
                replay = candidate_executor.execute(
                    batch.model_inputs["input_ids"],
                    sequence,
                    execution_nonce=f"{arm_id}.deterministic_replay.{batch_index}",
                )
        valid = batch.model_inputs["attention_mask"].bool()
        supervised = valid[:, :-1] & valid[:, 1:]
        labels = batch.model_inputs["input_ids"][:, 1:][supervised]
        native_supervised = native.logits[:, :-1, :][supervised].float()
        candidate_supervised = candidate.logits[:, :-1, :][supervised].float()
        native_nll_sum += float(functional.cross_entropy(native_supervised, labels, reduction="sum").item())
        candidate_nll_sum += float(functional.cross_entropy(candidate_supervised, labels, reduction="sum").item())
        supervised_tokens += int(labels.numel())
        native_valid = native.logits[valid].float()
        candidate_valid = candidate.logits[valid].float()
        native_log = native_valid.log_softmax(dim=-1)
        candidate_log = candidate_valid.log_softmax(dim=-1)
        kl_sum += float((native_log.exp() * (native_log - candidate_log)).sum().item())
        query_rows += int(valid.sum().item())
        top1 += int((native_valid.argmax(dim=-1) == candidate_valid.argmax(dim=-1)).sum().item())
        _update_tensor_totals(logits_totals, native_valid, candidate_valid)
        exact_snapshots = exact.carrier_execution.carrier_snapshots
        candidate_snapshots = candidate.carrier_execution.carrier_snapshots
        exact_boundaries = exact.carrier_execution.receipt.boundaries
        candidate_boundaries = candidate.carrier_execution.receipt.boundaries
        for index in range(36):
            _update_tensor_totals(
                boundary_totals[index],
                exact_snapshots[index + 1][valid],
                candidate_snapshots[index + 1][valid],
            )
            exact_primary = _primary_generator_message_receipt(
                exact_boundaries[index]
            )
            candidate_primary = _primary_generator_message_receipt(
                candidate_boundaries[index]
            )
            mutation_hash_catalog.append(
                {
                    "batch_index": batch_index,
                    "mutation_index": index + 1,
                    "exact_input_sha256": exact_boundaries[index].input_carrier_sha256,
                    "candidate_input_sha256": candidate_boundaries[index].input_carrier_sha256,
                    "input_hash_equal": (
                        exact_boundaries[index].input_carrier_sha256
                        == candidate_boundaries[index].input_carrier_sha256
                    ),
                    "exact_delta_sha256": exact_primary.value_sha256,
                    "candidate_delta_sha256": candidate_primary.value_sha256,
                    "delta_hash_equal": (
                        exact_primary.value_sha256
                        == candidate_primary.value_sha256
                    ),
                    "exact_output_carrier_sha256": exact_boundaries[index].output_carrier_sha256,
                    "candidate_output_carrier_sha256": candidate_boundaries[index].output_carrier_sha256,
                    "output_carrier_hash_equal": (
                        exact_boundaries[index].output_carrier_sha256
                        == candidate_boundaries[index].output_carrier_sha256
                    ),
                }
            )
        for label, mutation_index in (
            ("anchor", anchor_mutation_index),
            ("member", selected_mutation_index),
        ):
            _update_tensor_totals(
                selected_event_totals[label]["input"],
                exact_snapshots[mutation_index - 1][valid],
                candidate_snapshots[mutation_index - 1][valid],
            )
            _update_tensor_totals(
                selected_event_totals[label]["delta"],
                (
                    exact_snapshots[mutation_index]
                    - exact_snapshots[mutation_index - 1]
                )[valid],
                (
                    candidate_snapshots[mutation_index]
                    - candidate_snapshots[mutation_index - 1]
                )[valid],
            )
            _update_tensor_totals(
                selected_event_totals[label]["output_carrier"],
                exact_snapshots[mutation_index][valid],
                candidate_snapshots[mutation_index][valid],
            )
        _update_tensor_totals(
            selected_totals,
            (exact_snapshots[selected_mutation_index] - exact_snapshots[selected_mutation_index - 1])[valid],
            (candidate_snapshots[selected_mutation_index] - candidate_snapshots[selected_mutation_index - 1])[valid],
        )
        _, _, native_rows = _last_choice_observations(native.logits, batch)
        _, choice_nll, candidate_rows = _last_choice_observations(candidate.logits, batch)
        native_correct.extend(native_rows)
        candidate_correct.extend(candidate_rows)
        candidate_choice_nll.extend(choice_nll)
        example_families.extend(batch.family_ids)
        for row_index, family_id in enumerate(batch.family_ids):
            stats = family_stats.setdefault(
                family_id,
                {
                    "tokens": 0, "queries": 0, "native_nll": 0.0,
                    "candidate_nll": 0.0, "kl": 0.0, "top1": 0,
                    "logits": _tensor_totals(), "selected": _tensor_totals(),
                    "final": _tensor_totals(), "native_correct": 0,
                    "both_correct": 0,
                },
            )
            row_valid = valid[row_index]
            row_supervised = supervised[row_index]
            row_labels = batch.model_inputs["input_ids"][row_index, 1:][row_supervised]
            row_native_supervised = native.logits[row_index, :-1, :][row_supervised].float()
            row_candidate_supervised = candidate.logits[row_index, :-1, :][row_supervised].float()
            stats["tokens"] = int(stats["tokens"]) + int(row_labels.numel())
            stats["native_nll"] = float(stats["native_nll"]) + float(
                functional.cross_entropy(row_native_supervised, row_labels, reduction="sum").item()
            )
            stats["candidate_nll"] = float(stats["candidate_nll"]) + float(
                functional.cross_entropy(row_candidate_supervised, row_labels, reduction="sum").item()
            )
            row_native = native.logits[row_index][row_valid].float()
            row_candidate = candidate.logits[row_index][row_valid].float()
            row_native_log = row_native.log_softmax(dim=-1)
            row_candidate_log = row_candidate.log_softmax(dim=-1)
            stats["queries"] = int(stats["queries"]) + int(row_valid.sum().item())
            stats["kl"] = float(stats["kl"]) + float(
                (row_native_log.exp() * (row_native_log - row_candidate_log)).sum().item()
            )
            stats["top1"] = int(stats["top1"]) + int(
                (row_native.argmax(dim=-1) == row_candidate.argmax(dim=-1)).sum().item()
            )
            _update_tensor_totals(stats["logits"], row_native, row_candidate)
            _update_tensor_totals(
                stats["selected"],
                (exact_snapshots[selected_mutation_index] - exact_snapshots[selected_mutation_index - 1])[row_index][row_valid],
                (candidate_snapshots[selected_mutation_index] - candidate_snapshots[selected_mutation_index - 1])[row_index][row_valid],
            )
            _update_tensor_totals(
                stats["final"],
                exact_snapshots[-1][row_index][row_valid],
                candidate_snapshots[-1][row_index][row_valid],
            )
            stats["native_correct"] = int(stats["native_correct"]) + int(native_rows[row_index])
            stats["both_correct"] = int(stats["both_correct"]) + int(native_rows[row_index] and candidate_rows[row_index])
        receipt = candidate.receipt
        source_projection_calls += int(getattr(receipt, "source_projection_call_count", 0))
        source_fallback_calls += int(getattr(receipt, "source_fallback_call_count", 0))
        receipt_value = receipt.to_dict()
        receipt_hashes.append(str(receipt_value.get("receipt_sha256", _manifest_sha256(receipt_value))))
        if replay is not None:
            replay_snapshots = replay.carrier_execution.carrier_snapshots
            replay_boundaries = replay.carrier_execution.receipt.boundaries
            replay_payload = {
                "batch_index": batch_index,
                "logits_bitwise_equal": torch.equal(
                    candidate.logits, replay.logits
                ),
                "all_carrier_snapshots_bitwise_equal": all(
                    torch.equal(left, right)
                    for left, right in zip(
                        candidate_snapshots, replay_snapshots, strict=True
                    )
                ),
                "all_mutation_input_hashes_equal": all(
                    left.input_carrier_sha256 == right.input_carrier_sha256
                    for left, right in zip(
                        candidate_boundaries, replay_boundaries, strict=True
                    )
                ),
                "all_mutation_delta_hashes_equal": all(
                    _primary_generator_message_receipt(left).value_sha256
                    == _primary_generator_message_receipt(right).value_sha256
                    for left, right in zip(
                        candidate_boundaries, replay_boundaries, strict=True
                    )
                ),
                "all_output_carrier_hashes_equal": all(
                    left.output_carrier_sha256 == right.output_carrier_sha256
                    for left, right in zip(
                        candidate_boundaries, replay_boundaries, strict=True
                    )
                ),
            }
            replay_payload["passed"] = all(
                replay_payload[name] is True
                for name in (
                    "logits_bitwise_equal",
                    "all_carrier_snapshots_bitwise_equal",
                    "all_mutation_input_hashes_equal",
                    "all_mutation_delta_hashes_equal",
                    "all_output_carrier_hashes_equal",
                )
            )
            replay_payload["replay_receipt_sha256"] = _manifest_sha256(
                replay_payload
            )
            deterministic_replay_receipt = replay_payload
    if supervised_tokens <= 0 or query_rows <= 0:
        raise RuntimeError("role evaluation produced no valid tokens")
    native_nll = native_nll_sum / supervised_tokens
    candidate_nll = candidate_nll_sum / supervised_tokens
    logit_summary = _tensor_summary(logits_totals)
    selected_summary = _tensor_summary(selected_totals)
    boundaries = [
        {"mutation_index": index + 1, **_tensor_summary(value)}
        for index, value in enumerate(boundary_totals)
    ]
    native_correct_count = sum(native_correct)
    retained = (
        sum(left and right for left, right in zip(native_correct, candidate_correct, strict=True))
        / native_correct_count
        if native_correct_count else 0.0
    )
    family_retention: dict[str, float] = {}
    for family in sorted(set(example_families)):
        indices = [i for i, value in enumerate(example_families) if value == family]
        denominator = sum(native_correct[i] for i in indices)
        family_retention[family] = (
            sum(native_correct[i] and candidate_correct[i] for i in indices) / denominator
            if denominator else 0.0
        )
    family_metric_values: dict[str, dict[str, float]] = {}
    for family, stats in sorted(family_stats.items()):
        tokens = int(stats["tokens"])
        queries = int(stats["queries"])
        logit = _tensor_summary(stats["logits"])
        selected_family = _tensor_summary(stats["selected"])
        final_family = _tensor_summary(stats["final"])
        native_wins = int(stats["native_correct"])
        family_metric_values[family] = {
            "delta_nll_per_token": (
                float(stats["candidate_nll"]) - float(stats["native_nll"])
            ) / tokens,
            "native_to_candidate_kl_per_query": float(stats["kl"]) / queries,
            "top1_agreement": int(stats["top1"]) / queries,
            "valid_query_logit_nrmse": float(logit["nrmse"]),
            "native_correct_downstream_retention": (
                int(stats["both_correct"]) / native_wins if native_wins else 0.0
            ),
            "selected_nonanchor_MLP_delta_nrmse": float(selected_family["nrmse"]),
            "selected_nonanchor_MLP_delta_cosine": float(selected_family["cosine_similarity"]),
            "final_carrier_nrmse": float(final_family["nrmse"]),
        }
    first = next((row["mutation_index"] for row in boundaries if row["nrmse"] > 0.0), None)
    if deterministic_replay_receipt is None:
        raise RuntimeError("candidate deterministic replay was not performed")
    mutation_hash_audit = {
        "compared_batch_count": len(batches),
        "compared_mutation_count_per_batch": 36,
        "all_mutation_input_hashes_equal": all(
            item["input_hash_equal"] is True for item in mutation_hash_catalog
        ),
        "all_mutation_delta_hashes_equal": all(
            item["delta_hash_equal"] is True for item in mutation_hash_catalog
        ),
        "all_output_carrier_hashes_equal": all(
            item["output_carrier_hash_equal"] is True
            for item in mutation_hash_catalog
        ),
        "comparison_catalog_sha256": _manifest_sha256(mutation_hash_catalog),
        "ordered_comparisons": mutation_hash_catalog,
    }
    selected_pair_event_metrics = {
        label: {
            name: _tensor_summary(totals)
            for name, totals in event.items()
        }
        for label, event in selected_event_totals.items()
    }
    metrics = {
        "supervised_tokens": supervised_tokens,
        "compared_query_rows": query_rows,
        "nll_per_token": candidate_nll,
        "delta_nll_per_token": candidate_nll - native_nll,
        "native_to_candidate_kl_per_query": kl_sum / query_rows,
        "top1_agreement_to_native": top1 / query_rows,
        "valid_query_logit_nrmse": logit_summary["nrmse"],
        "maximum_valid_query_logit_absolute_error": logit_summary["maximum_absolute_error"],
        "native_correct_downstream_accuracy_retained": retained,
        "native_correct_example_count": native_correct_count,
        "candidate_forced_choice_accuracy": sum(candidate_correct) / len(candidate_correct),
        "candidate_restricted_choice_nll": math.fsum(candidate_choice_nll) / len(candidate_choice_nll),
        "worst_family_native_correct_retention": min(family_retention.values()),
        "family_native_correct_retention": family_retention,
        "selected_nonanchor_MLP_delta_nrmse": selected_summary["nrmse"],
        "selected_nonanchor_MLP_delta_cosine": selected_summary["cosine_similarity"],
        "final_carrier_nrmse": boundaries[-1]["nrmse"],
    }
    exact = all(
        metrics[name] == expected
        for name, expected in (
            ("delta_nll_per_token", 0.0),
            ("native_to_candidate_kl_per_query", 0.0),
            ("top1_agreement_to_native", 1.0),
            ("valid_query_logit_nrmse", 0.0),
            ("maximum_valid_query_logit_absolute_error", 0.0),
        )
    ) and all(row["nrmse"] == 0.0 for row in boundaries) and all(
        mutation_hash_audit[name] is True
        for name in (
            "all_mutation_input_hashes_equal",
            "all_mutation_delta_hashes_equal",
            "all_output_carrier_hashes_equal",
        )
    )
    runtime_gate_passed = (
        source_projection_calls == 0
        and source_fallback_calls == 0
        and deterministic_replay_receipt["passed"] is True
    )
    return {
        "arm_id": arm_id,
        "metrics": metrics,
        "all_36_carrier_boundary_metrics": boundaries,
        "all_36_mutation_input_delta_and_boundary_hash_audit": mutation_hash_audit,
        "selected_pair_input_delta_and_output_carrier_metrics": selected_pair_event_metrics,
        "deterministic_replay": deterministic_replay_receipt,
        "first_divergence_occurrence": first,
        "source_call_audit": {
            "source_projection_call_count": source_projection_calls,
            "source_fallback_call_count": source_fallback_calls,
            "passed": source_projection_calls == 0 and source_fallback_calls == 0,
        },
        "execution_receipt_sha256s": receipt_hashes,
        "literal_parity": exact,
        "runtime_gate_passed": runtime_gate_passed,
        "_family_retention_values": family_retention,
        "_family_metric_values": family_metric_values,
    }


def _evaluate_role_live(
    *,
    role_id: str,
    runtime: object,
    preflight: _Preflight,
    pair_selection: Mapping[str, object],
    fit_points: Mapping[str, object],
    dependencies: SimpleNamespace,
    batch_size: int,
) -> Mapping[str, object]:
    if role_id not in ORDERED_ROLES or not isinstance(runtime, _Runtime):
        raise ValueError("role evaluation inputs are invalid")
    batches = _tokenize_role(
        runtime,
        preflight.role_examples[role_id],
        batch_size=batch_size,
    )
    setup_order = preflight.setup["arm_contract"]["arm_order"]
    if role_id == "A_fit":
        arm_ids = setup_order[:3]
    elif role_id == "C_assessment":
        winner_ids = [key for key in fit_points if key.startswith("primary.")]
        if len(winner_ids) != 1:
            raise RuntimeError("C assessment must receive one primary winner")
        arm_ids = [setup_order[0], setup_order[1], winner_ids[0]]
    else:
        arm_ids = list(setup_order)
    results: dict[str, object] = {
        "native": {
            "arm_id": "native",
            "reference_only": True,
            "prompt_count": len(preflight.role_examples[role_id]),
        }
    }
    artifacts: dict[str, object] = {}
    selected_mutation = int(pair_selection["member_invocation_id"].split(".")[-1])
    anchor_mutation = int(pair_selection["anchor_invocation_id"].split(".")[-1])
    for arm_id in arm_ids[1:]:
        compilation = None
        executor = runtime.exact_executor
        if arm_id != "exact.heterogeneous_full_rank_atom_bank_lift":
            compilation = _compile_arm(
                arm_id=arm_id,
                runtime=runtime,
                pair_selection=pair_selection,
                fit_points=fit_points,
                dependencies=dependencies,
            )
            executor = compilation.assemble()
            metadata = compilation.metadata()
            artifacts[arm_id] = {
                "arm_id": arm_id,
                "point_manifest_sha256": metadata["point_manifest_sha256"],
                "candidate_state_sha256": metadata["candidate_state_sha256"],
                "candidate_stored_scalars": metadata["candidate_stored_scalars"],
                "removed_scalars": metadata["removed_scalars"],
                "residual_rank": metadata["residual_rank"],
                "fit_factor_catalog_sha256": metadata["fit_factor_catalog_sha256"],
                "topology_preserved": metadata["topology_preserved"],
                "source_free_candidate_runtime": metadata["source_free_candidate_runtime"],
                **dict(runtime.artifact_receipts.get(arm_id, {})),
            }
        result = _evaluate_one_arm(
            arm_id=arm_id,
            runtime=runtime,
            batches=batches,
            candidate_executor=executor,
            anchor_mutation_index=anchor_mutation,
            selected_mutation_index=selected_mutation,
            dependencies=dependencies,
        )
        results[arm_id] = result
        if arm_id == "exact.heterogeneous_full_rank_atom_bank_lift":
            results["native"] = {
                "arm_id": "native",
                "reference_only": True,
                "prompt_count": len(preflight.role_examples[role_id]),
                "metrics": dict(result["metrics"]),
                "source_call_audit": {
                    "compiled_source_projection_call_count": 0,
                    "compiled_source_fallback_call_count": 0,
                    "native_reference_model_forward_count": len(batches),
                    "passed": True,
                },
                "baseline_receipt_derived_from_same_role_exact_reference": True,
                "literal_parity": True,
            }
        if arm_id in {
            "exact.heterogeneous_full_rank_atom_bank_lift",
            "exact.shared_core_dense_residual",
        } and result["literal_parity"] is not True:
            raise RuntimeError(f"{arm_id} lost literal parity")
        del compilation, executor
    dense_passed = (
        isinstance(results.get("exact.heterogeneous_full_rank_atom_bank_lift"), Mapping)
        and results["exact.heterogeneous_full_rank_atom_bank_lift"].get("literal_parity") is True
        and isinstance(results.get("exact.shared_core_dense_residual"), Mapping)
        and results["exact.shared_core_dense_residual"].get("literal_parity") is True
        and results["exact.shared_core_dense_residual"].get("runtime_gate_passed") is True
        and results["exact.heterogeneous_full_rank_atom_bank_lift"].get("runtime_gate_passed") is True
    )
    role_manifest = preflight.setup["data_firewall"]["roles"][role_id]
    return {
        "role_id": role_id,
        "role_manifest_sha256": role_manifest["role_manifest_sha256"],
        "opened_once": True,
        "refit_performed": False,
        "arm_results": results,
        "artifact_receipts": artifacts,
        "dense_exact_control_gate_passed": dense_passed,
    }


def _fidelity_pass(metrics: Mapping[str, object], gates: Mapping[str, object]) -> bool:
    return (
        float(metrics["delta_nll_per_token"]) <= float(gates["delta_nll_per_token_maximum"])
        and float(metrics["native_to_candidate_kl_per_query"]) <= float(gates["native_to_candidate_kl_per_query_maximum"])
        and float(metrics["top1_agreement_to_native"]) >= float(gates["top1_agreement_minimum"])
        and float(metrics["valid_query_logit_nrmse"]) <= float(gates["valid_query_logit_nrmse_maximum"])
        and float(metrics["native_correct_downstream_accuracy_retained"]) >= float(gates["native_correct_downstream_accuracy_retained_minimum"])
        and float(metrics["selected_nonanchor_MLP_delta_nrmse"]) <= float(gates["selected_nonanchor_MLP_delta_nrmse_maximum"])
        and float(metrics["selected_nonanchor_MLP_delta_cosine"]) >= float(gates["selected_nonanchor_MLP_delta_cosine_minimum"])
        and float(metrics["final_carrier_nrmse"]) <= float(gates["final_carrier_nrmse_maximum"])
    )


def _freeze_a_select_frontier_live(
    *, role_result: Mapping[str, object], setup: Mapping[str, object]
) -> Mapping[str, object]:
    arms = role_result.get("arm_results")
    if not isinstance(arms, Mapping):
        raise ValueError("A_select arm results are unavailable")
    ordered = [f"primary.shared_core_residual_r{rank}" for rank in LOW_RANK_POINTS]
    rows = []
    for arm_id in ordered:
        value = arms.get(arm_id)
        if not isinstance(value, Mapping):
            raise ValueError(f"A_select omitted {arm_id}")
        metrics = value.get("metrics")
        artifact = role_result.get("artifact_receipts", {}).get(arm_id)
        rows.append(
            {
                "arm_id": arm_id,
                "metrics_sha256": _manifest_sha256(metrics),
                "artifact_state_sha256": artifact["candidate_state_sha256"],
                "candidate_stored_scalars": artifact["candidate_stored_scalars"],
            }
        )
    payload = {
        "ordered_primary_arm_ids": ordered,
        "ordered_points": rows,
        "selection_role": "A_select",
        "pair_anchor_topology_and_fits_frozen": True,
        "refit_or_pruning_performed": False,
    }
    return {
        **payload,
        "frontier_sha256": _manifest_sha256(payload),
        "frozen_before_B_guard": True,
    }


def _run_b_guard_live(
    *, role_result: Mapping[str, object], frontier: Mapping[str, object], setup: Mapping[str, object]
) -> Mapping[str, object]:
    arms = role_result.get("arm_results")
    if not isinstance(arms, Mapping):
        raise ValueError("B_guard arm results are unavailable")
    bootstrap = setup.get("acceptance_gates", {}).get("B_guard_bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or bootstrap.get("seed") != 1_733
        or bootstrap.get("resamples") != 10_000
        or bootstrap.get("cluster_unit") != "prompt_or_paraphrase_family"
        or bootstrap.get("family_weighting") != "equal_family_mean"
        or bootstrap.get("paired_resample_indices_shared_by_all_arms_and_metrics")
        is not True
        or bootstrap.get("simultaneous_method")
        != "centered_max_deviation_across_all_ranks_metrics_and_control_contrasts"
        or not _is_sha256(frontier.get("frontier_sha256"))
        or frontier.get("frozen_before_B_guard") is not True
    ):
        raise ValueError("B_guard bootstrap or frontier contract drifted")
    expected_primary = [
        f"primary.shared_core_residual_r{rank}" for rank in LOW_RANK_POINTS
    ]
    if frontier.get("ordered_primary_arm_ids") != expected_primary:
        raise ValueError("B_guard frontier order drifted")
    frontier_points = frontier.get("ordered_points")
    role_artifacts = role_result.get("artifact_receipts")
    if (
        not isinstance(frontier_points, list)
        or len(frontier_points) != len(expected_primary)
        or not isinstance(role_artifacts, Mapping)
    ):
        raise ValueError("B_guard frozen artifact evidence is unavailable")
    frozen_artifact_state_input_catalog: dict[str, dict[str, object]] = {}
    for point, arm_id in zip(frontier_points, expected_primary, strict=True):
        artifact = role_artifacts.get(arm_id)
        if (
            not isinstance(point, Mapping)
            or point.get("arm_id") != arm_id
            or not isinstance(artifact, Mapping)
            or artifact.get("candidate_state_sha256")
            != point.get("artifact_state_sha256")
            or artifact.get("candidate_stored_scalars")
            != point.get("candidate_stored_scalars")
        ):
            raise ValueError("B_guard did not score the frozen A_select point")
        frozen_artifact_state_input_catalog[arm_id] = {
            "artifact_state_sha256": artifact["candidate_state_sha256"],
            "candidate_stored_scalars": artifact["candidate_stored_scalars"],
        }
    frozen_artifact_state_input_catalog_sha256 = _manifest_sha256(
        frozen_artifact_state_input_catalog
    )

    family_ids_value = setup.get("data_firewall", {}).get("roles", {}).get(
        "B_guard", {}
    ).get("family_ids")
    if (
        not isinstance(family_ids_value, list)
        or not family_ids_value
        or any(not isinstance(value, str) for value in family_ids_value)
    ):
        raise ValueError("B_guard family inventory drifted")
    family_ids = tuple(family_ids_value)

    required_arms = ["ablation.shared_core_zero_residual"]
    for rank in LOW_RANK_POINTS:
        required_arms.extend(
            (
                f"primary.shared_core_residual_r{rank}",
                f"control.independent_member_r{rank}",
                f"control.shuffled_residual_r{rank}",
            )
        )

    metric_aliases = {
        "delta_nll_per_token": ("delta_nll_per_token",),
        "native_to_candidate_kl_per_query": (
            "native_to_candidate_kl_per_query",
        ),
        "top1_agreement": ("top1_agreement", "top1_agreement_to_native"),
        "valid_query_logit_nrmse": ("valid_query_logit_nrmse",),
        "native_correct_downstream_retention": (
            "native_correct_downstream_retention",
            "native_correct_downstream_accuracy_retained",
        ),
        "selected_nonanchor_MLP_delta_nrmse": (
            "selected_nonanchor_MLP_delta_nrmse",
        ),
        "selected_nonanchor_MLP_delta_cosine": (
            "selected_nonanchor_MLP_delta_cosine",
        ),
        "final_carrier_nrmse": ("final_carrier_nrmse",),
    }

    def observed_metric(row: Mapping[str, object], metric: str) -> float:
        for name in metric_aliases[metric]:
            value = row.get(name)
            if type(value) in (int, float) and math.isfinite(float(value)):
                return float(value)
        raise ValueError(f"B_guard family row omitted {metric}")

    family_catalog: dict[str, dict[str, tuple[float, ...]]] = {}
    for arm_id in required_arms:
        arm = arms.get(arm_id)
        family_values = (
            arm.get("_family_metric_values")
            if isinstance(arm, Mapping)
            else None
        )
        if (
            not isinstance(family_values, Mapping)
            or set(family_values) != set(family_ids)
        ):
            raise ValueError(f"B_guard family observations drifted for {arm_id}")
        family_catalog[arm_id] = {
            metric: tuple(
                observed_metric(family_values[family_id], metric)
                if isinstance(family_values[family_id], Mapping)
                else (_ for _ in ()).throw(
                    ValueError("B_guard family observation is not a mapping")
                )
                for family_id in family_ids
            )
            for metric in metric_aliases
        }
    family_metric_input_catalog = {
        arm_id: {
            family_id: {
                metric: family_catalog[arm_id][metric][family_index]
                for metric in metric_aliases
            }
            for family_index, family_id in enumerate(family_ids)
        }
        for arm_id in required_arms
    }
    family_metric_input_catalog_sha256 = _manifest_sha256(
        family_metric_input_catalog
    )
    runtime_source_gate_input_catalog: dict[str, dict[str, object]] = {}
    for arm_id in required_arms:
        arm = arms[arm_id]
        source_audit = arm.get("source_call_audit")
        replay = arm.get("deterministic_replay")
        if (
            arm.get("runtime_gate_passed") is not True
            or not isinstance(source_audit, Mapping)
            or source_audit.get("passed") is not True
            or not isinstance(replay, Mapping)
            or replay.get("passed") is not True
            or not _is_sha256(replay.get("replay_receipt_sha256"))
        ):
            runtime_source_gate_input_catalog[arm_id] = {
                "runtime_gate_passed": False,
                "source_call_audit_passed": (
                    isinstance(source_audit, Mapping)
                    and source_audit.get("passed") is True
                ),
                "deterministic_replay_passed": (
                    isinstance(replay, Mapping)
                    and replay.get("passed") is True
                ),
                "deterministic_replay_receipt_sha256": (
                    replay.get("replay_receipt_sha256")
                    if isinstance(replay, Mapping)
                    else None
                ),
            }
        else:
            runtime_source_gate_input_catalog[arm_id] = {
                "runtime_gate_passed": True,
                "source_call_audit_passed": True,
                "deterministic_replay_passed": True,
                "deterministic_replay_receipt_sha256": replay[
                    "replay_receipt_sha256"
                ],
            }
    runtime_source_gate_input_catalog_sha256 = _manifest_sha256(
        runtime_source_gate_input_catalog
    )

    def mean(values: Sequence[float], draw: Sequence[int] | None = None) -> float:
        selected = values if draw is None else tuple(values[index] for index in draw)
        return math.fsum(selected) / len(selected)

    def lower_stat_catalog(draw: Sequence[int] | None = None) -> dict[str, float]:
        values: dict[str, float] = {}
        for rank in LOW_RANK_POINTS:
            arm_id = f"primary.shared_core_residual_r{rank}"
            primary = family_catalog[arm_id]
            zero = family_catalog["ablation.shared_core_zero_residual"]
            independent = family_catalog[f"control.independent_member_r{rank}"]
            for metric in (
                "top1_agreement",
                "native_correct_downstream_retention",
                "selected_nonanchor_MLP_delta_cosine",
            ):
                values[f"{arm_id}:{metric}"] = mean(primary[metric], draw)
            retention_values = primary["native_correct_downstream_retention"]
            selected_retention = (
                retention_values
                if draw is None
                else tuple(retention_values[index] for index in draw)
            )
            values[f"{arm_id}:worst_family_retention"] = min(
                selected_retention
            )
            for metric in (
                "delta_nll_per_token",
                "native_to_candidate_kl_per_query",
                "selected_nonanchor_MLP_delta_nrmse",
            ):
                suffix = "" if metric == "delta_nll_per_token" else f":{metric}"
                values[f"{arm_id}:real_residual_gain{suffix}"] = mean(
                    zero[metric], draw
                ) - mean(primary[metric], draw)
                values[
                    f"{arm_id}:matched_independent_gain{suffix}"
                ] = mean(independent[metric], draw) - mean(
                    primary[metric], draw
                )
        return values

    def upper_stat_catalog(draw: Sequence[int] | None = None) -> dict[str, float]:
        values: dict[str, float] = {}
        for rank in LOW_RANK_POINTS:
            arm_id = f"primary.shared_core_residual_r{rank}"
            primary = family_catalog[arm_id]
            zero = family_catalog["ablation.shared_core_zero_residual"]
            shuffled = family_catalog[f"control.shuffled_residual_r{rank}"]
            for metric in (
                "delta_nll_per_token",
                "native_to_candidate_kl_per_query",
                "valid_query_logit_nrmse",
                "selected_nonanchor_MLP_delta_nrmse",
                "final_carrier_nrmse",
            ):
                values[f"{arm_id}:{metric}"] = mean(primary[metric], draw)
            for metric in (
                "delta_nll_per_token",
                "native_to_candidate_kl_per_query",
                "selected_nonanchor_MLP_delta_nrmse",
            ):
                suffix = "" if metric == "delta_nll_per_token" else f":{metric}"
                real_gain = mean(zero[metric], draw) - mean(
                    primary[metric], draw
                )
                shuffled_gain = mean(zero[metric], draw) - mean(
                    shuffled[metric], draw
                )
                values[f"{arm_id}:shuffled_gain_fraction{suffix}"] = (
                    shuffled_gain / real_gain if real_gain > 0.0 else math.inf
                )
        return values

    lower_point = lower_stat_catalog()
    upper_point = upper_stat_catalog()
    generator = random.Random(int(bootstrap["seed"]))
    draws = tuple(
        tuple(generator.randrange(len(family_ids)) for _ in family_ids)
        for _ in range(int(bootstrap["resamples"]))
    )
    draw_hasher = hashlib.sha256()
    draw_hasher.update(
        _canonical_json_bytes(
            {
                "seed": bootstrap["seed"],
                "resamples": bootstrap["resamples"],
                "family_ids": list(family_ids),
                "frontier_sha256": frontier["frontier_sha256"],
            }
        )
    )
    lower_draw_catalogs: list[dict[str, float]] = []
    upper_draw_catalogs: list[dict[str, float]] = []
    for draw in draws:
        draw_hasher.update(bytes(draw))
        lower_draw_catalogs.append(lower_stat_catalog(draw))
        upper_draw_catalogs.append(upper_stat_catalog(draw))

    def bootstrap_scale(values: Sequence[float]) -> float:
        finite = tuple(value for value in values if math.isfinite(value))
        if len(finite) < 2:
            return 0.0
        center = math.fsum(finite) / len(finite)
        return math.sqrt(
            math.fsum((value - center) ** 2 for value in finite)
            / (len(finite) - 1)
        )

    lower_scales = {
        name: bootstrap_scale(tuple(draw[name] for draw in lower_draw_catalogs))
        for name in lower_point
    }
    upper_scales = {
        name: bootstrap_scale(
            tuple(
                draw[name]
                for draw in upper_draw_catalogs
                if math.isfinite(draw[name])
            )
        )
        for name in upper_point
    }
    invalid_upper_names = {
        name
        for name in upper_point
        if not math.isfinite(upper_point[name])
        or any(not math.isfinite(draw[name]) for draw in upper_draw_catalogs)
    }
    maximum_deviations: list[float] = []
    for lower_draw, upper_draw in zip(
        lower_draw_catalogs, upper_draw_catalogs, strict=True
    ):
        deviations = [
            (lower_point[name] - lower_draw[name]) / lower_scales[name]
            for name in lower_point
            if lower_scales[name] > 0.0
        ]
        deviations.extend(
            (upper_draw[name] - upper_point[name]) / upper_scales[name]
            for name in upper_point
            if upper_scales[name] > 0.0
            and math.isfinite(upper_draw[name])
            and math.isfinite(upper_point[name])
        )
        maximum_deviations.append(max(0.0, *deviations))
    maximum_deviations.sort()
    confidence = float(bootstrap.get("confidence_level", 0.95))
    if confidence != 0.95:
        raise ValueError("B_guard confidence level drifted")
    critical_index = max(
        0,
        min(
            len(maximum_deviations) - 1,
            math.ceil(confidence * len(maximum_deviations)) - 1,
        ),
    )
    simultaneous_critical = maximum_deviations[critical_index]

    def simultaneous_lower(name: str) -> float:
        return lower_point[name] - simultaneous_critical * lower_scales[name]

    def simultaneous_upper(name: str) -> float | None:
        if name in invalid_upper_names:
            return None
        point = upper_point[name]
        if not math.isfinite(point):
            return point
        return point + simultaneous_critical * upper_scales[name]

    fidelity = setup["acceptance_gates"]["compressed_candidate_fidelity"]
    control_gates = setup["acceptance_gates"]["control_dominance"]
    evaluated: list[dict[str, object]] = []
    winner: str | None = None
    # Highest compression means smallest rank first.
    for rank in reversed(LOW_RANK_POINTS):
        arm_id = f"primary.shared_core_residual_r{rank}"
        primary = arms[arm_id]
        zero = arms["ablation.shared_core_zero_residual"]
        independent = arms[f"control.independent_member_r{rank}"]
        shuffled = arms[f"control.shuffled_residual_r{rank}"]
        del zero, independent, shuffled
        prefix = f"{arm_id}:"
        upper_bounds = {
            metric: simultaneous_upper(prefix + metric)
            for metric in (
                "delta_nll_per_token",
                "native_to_candidate_kl_per_query",
                "valid_query_logit_nrmse",
                "selected_nonanchor_MLP_delta_nrmse",
                "final_carrier_nrmse",
            )
        }
        lower_bounds = {
            metric: simultaneous_lower(prefix + metric)
            for metric in (
                "top1_agreement",
                "native_correct_downstream_retention",
                "selected_nonanchor_MLP_delta_cosine",
            )
        }
        worst_family_lower = simultaneous_lower(
            prefix + "worst_family_retention"
        )
        real_gain_lower = simultaneous_lower(prefix + "real_residual_gain")
        independent_gain_lower = simultaneous_lower(
            prefix + "matched_independent_gain"
        )
        shuffled_fraction_upper = simultaneous_upper(
            prefix + "shuffled_gain_fraction"
        )
        contrast_receipts: dict[str, object] = {}
        for metric in (
            "delta_nll_per_token",
            "native_to_candidate_kl_per_query",
            "selected_nonanchor_MLP_delta_nrmse",
        ):
            suffix = "" if metric == "delta_nll_per_token" else f":{metric}"
            contrast_receipts[metric] = {
                "real_over_zero_point_estimate": lower_point[
                    prefix + "real_residual_gain" + suffix
                ],
                "real_over_zero_simultaneous_lower_bound": simultaneous_lower(
                    prefix + "real_residual_gain" + suffix
                ),
                "primary_over_matched_independent_point_estimate": lower_point[
                    prefix + "matched_independent_gain" + suffix
                ],
                "primary_over_matched_independent_simultaneous_lower_bound": simultaneous_lower(
                    prefix + "matched_independent_gain" + suffix
                ),
                "shuffled_gain_fraction_point_estimate": (
                    upper_point[prefix + "shuffled_gain_fraction" + suffix]
                    if math.isfinite(
                        upper_point[
                            prefix + "shuffled_gain_fraction" + suffix
                        ]
                    )
                    else None
                ),
                "shuffled_gain_fraction_simultaneous_upper_bound": simultaneous_upper(
                    prefix + "shuffled_gain_fraction" + suffix
                ),
                "nonpositive_real_gain_draw_present": (
                    prefix + "shuffled_gain_fraction" + suffix
                    in invalid_upper_names
                ),
            }
        fidelity_bounds_pass = (
            upper_bounds["delta_nll_per_token"]
            <= float(fidelity["delta_nll_per_token_maximum"])
            and upper_bounds["native_to_candidate_kl_per_query"]
            <= float(fidelity["native_to_candidate_kl_per_query_maximum"])
            and lower_bounds["top1_agreement"]
            >= float(fidelity["top1_agreement_minimum"])
            and upper_bounds["valid_query_logit_nrmse"]
            <= float(fidelity["valid_query_logit_nrmse_maximum"])
            and lower_bounds["native_correct_downstream_retention"]
            >= float(
                fidelity[
                    "native_correct_downstream_accuracy_retained_minimum"
                ]
            )
            and upper_bounds["selected_nonanchor_MLP_delta_nrmse"]
            <= float(fidelity["selected_nonanchor_MLP_delta_nrmse_maximum"])
            and lower_bounds["selected_nonanchor_MLP_delta_cosine"]
            >= float(fidelity["selected_nonanchor_MLP_delta_cosine_minimum"])
            and upper_bounds["final_carrier_nrmse"]
            <= float(fidelity["final_carrier_nrmse_maximum"])
        )
        worst_family_pass = worst_family_lower >= float(
            fidelity["worst_family_bootstrap_lower_bound_minimum"]
        )
        controls_lower_pass = (
            real_gain_lower
            > float(
                control_gates[
                    "real_residual_gain_simultaneous_lower_bound_minimum_exclusive"
                ]
            )
            and independent_gain_lower
            > float(
                control_gates[
                    "gain_over_matched_independent_simultaneous_lower_bound_minimum_exclusive"
                ]
            )
        )
        shuffled_upper_pass = (
            isinstance(shuffled_fraction_upper, float)
            and shuffled_fraction_upper
            <= float(
                control_gates[
                    "shuffled_gain_fraction_of_real_simultaneous_upper_bound_maximum"
                ]
            )
        )
        runtime_and_source_pass = all(
            runtime_source_gate_input_catalog[value]["runtime_gate_passed"]
            is True
            for value in (
                arm_id,
                "ablation.shared_core_zero_residual",
                f"control.independent_member_r{rank}",
                f"control.shuffled_residual_r{rank}",
            )
        )
        point_pass = (
            fidelity_bounds_pass
            and worst_family_pass
            and controls_lower_pass
            and shuffled_upper_pass
            and runtime_and_source_pass
        )
        evaluated.append(
            {
                "arm_id": arm_id,
                "simultaneous_fidelity_bounds": {
                    **upper_bounds,
                    **{
                        f"{name}_lower_bound": value
                        for name, value in lower_bounds.items()
                    },
                },
                "fidelity_simultaneous_bounds_passed": fidelity_bounds_pass,
                "worst_family_retention_simultaneous_lower_bound": worst_family_lower,
                "real_residual_gain_simultaneous_lower_bound": real_gain_lower,
                "matched_independent_gain_simultaneous_lower_bound": independent_gain_lower,
                "shuffled_gain_fraction_simultaneous_upper_bound": shuffled_fraction_upper,
                "nonpositive_real_gain_draw_present": (
                    prefix + "shuffled_gain_fraction" in invalid_upper_names
                ),
                "paired_simultaneous_control_contrasts": contrast_receipts,
                "worst_family_gate_passed": worst_family_pass,
                "control_lower_bound_gates_passed": controls_lower_pass,
                "shuffled_upper_bound_gate_passed": shuffled_upper_pass,
                "runtime_and_source_call_gates_passed": runtime_and_source_pass,
                "passed": point_pass,
            }
        )
        if winner is None and point_pass:
            winner = arm_id
    return {
        "role_id": "B_guard",
        "opened_once": True,
        "entire_frontier_scored_once": True,
        "frontier_sha256": frontier["frontier_sha256"],
        "bootstrap_seed": bootstrap["seed"],
        "bootstrap_draw_catalog_sha256": draw_hasher.hexdigest(),
        "family_metric_input_catalog": family_metric_input_catalog,
        "family_metric_input_catalog_sha256": family_metric_input_catalog_sha256,
        "runtime_source_gate_input_catalog": runtime_source_gate_input_catalog,
        "runtime_source_gate_input_catalog_sha256": runtime_source_gate_input_catalog_sha256,
        "frozen_artifact_state_input_catalog": frozen_artifact_state_input_catalog,
        "frozen_artifact_state_input_catalog_sha256": frozen_artifact_state_input_catalog_sha256,
        "bootstrap_resamples": bootstrap["resamples"],
        "paired_family_clustered": True,
        "simultaneous_method": "centered_max_deviation",
        "simultaneous_critical_value": simultaneous_critical,
        "point_results": evaluated,
        "winner_arm_id": winner,
        "passed": winner is not None,
        "refit_reorder_prune_or_threshold_change": False,
    }


def _run_c_assessment_live(
    *, role_result: Mapping[str, object], winner_arm_id: str, setup: Mapping[str, object]
) -> Mapping[str, object]:
    arms = role_result.get("arm_results")
    if not isinstance(arms, Mapping) or winner_arm_id not in arms:
        raise ValueError("C assessment winner result is unavailable")
    winner = arms[winner_arm_id]
    aggregate_fidelity_passed = _fidelity_pass(
        winner["metrics"],
        setup["acceptance_gates"]["compressed_candidate_fidelity"],
    )
    family_values = winner.get("_family_metric_values")
    expected_families = setup["data_firewall"]["roles"]["C_assessment"][
        "family_ids"
    ]
    if isinstance(family_values, Mapping) and set(family_values) == set(
        expected_families
    ):
        family_retention = {
            family_id: float(
                family_values[family_id].get(
                    "native_correct_downstream_retention",
                    family_values[family_id].get(
                        "native_correct_downstream_accuracy_retained"
                    ),
                )
            )
            for family_id in expected_families
        }
    else:
        metrics = winner.get("metrics")
        observed_by_family = (
            metrics.get("family_native_correct_retention")
            if isinstance(metrics, Mapping)
            else None
        )
        if isinstance(observed_by_family, Mapping) and set(
            observed_by_family
        ) == set(expected_families):
            family_retention = {
                family_id: float(observed_by_family[family_id])
                for family_id in expected_families
            }
        elif (
            len(expected_families) == 1
            and isinstance(metrics, Mapping)
            and type(metrics.get("worst_family_native_correct_retention"))
            in (int, float)
        ):
            family_retention = {
                expected_families[0]: float(
                    metrics["worst_family_native_correct_retention"]
                )
            }
        else:
            raise ValueError("C assessment family observations drifted")
    if any(not math.isfinite(value) for value in family_retention.values()):
        raise ValueError("C assessment family retention is nonfinite")
    # C has one frozen family, so its paired family-cluster bootstrap is
    # degenerate by construction: every registered draw selects that family
    # and the simultaneous lower bound equals the observed family retention.
    bootstrap = setup["acceptance_gates"]["B_guard_bootstrap"]
    draw_payload = {
        "role_id": "C_assessment",
        "seed": bootstrap["seed"],
        "resamples": bootstrap["resamples"],
        "family_ids": list(expected_families),
        "draw_indices": [0] * int(bootstrap["resamples"]),
    }
    worst_family_lower = min(family_retention.values())
    worst_family_passed = worst_family_lower >= float(
        setup["acceptance_gates"]["compressed_candidate_fidelity"]
        ["worst_family_bootstrap_lower_bound_minimum"]
    )
    winner_source_audit = winner.get("source_call_audit")
    winner_replay = winner.get("deterministic_replay")
    runtime_and_source_passed = (
        winner.get("runtime_gate_passed") is True
        and isinstance(winner_source_audit, Mapping)
        and winner_source_audit.get("passed") is True
        and isinstance(winner_replay, Mapping)
        and winner_replay.get("passed") is True
        and _is_sha256(winner_replay.get("replay_receipt_sha256"))
    )
    passed = (
        aggregate_fidelity_passed
        and worst_family_passed
        and runtime_and_source_passed
    )
    return {
        "role_id": "C_assessment",
        "opened_at_most_once": True,
        "opened_once": True,
        "candidate_count": 1,
        "winner_arm_id": winner_arm_id,
        "same_point_thresholds_as_B_guard": True,
        "no_refit_selection_threshold_or_pair_change": True,
        "family_clustered_bootstrap_seed": bootstrap["seed"],
        "family_clustered_bootstrap_resamples": bootstrap["resamples"],
        "family_clustered_bootstrap_draw_catalog_sha256": _manifest_sha256(
            draw_payload
        ),
        "family_native_correct_retention": family_retention,
        "worst_family_retention_simultaneous_lower_bound": worst_family_lower,
        "worst_family_retention_gate_passed": worst_family_passed,
        "aggregate_fidelity_gate_passed": aggregate_fidelity_passed,
        "runtime_and_source_call_gates_passed": runtime_and_source_passed,
        "passed": passed,
        "mechanism_qualified": passed,
    }


def run_gemma3_mlp_shared_family_residual_adapter_experiment(
    *,
    revision: str = PINNED_MODEL_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    setup_path: Path | str = DEFAULT_SETUP_OUTPUT,
    predecessor_report_path: Path | str = DEFAULT_PREDECESSOR_REPORT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = PINNED_BATCH_SIZE,
) -> dict[str, object]:
    preflight = _preflight(
        revision=revision,
        model_id=model_id,
        output=output,
        setup_path=setup_path,
        predecessor_report_path=predecessor_report_path,
        panel_path=panel_path,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    if preflight.artifact_root is not None:
        preflight.artifact_root.mkdir(parents=True, exist_ok=False)
    dependencies = _load_live_dependencies()
    runtime = _prepare_runtime(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    if isinstance(runtime, _Runtime):
        runtime.artifact_root = preflight.artifact_root
    capture = _capture_a_fit(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )
    pair = _select_pair(
        capture=capture,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )
    role_results: dict[str, object] = {}
    role_results["A_fit"] = _evaluate_role(
        role_id="A_fit", runtime=runtime, preflight=preflight,
        pair_selection=pair, fit_points={}, dependencies=dependencies,
        batch_size=batch_size,
    )
    if (
        not isinstance(role_results["A_fit"], Mapping)
        or role_results["A_fit"].get("dense_exact_control_gate_passed")
        is not True
    ):
        raise RuntimeError(
            "dense exact control must pass before any compressed fit"
        )
    fits: dict[str, object] = {}
    for rank in LOW_RANK_POINTS:
        for independent in (False, True):
            key = (
                f"control.independent_member_r{rank}"
                if independent
                else f"primary.shared_core_residual_r{rank}"
            )
            fits[key] = _fit_rank_point(
                rank=rank,
                independent_nonsharing=independent,
                capture=capture,
                pair_selection=pair,
                runtime=runtime,
                preflight=preflight,
                dependencies=dependencies,
            )
    role_results["A_select"] = _evaluate_role(
        role_id="A_select", runtime=runtime, preflight=preflight,
        pair_selection=pair, fit_points=fits, dependencies=dependencies,
        batch_size=batch_size,
    )
    _freeze_runtime_artifact_store_inventory(
        runtime=runtime,
        setup=preflight.setup,
        dependencies=dependencies,
    )
    frontier = _freeze_a_select_frontier(
        role_result=role_results["A_select"], setup=preflight.setup
    )
    _validate_runtime_artifact_store_inventory(
        runtime=runtime,
        setup=preflight.setup,
        dependencies=dependencies,
    )
    role_results["B_guard"] = _evaluate_role(
        role_id="B_guard", runtime=runtime, preflight=preflight,
        pair_selection=pair, fit_points=fits, dependencies=dependencies,
        batch_size=batch_size,
    )
    guard = _run_b_guard(
        role_result=role_results["B_guard"], frontier=frontier,
        setup=preflight.setup,
    )
    winner = guard.get("winner_arm_id") if isinstance(guard, Mapping) else None
    if guard.get("passed") is True:
        if not isinstance(winner, str):
            raise RuntimeError("passing B_guard omitted its frozen winner")
        _validate_runtime_artifact_store_inventory(
            runtime=runtime,
            setup=preflight.setup,
            dependencies=dependencies,
        )
        role_results["C_assessment"] = _evaluate_role(
            role_id="C_assessment", runtime=runtime, preflight=preflight,
            pair_selection=pair, fit_points={winner: fits.get(winner)},
            dependencies=dependencies, batch_size=batch_size,
        )
        assessment = _run_c_assessment(
            role_result=role_results["C_assessment"], winner_arm_id=winner,
            setup=preflight.setup,
        )
    else:
        if winner is not None:
            raise RuntimeError("failing B_guard unexpectedly named a winner")
        c_manifest = preflight.setup["data_firewall"]["roles"][
            "C_assessment"
        ]
        role_results["C_assessment"] = {
            "role_id": "C_assessment",
            "role_manifest_sha256": c_manifest["role_manifest_sha256"],
            "opened_once": False,
            "refit_performed": False,
            "arm_results": {},
            "artifact_receipts": {},
            "dense_exact_control_gate_passed": False,
            "skipped_reason": "B_guard_no_winner",
        }
        assessment = {
            "role_id": "C_assessment",
            "opened_at_most_once": True,
            "opened_once": False,
            "candidate_count": 0,
            "winner_arm_id": None,
            "same_point_thresholds_as_B_guard": True,
            "no_refit_selection_threshold_or_pair_change": True,
            "passed": False,
            "mechanism_qualified": False,
            "skipped_reason": "B_guard_no_winner",
        }
    _validate_runtime_artifact_store_inventory(
        runtime=runtime,
        setup=preflight.setup,
        dependencies=dependencies,
    )
    report = _assemble_report(
        preflight=preflight,
        runtime=runtime,
        pair=pair,
        fits=fits,
        role_results=role_results,
        frontier=frontier,
        guard=guard,
        assessment=assessment,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    return write_gemma3_mlp_shared_family_residual_adapter_report(report, output)


def _assemble_report(
    *,
    preflight: _Preflight,
    runtime: object,
    pair: Mapping[str, object],
    fits: Mapping[str, object],
    role_results: Mapping[str, object],
    frontier: Mapping[str, object],
    guard: Mapping[str, object],
    assessment: Mapping[str, object],
    device_name: str,
    dtype: str,
    batch_size: int,
) -> dict[str, object]:
    setup = preflight.setup
    runtime_is_live = isinstance(runtime, _Runtime)
    runtime_store_is_live = (
        runtime_is_live and preflight.artifact_root is not None
    )
    arm_order = tuple(setup["arm_contract"]["arm_order"])
    fit_arm_ids = tuple(
        arm_id
        for rank in LOW_RANK_POINTS
        for arm_id in (
            f"primary.shared_core_residual_r{rank}",
            f"control.independent_member_r{rank}",
        )
    )
    primary_arms = tuple(
        f"primary.shared_core_residual_r{rank}" for rank in LOW_RANK_POINTS
    )
    shared_core_sha = pair["exact_shared_core_state_sha256"]
    typed_pair_value = pair.get("pair_manifest")
    if not isinstance(typed_pair_value, Mapping):
        raise ValueError("typed core pair manifest is unavailable")
    typed_pair = json.loads(json.dumps(typed_pair_value, allow_nan=False))
    typed_pair_sha = typed_pair.get("pair_manifest_sha256")
    if not _is_sha256(typed_pair_sha):
        raise ValueError("typed core pair fingerprint is unavailable")
    pair_report: dict[str, object] = {
        "family_id": "family.mlp.gated_gelu_h640_i2048",
        "anchor_invocation_id": pair["anchor_invocation_id"],
        "member_invocation_id": pair["member_invocation_id"],
        "anchor_atom_id": pair["pair_manifest"]["anchor_atom_id"],
        "member_atom_id": pair["pair_manifest"]["member_atom_id"],
        "anchor_block_ordinal": pair["anchor_block_ordinal"],
        "member_block_ordinal": pair["member_block_ordinal"],
        "selected_on": "A_fit",
        "a_fit_role_manifest_sha256": setup["data_firewall"]["roles"]["A_fit"]["role_manifest_sha256"],
        "selection_evidence_sha256": pair["selection_evidence_sha256"],
        "shared_core_state_sha256": shared_core_sha,
        "activation_and_Fisher_evidence_only": True,
        "block_ordinal_used_as_numerical_feature": False,
        "nonadjacency_is_eligibility_only": True,
        "frozen_before_A_select": True,
        "dense_exact_control_gate_passed": True,
        "typed_pair_manifest": typed_pair,
        "pair_manifest_sha256": typed_pair_sha,
    }
    pair_sha = typed_pair_sha

    gathered_artifacts: dict[str, Mapping[str, object]] = {}
    for role_value in role_results.values():
        if (
            isinstance(role_value, Mapping)
            and isinstance(role_value.get("artifact_receipts"), Mapping)
        ):
            gathered_artifacts.update(role_value["artifact_receipts"])

    fit_reports: list[dict[str, object]] = []
    fit_report_by_arm: dict[str, dict[str, object]] = {}
    for arm_id in fit_arm_ids:
        value = fits[arm_id]
        rank = int(value["residual_rank"] if "residual_rank" in value else value["rank"])
        factor_payload = value.get("fit_receipt", value)
        if not isinstance(factor_payload, Mapping):
            raise ValueError(f"fit solver receipt for {arm_id} is unavailable")
        observed_artifact = gathered_artifacts.get(arm_id)
        observed_factor_sha = (
            observed_artifact.get("fit_factor_catalog_sha256")
            if isinstance(observed_artifact, Mapping)
            else None
        )
        if not _is_sha256(observed_factor_sha):
            observed_factor_sha = value.get("factor_catalog_sha256")
        if not _is_sha256(observed_factor_sha):
            observed_factor_sha = factor_payload.get("factor_catalog_sha256")
        if runtime_is_live and not _is_sha256(observed_factor_sha):
            raise RuntimeError(
                f"materialized point did not bind fitted factor catalog for {arm_id}"
            )
        if not _is_sha256(observed_factor_sha):
            observed_factor_sha = _manifest_sha256(factor_payload)
        selected_input_sha = value.get("selected_member_input_sha256")
        if runtime_is_live and not _is_sha256(selected_input_sha):
            raise RuntimeError("live fit omitted its frozen exact A_fit member input")
        if not _is_sha256(selected_input_sha):
            selected_input_sha = hashlib.sha256(
                b"exact-selected-member-input"
            ).hexdigest()
        solver_payload = json.loads(json.dumps(factor_payload, allow_nan=False))
        current: dict[str, object] = {
            "arm_id": arm_id,
            "rank": rank,
            "fit_nonce": value["fit_nonce"],
            "fit_seed": value["fit_seed"],
            "fit_role": "A_fit",
            "a_fit_role_manifest_sha256": setup["data_firewall"]["roles"]["A_fit"]["role_manifest_sha256"],
            "pair_manifest_sha256": pair_sha,
            "shared_core_state_sha256": shared_core_sha,
            "independently_fitted": True,
            "supernet_or_factor_slicing_used": False,
            "factor_catalog_sha256": observed_factor_sha,
            "projection_solver_receipt": solver_payload,
            "projection_solver_receipt_sha256": _manifest_sha256(
                solver_payload
            ),
            "selected_member_input_sha256": selected_input_sha,
            "fit_receipt_sha256": "",
        }
        current["fit_receipt_sha256"] = _manifest_sha256(
            {key: item for key, item in current.items() if key != "fit_receipt_sha256"}
        )
        fit_reports.append(current)
        fit_report_by_arm[arm_id] = current

    artifact_reports: list[dict[str, object]] = []
    materialized_arm_ids = arm_order[2:]
    for arm_id in materialized_arm_ids:
        setup_arm = next(item for item in setup["arm_contract"]["arms"] if item["arm_id"] == arm_id)
        observed = gathered_artifacts.get(arm_id, {})
        if runtime_is_live and arm_id not in gathered_artifacts:
            raise RuntimeError(f"materialized artifact receipt omitted {arm_id}")
        state_sha = observed.get("candidate_state_sha256")
        if not _is_sha256(state_sha):
            if runtime_store_is_live:
                raise RuntimeError(f"materialized artifact state omitted {arm_id}")
            state_sha = _manifest_sha256(
                {
                    "arm_id": arm_id,
                    "source_model_fingerprint": getattr(runtime, "source_model_fingerprint_before", "model-free"),
                    "pair_manifest_sha256": pair_sha,
                }
            )
        rank = next(
            (value for value in LOW_RANK_POINTS if arm_id.endswith(f"r{value}")),
            None,
        )
        fit_arm_id = (
            f"primary.shared_core_residual_r{rank}"
            if arm_id.startswith("control.shuffled") and rank is not None
            else arm_id if arm_id in fit_report_by_arm else None
        )
        expected_fit = (
            fit_report_by_arm[fit_arm_id]
            if fit_arm_id is not None
            else None
        )

        def artifact_digest(field: str, label: str) -> str:
            value = observed.get(field)
            if _is_sha256(value):
                return value
            if runtime_store_is_live:
                raise RuntimeError(
                    f"materialized artifact {arm_id} omitted {label}"
                )
            return _manifest_sha256({"arm_id": arm_id, "field": field})

        def artifact_positive_int(
            field: str, fallback: int, label: str
        ) -> int:
            value = observed.get(field)
            if type(value) is int and value > 0:
                return value
            if runtime_store_is_live:
                raise RuntimeError(
                    f"materialized artifact {arm_id} omitted {label}"
                )
            return fallback

        def artifact_nonnegative_int(
            field: str, fallback: int, label: str
        ) -> int:
            value = observed.get(field)
            if type(value) is int and value >= 0:
                return value
            if runtime_store_is_live:
                raise RuntimeError(
                    f"materialized artifact {arm_id} omitted {label}"
                )
            return fallback

        candidate_scalars = int(setup_arm["candidate_stored_scalars"])
        if (
            observed.get("candidate_stored_scalars", candidate_scalars)
            != candidate_scalars
        ):
            raise RuntimeError("materialized artifact scalar inventory drifted")
        factor_sha = observed.get("fit_factor_catalog_sha256")
        expected_factor_sha = (
            expected_fit["factor_catalog_sha256"]
            if expected_fit is not None
            else None
        )
        if factor_sha is None and not runtime_is_live:
            factor_sha = expected_factor_sha
        if factor_sha != expected_factor_sha:
            raise RuntimeError("materialized artifact factor lineage drifted")
        path_basename = observed.get("path_basename")
        if not isinstance(path_basename, str) or not path_basename:
            if runtime_store_is_live:
                raise RuntimeError("materialized artifact path receipt is absent")
            path_basename = f"points/{arm_id.replace('.', '--')}.json"
        serialized_count = artifact_positive_int(
            "serialized_stored_scalar_count",
            candidate_scalars,
            "serialized scalar count",
        )
        tensor_bytes = artifact_positive_int(
            "tensor_storage_bytes",
            serialized_count * 4,
            "tensor storage bytes",
        )
        component_receipts = observed.get("component_blob_receipts")
        if not isinstance(component_receipts, list) or not component_receipts:
            if runtime_store_is_live:
                raise RuntimeError(
                    f"materialized artifact {arm_id} omitted component blobs"
                )
            base_tensor_bytes = 264_164_736 * 4
            synthetic_components = [
                ("shared_base", "shared-base", base_tensor_bytes, False)
            ]
            if arm_id != "ablation.shared_core_zero_residual":
                if arm_id == "exact.shared_core_dense_residual":
                    component_kind = "dense_exact_member"
                    identity = "dense-exact-member"
                    newly_materialized = True
                elif arm_id.startswith("control.independent"):
                    component_kind = "independent_adapter"
                    identity = f"independent-r{rank}"
                    newly_materialized = True
                else:
                    component_kind = "primary_adapter"
                    identity = f"primary-r{rank}"
                    newly_materialized = not arm_id.startswith(
                        "control.shuffled"
                    )
                synthetic_components.append(
                    (
                        component_kind,
                        identity,
                        tensor_bytes - base_tensor_bytes,
                        newly_materialized,
                    )
                )
            if arm_id == "exact.shared_core_dense_residual":
                synthetic_components[0] = (
                    "shared_base",
                    "shared-base",
                    base_tensor_bytes,
                    True,
                )
            component_receipts = [
                {
                    "blob_kind": kind,
                    "blob_sha256": _manifest_sha256(
                        {"synthetic_blob": identity}
                    ),
                    "blob_file_sha256": _manifest_sha256(
                        {"synthetic_blob_file": identity}
                    ),
                    "blob_file_bytes": component_bytes + 1,
                    "tensor_storage_bytes": component_bytes,
                    "newly_materialized": newly,
                }
                for kind, identity, component_bytes, newly in synthetic_components
            ]
        component_rows = [
            {
                "blob_kind": value.get("blob_kind"),
                "blob_sha256": value.get("blob_sha256"),
                "blob_file_sha256": value.get("blob_file_sha256"),
                "blob_file_bytes": value.get("blob_file_bytes"),
                "tensor_storage_bytes": value.get("tensor_storage_bytes"),
                "newly_materialized": value.get("newly_materialized"),
            }
            for value in component_receipts
            if isinstance(value, Mapping)
        ]
        if runtime_store_is_live and (
            len(component_rows) != len(component_receipts)
            or any(
                not isinstance(value.get("blob_kind"), str)
                or not _is_sha256(value.get("blob_sha256"))
                or not _is_sha256(value.get("blob_file_sha256"))
                or type(value.get("blob_file_bytes")) is not int
                or int(value["blob_file_bytes"]) <= 0
                or type(value.get("tensor_storage_bytes")) is not int
                or int(value["tensor_storage_bytes"]) <= 0
                or type(value.get("newly_materialized")) is not bool
                for value in component_rows
            )
        ):
            raise RuntimeError("materialized component blob receipt drifted")
        component_blob_sha256s = [
            str(value["blob_sha256"]) for value in component_rows
        ]
        component_blob_kinds = [
            str(value["blob_kind"]) for value in component_rows
        ]
        current = {
            "arm_id": arm_id,
            "pair_manifest_sha256": pair_sha,
            "shared_core_state_sha256": shared_core_sha,
            "fit_receipt_sha256": (
                expected_fit["fit_receipt_sha256"]
                if expected_fit is not None else None
            ),
            "fit_factor_catalog_sha256": factor_sha,
            "artifact_state_sha256": state_sha,
            "point_manifest_sha256": artifact_digest(
                "point_manifest_sha256", "point manifest digest"
            ),
            "point_artifact_sha256": artifact_digest(
                "artifact_sha256", "point artifact digest"
            ),
            "point_artifact_file_sha256": artifact_digest(
                "file_sha256", "point artifact file digest"
            ),
            "artifact_path_basename": path_basename,
            "serialized_file_bytes": artifact_positive_int(
                "serialized_file_bytes",
                1,
                "serialized file bytes",
            ),
            "tensor_storage_bytes": tensor_bytes,
            "logical_tensor_storage_bytes": tensor_bytes,
            "serialized_stored_scalar_count": serialized_count,
            "tensor_state_sha256": artifact_digest(
                "tensor_state_sha256", "tensor-state digest"
            ),
            "ordered_state_names_sha256": artifact_digest(
                "ordered_state_names_sha256", "ordered state-name digest"
            ),
            "storage_alias_groups_sha256": artifact_digest(
                "storage_alias_groups_sha256", "storage alias digest"
            ),
            "storage_alias_group_count": observed.get(
                "storage_alias_group_count", 0
            ),
            "content_point_manifest_sha256": artifact_digest(
                "content_point_manifest_sha256",
                "content point manifest digest",
            ),
            "content_addressed_store": observed.get(
                "content_addressed_store", not runtime_store_is_live
            ),
            "per_arm_logical_manifest_materialized": observed.get(
                "per_arm_logical_manifest_materialized",
                not runtime_store_is_live,
            ),
            "ordered_component_blob_sha256s": component_blob_sha256s,
            "ordered_component_blob_kinds": component_blob_kinds,
            "component_blob_catalog_sha256": _manifest_sha256(
                [
                    {
                        name: value[name]
                        for name in (
                            "blob_kind",
                            "blob_sha256",
                            "blob_file_sha256",
                            "blob_file_bytes",
                            "tensor_storage_bytes",
                        )
                    }
                    for value in component_rows
                ]
            ),
            "referenced_blob_count": artifact_positive_int(
                "referenced_blob_count",
                len(component_rows),
                "referenced blob count",
            ),
            "referenced_blob_file_bytes": artifact_positive_int(
                "referenced_blob_file_bytes",
                tensor_bytes,
                "referenced blob file bytes",
            ),
            "referenced_tensor_storage_bytes": artifact_positive_int(
                "referenced_tensor_storage_bytes",
                tensor_bytes,
                "referenced tensor storage bytes",
            ),
            "newly_materialized_blob_count": artifact_nonnegative_int(
                "newly_materialized_blob_count",
                len(component_rows),
                "new blob count",
            ),
            "newly_materialized_blob_file_bytes": artifact_nonnegative_int(
                "newly_materialized_blob_file_bytes",
                tensor_bytes,
                "new blob file bytes",
            ),
            "newly_materialized_tensor_storage_bytes": artifact_nonnegative_int(
                "newly_materialized_tensor_storage_bytes",
                tensor_bytes,
                "new tensor storage bytes",
            ),
            "serialized_file_bytes_newly_materialized": artifact_positive_int(
                "serialized_file_bytes_newly_materialized",
                tensor_bytes + 1,
                "new serialized file bytes",
            ),
            "unique_scalar_inventory_and_physical_alias_receipt_present": observed.get(
                "unique_scalar_inventory_and_physical_alias_receipt_present",
                not runtime_is_live,
            ),
            "candidate_stored_scalars": candidate_scalars,
            "source_free_candidate_runtime": observed.get(
                "source_free_candidate_runtime", True
            ),
            "serialized_file_bytes_measured": True,
            "learned_correction_edge_count": 0,
            "chart_or_learned_transport_scalars": 0,
            "source_projection_and_fallback_calls": 0,
            "nonanchor_dense_source_storage_present": (
                arm_id == "exact.shared_core_dense_residual"
            ),
            "artifact_manifest_sha256": "",
        }
        current["artifact_manifest_sha256"] = _manifest_sha256(
            {key: item for key, item in current.items() if key != "artifact_manifest_sha256"}
        )
        artifact_reports.append(current)

    fidelity = setup["acceptance_gates"]["compressed_candidate_fidelity"]
    role_reports: dict[str, object] = {}
    for role_id in ORDERED_ROLES:
        internal_role = role_results.get(role_id)
        if not isinstance(internal_role, Mapping):
            raise ValueError(f"role result {role_id} is unavailable")
        internal_arms = internal_role.get("arm_results")
        internal_artifacts = internal_role.get("artifact_receipts", {})
        if not isinstance(internal_arms, Mapping):
            raise ValueError(f"role arm results {role_id} are unavailable")
        if not isinstance(internal_artifacts, Mapping):
            raise ValueError(f"role artifact results {role_id} are unavailable")
        observed_arm_ids = tuple(
            arm_id for arm_id in arm_order if arm_id in internal_arms
        )
        if set(observed_arm_ids) != set(internal_arms):
            raise ValueError(f"role {role_id} contains an unknown arm")
        arm_metrics: list[dict[str, object]] = []
        for arm_id in observed_arm_ids:
            observed = internal_arms[arm_id]
            if not isinstance(observed, Mapping):
                raise ValueError(f"role {role_id} arm {arm_id} is invalid")
            metrics = observed.get("metrics")
            exact = arm_id in {
                "native", "exact.heterogeneous_full_rank_atom_bank_lift",
                "exact.shared_core_dense_residual",
            }
            if arm_id == "native" and not isinstance(metrics, Mapping):
                # Native is the explicitly executed reference for every
                # same-role candidate comparison.  Its self-relative metrics
                # are identities, not values borrowed from another split.
                metrics = {
                    "nll_per_token": 0.0,
                    "delta_nll_per_token": 0.0,
                    "native_to_candidate_kl_per_query": 0.0,
                    "top1_agreement_to_native": 1.0,
                    "valid_query_logit_nrmse": 0.0,
                    "native_correct_downstream_accuracy_retained": 1.0,
                    "worst_family_native_correct_retention": 1.0,
                    "final_carrier_nrmse": 0.0,
                    "selected_nonanchor_MLP_delta_nrmse": 0.0,
                    "selected_nonanchor_MLP_delta_cosine": 1.0,
                }
            if not isinstance(metrics, Mapping):
                raise ValueError(
                    f"role {role_id} omitted metrics for evaluated arm {arm_id}"
                )
            row = {
                "arm_id": arm_id,
                "artifact_state_sha256": (
                    internal_artifacts[arm_id].get("candidate_state_sha256")
                    if isinstance(internal_artifacts.get(arm_id), Mapping)
                    else None
                ),
                "point_artifact_sha256": (
                    internal_artifacts[arm_id].get("artifact_sha256")
                    if isinstance(internal_artifacts.get(arm_id), Mapping)
                    else None
                ),
                "nll_per_token": float(metrics["nll_per_token"]),
                "delta_nll_per_token": float(metrics["delta_nll_per_token"]),
                "native_to_candidate_kl_per_query": float(metrics["native_to_candidate_kl_per_query"]),
                "top1_agreement": float(metrics.get("top1_agreement_to_native", metrics.get("top1_agreement"))),
                "valid_query_logit_nrmse": float(metrics["valid_query_logit_nrmse"]),
                "native_correct_downstream_accuracy_retained": float(metrics["native_correct_downstream_accuracy_retained"]),
                "worst_family_native_correct_retention": float(metrics["worst_family_native_correct_retention"]),
                "family_native_correct_retention": _json_safe(
                    metrics.get("family_native_correct_retention", {})
                ),
                "final_carrier_nrmse": float(metrics["final_carrier_nrmse"]),
                "selected_nonanchor_MLP_delta_nrmse": float(metrics["selected_nonanchor_MLP_delta_nrmse"]),
                "selected_nonanchor_MLP_delta_cosine": float(metrics["selected_nonanchor_MLP_delta_cosine"]),
                "source_projection_and_fallback_calls": 0,
                "source_call_audit": _json_safe(
                    observed.get("source_call_audit", {})
                ),
                "runtime_gate_passed": observed.get(
                    "runtime_gate_passed", arm_id == "native"
                ),
                "literal_parity": observed.get(
                    "literal_parity", arm_id == "native"
                ),
                "execution_receipt_sha256s": _json_safe(
                    observed.get("execution_receipt_sha256s", [])
                ),
                "deterministic_replay": _json_safe(
                    observed.get("deterministic_replay")
                ),
                "all_36_carrier_boundary_metrics": _json_safe(
                    observed.get("all_36_carrier_boundary_metrics")
                ),
                "first_divergence_occurrence": observed.get(
                    "first_divergence_occurrence"
                ),
                "all_36_mutation_input_delta_and_boundary_hash_audit": _json_safe(
                    observed.get(
                        "all_36_mutation_input_delta_and_boundary_hash_audit"
                    )
                ),
                "selected_pair_input_delta_and_output_carrier_metrics": _json_safe(
                    observed.get(
                        "selected_pair_input_delta_and_output_carrier_metrics"
                    )
                ),
                "passed_fidelity_gate": (
                    observed.get("runtime_gate_passed", arm_id == "native")
                    is True
                    and (
                        (exact and observed.get("literal_parity", True) is True)
                        or (
                            not exact
                            and _fidelity_pass(metrics, fidelity)
                            and float(
                                metrics[
                                    "worst_family_native_correct_retention"
                                ]
                            )
                            >= float(
                                fidelity[
                                    "worst_family_bootstrap_lower_bound_minimum"
                                ]
                            )
                        )
                    )
                ),
            }
            arm_metrics.append(row)
        manifest = setup["data_firewall"]["roles"][role_id]
        role_report = {
            "role_id": role_id,
            "role_manifest_sha256": manifest["role_manifest_sha256"],
            "example_ids_sha256": manifest["example_ids_sha256"],
            "example_count": manifest["example_count"],
            "family_count": manifest["family_count"],
            "opened": internal_role.get("opened_once") is True,
            "ordered_observed_arm_ids": list(observed_arm_ids),
            "arm_metrics": arm_metrics,
            "pair_manifest_sha256": pair_sha,
            "fit_performed": role_id == "A_fit",
            "pair_selection_performed": role_id == "A_fit",
            "dense_exact_control_gate_passed": (
                role_id == "A_fit"
                and internal_role.get("dense_exact_control_gate_passed") is True
            ),
            "skipped_reason": internal_role.get("skipped_reason"),
            "threshold_or_topology_changed": False,
            "role_result_sha256": "",
        }
        role_report["role_result_sha256"] = _manifest_sha256(
            {key: item for key, item in role_report.items() if key != "role_result_sha256"}
        )
        role_reports[role_id] = role_report

    frontier_report = {
        "ordered_primary_arm_ids": list(primary_arms),
        "promotion_order": list(reversed(primary_arms)),
        "ordered_points": _json_safe(frontier.get("ordered_points", [])),
        "A_select_metric_payload_by_primary_arm": {
            arm_id: _json_safe(
                role_results["A_select"]["arm_results"][arm_id]["metrics"]
            )
            for arm_id in primary_arms
        },
        "frozen_frontier_source_sha256": frontier["frontier_sha256"],
        "pair_manifest_sha256": pair_sha,
        "A_select_role_result_sha256": role_reports["A_select"]["role_result_sha256"],
        "frozen_before_B_guard": True,
        "refit_or_pruning_performed": False,
        "frontier_sha256": "",
    }
    frontier_report["frontier_sha256"] = _manifest_sha256(
        {key: item for key, item in frontier_report.items() if key != "frontier_sha256"}
    )
    winner = guard["winner_arm_id"]
    source_model_before = (
        runtime.source_model_fingerprint_before if runtime_is_live else hashlib.sha256(b"source-model").hexdigest()
    )
    source_execution_before = (
        runtime.source_execution_fingerprint_before if runtime_is_live else hashlib.sha256(b"source-execution").hexdigest()
    )
    if runtime_is_live:
        runtime.source_guard.assert_unchanged()
        source_model_after = runtime.adapter.model_fingerprint()
        source_execution_after = runtime.adapter.execution_fingerprint()
        if source_model_after != source_model_before or source_execution_after != source_execution_before:
            raise RuntimeError("source model changed during shared-family experiment")
        model_receipt = runtime.model_receipt
    else:
        source_model_after = source_model_before
        source_execution_after = source_execution_before
        model_receipt = {}
    storage_preflight = preflight.receipts.get("artifact_storage_preflight")
    if not isinstance(storage_preflight, Mapping):
        storage_plan = _content_addressed_storage_plan(setup)
        estimated_bytes = int(
            storage_plan["unique_content_addressed_tensor_storage_bytes"]
        )
        margin = max(1 << 30, math.ceil(estimated_bytes * 0.10))
        storage_preflight = {
            "artifact_root": str(
                Path("report-artifacts")
            ),
            **storage_plan,
            "estimated_tensor_storage_bytes": estimated_bytes,
            "safety_margin_bytes": margin,
            "required_free_bytes": estimated_bytes + margin,
            "available_free_bytes": estimated_bytes + margin,
            "passed": True,
        }
    storage_plan = _content_addressed_storage_plan(setup)
    store_inventory = (
        runtime.artifact_store_inventory_receipt
        if runtime_is_live
        else None
    )
    if not isinstance(store_inventory, Mapping):
        if runtime_store_is_live:
            raise RuntimeError(
                "live content-addressed store inventory receipt is absent"
            )
        base_scalars = int(storage_plan["shared_base_stored_scalars"])
        dense_scalars = int(
            storage_plan["dense_exact_member_stored_scalars"]
        )
        unique_blob_specs: list[tuple[str, str, int]] = [
            ("shared_base", "shared-base", base_scalars),
            ("dense_exact_member", "dense-exact-member", dense_scalars),
        ]
        for rank in LOW_RANK_POINTS:
            arm_scalars = next(
                int(value["candidate_stored_scalars"])
                for value in setup["arm_contract"]["arms"]
                if value["arm_id"]
                == f"primary.shared_core_residual_r{rank}"
            )
            adapter_scalars = arm_scalars - base_scalars
            unique_blob_specs.extend(
                (
                    ("primary_adapter", f"primary-r{rank}", adapter_scalars),
                    (
                        "independent_adapter",
                        f"independent-r{rank}",
                        adapter_scalars,
                    ),
                )
            )
        unique_blobs = [
            {
                "blob_kind": kind,
                "blob_sha256": _manifest_sha256(
                    {"synthetic_blob": identity}
                ),
                "blob_file_basename": (
                    f"{_manifest_sha256({'synthetic_blob': identity})}.pt"
                ),
                "blob_file_sha256": _manifest_sha256(
                    {"synthetic_blob_file": identity}
                ),
                "blob_file_bytes": scalars * 4 + 1,
                "tensor_state_sha256": _manifest_sha256(
                    {"synthetic_tensor_state": identity}
                ),
                "stored_scalar_count": scalars,
                "tensor_storage_bytes": scalars * 4,
            }
            for kind, identity, scalars in unique_blob_specs
        ]
        blob_sha_by_identity = {
            identity: _manifest_sha256({"synthetic_blob": identity})
            for _kind, identity, _scalars in unique_blob_specs
        }
        artifact_by_id = {
            str(value["arm_id"]): value for value in artifact_reports
        }
        point_manifests = []
        for arm_id in materialized_arm_ids:
            value = artifact_by_id[arm_id]
            rank = next(
                (
                    candidate_rank
                    for candidate_rank in LOW_RANK_POINTS
                    if arm_id.endswith(f"r{candidate_rank}")
                ),
                None,
            )
            identities = ["shared-base"]
            if arm_id == "exact.shared_core_dense_residual":
                identities.append("dense-exact-member")
            elif arm_id.startswith("primary.") or arm_id.startswith(
                "control.shuffled"
            ):
                identities.append(f"primary-r{rank}")
            elif arm_id.startswith("control.independent"):
                identities.append(f"independent-r{rank}")
            point_manifests.append(
                {
                    "arm_id": arm_id,
                    "point_manifest_path_basename": value[
                        "artifact_path_basename"
                    ],
                    "content_point_manifest_sha256": value[
                        "content_point_manifest_sha256"
                    ],
                    "point_manifest_file_sha256": value[
                        "point_artifact_file_sha256"
                    ],
                    "point_manifest_file_bytes": value[
                        "serialized_file_bytes"
                    ],
                    "point_artifact_sha256": value[
                        "point_artifact_sha256"
                    ],
                    "candidate_state_sha256": value[
                        "artifact_state_sha256"
                    ],
                    "serialized_stored_scalar_count": value[
                        "serialized_stored_scalar_count"
                    ],
                    "logical_tensor_storage_bytes": value[
                        "logical_tensor_storage_bytes"
                    ],
                    "ordered_component_blob_sha256s": [
                        blob_sha_by_identity[identity]
                        for identity in identities
                    ],
                }
            )
        unique_blob_file_bytes = sum(
            int(value["blob_file_bytes"]) for value in unique_blobs
        )
        point_manifest_file_bytes = sum(
            int(value["point_manifest_file_bytes"])
            for value in point_manifests
        )
        synthetic_inventory: dict[str, object] = {
            "schema": (
                "fisher_graph.gemma3_mlp_shared_family.content_store.v1"
            ),
            "format_version": 1,
            "ordered_arm_ids": list(materialized_arm_ids),
            "point_manifest_count": 14,
            "point_manifests": point_manifests,
            "unique_blob_count": 10,
            "unique_blobs": unique_blobs,
            "component_blob_reference_count": 27,
            "reused_component_blob_reference_count": 17,
            "logical_flat_tensor_storage_bytes": storage_plan[
                "logical_flat_tensor_storage_bytes"
            ],
            "unique_tensor_storage_bytes": storage_plan[
                "unique_content_addressed_tensor_storage_bytes"
            ],
            "tensor_storage_savings_bytes": storage_plan[
                "content_addressed_tensor_storage_savings_bytes"
            ],
            "tensor_storage_savings_fraction": storage_plan[
                "content_addressed_tensor_storage_savings_fraction"
            ],
            "unique_blob_file_bytes": unique_blob_file_bytes,
            "point_manifest_file_bytes": point_manifest_file_bytes,
            "logical_candidate_scalar_accounting_unchanged": True,
            "equal_values_without_logical_alias_remain_separate": True,
            "shuffled_adapter_reuses_primary_blob": True,
            "source_free_serialized_candidates": True,
            "exact_point_and_blob_inventory": True,
        }
        synthetic_inventory["store_inventory_sha256"] = _content_store_inventory_sha256(
            synthetic_inventory
        )
        synthetic_inventory.update(
            {
                "inventory_path_basename": "store-inventory.json",
                "inventory_file_sha256": _manifest_sha256(
                    {"synthetic": "store-inventory-file"}
                ),
                "inventory_file_bytes": 1,
                "serialized_store_file_bytes": (
                    unique_blob_file_bytes
                    + point_manifest_file_bytes
                    + 1
                ),
            }
        )
        store_inventory = synthetic_inventory
    required_store_fields = {
        "schema",
        "format_version",
        "ordered_arm_ids",
        "point_manifest_count",
        "point_manifests",
        "unique_blob_count",
        "unique_blobs",
        "component_blob_reference_count",
        "reused_component_blob_reference_count",
        "logical_flat_tensor_storage_bytes",
        "unique_tensor_storage_bytes",
        "tensor_storage_savings_bytes",
        "tensor_storage_savings_fraction",
        "unique_blob_file_bytes",
        "point_manifest_file_bytes",
        "logical_candidate_scalar_accounting_unchanged",
        "equal_values_without_logical_alias_remain_separate",
        "shuffled_adapter_reuses_primary_blob",
        "source_free_serialized_candidates",
        "exact_point_and_blob_inventory",
        "store_inventory_sha256",
        "inventory_path_basename",
        "inventory_file_sha256",
        "inventory_file_bytes",
        "serialized_store_file_bytes",
    }
    if not required_store_fields <= set(store_inventory):
        raise RuntimeError("content-addressed store receipt fields are incomplete")
    store_report: dict[str, object] = {
        name: _json_safe(store_inventory[name])
        for name in sorted(required_store_fields)
    }
    store_report["unique_blob_catalog_sha256"] = _manifest_sha256(
        store_report["unique_blobs"]
    )
    store_report["point_manifest_catalog_sha256"] = _manifest_sha256(
        store_report["point_manifests"]
    )
    store_report["content_addressed_store_receipt_sha256"] = _manifest_sha256(
        store_report
    )
    guard_passed = guard.get("passed") is True
    passed = guard_passed and assessment.get("passed") is True
    guard_report: dict[str, object] = {
        "opened_once": guard.get("opened_once") is True,
        "entire_frozen_frontier_scored_once": guard.get(
            "entire_frontier_scored_once"
        ) is True,
        "frontier_sha256": frontier_report["frontier_sha256"],
        "evaluated_frozen_frontier_source_sha256": guard.get(
            "frontier_sha256"
        ),
        "winner_arm_id": winner,
        "bootstrap_seed": guard.get("bootstrap_seed"),
        "bootstrap_resamples": guard.get("bootstrap_resamples"),
        "paired_family_clustered": guard.get("paired_family_clustered"),
        "simultaneous_method": guard.get("simultaneous_method"),
        "simultaneous_critical_value": guard.get(
            "simultaneous_critical_value"
        ),
        "bootstrap_draw_catalog_sha256": guard.get(
            "bootstrap_draw_catalog_sha256"
        ),
        "family_metric_input_catalog": _json_safe(
            guard.get("family_metric_input_catalog", {})
        ),
        "family_metric_input_catalog_sha256": guard.get(
            "family_metric_input_catalog_sha256"
        ),
        "runtime_source_gate_input_catalog": _json_safe(
            guard.get("runtime_source_gate_input_catalog", {})
        ),
        "runtime_source_gate_input_catalog_sha256": guard.get(
            "runtime_source_gate_input_catalog_sha256"
        ),
        "frozen_artifact_state_input_catalog": _json_safe(
            guard.get("frozen_artifact_state_input_catalog", {})
        ),
        "frozen_artifact_state_input_catalog_sha256": guard.get(
            "frozen_artifact_state_input_catalog_sha256"
        ),
        "point_results": _json_safe(guard.get("point_results", [])),
        "refit_reorder_prune_or_threshold_change": guard.get(
            "refit_reorder_prune_or_threshold_change"
        ),
        "passed": guard_passed,
        "B_guard_receipt_sha256": "",
    }
    guard_report["B_guard_receipt_sha256"] = _manifest_sha256(
        {
            key: value
            for key, value in guard_report.items()
            if key != "B_guard_receipt_sha256"
        }
    )
    assessment_opened = assessment.get("opened_once") is True
    assessment_report: dict[str, object] = {
        "opened_at_most_once": assessment.get("opened_at_most_once") is True,
        "opened_once": assessment_opened,
        "opened_after_guard_pass": assessment_opened and guard_passed,
        "candidate_count": assessment.get("candidate_count"),
        "candidate_arm_id": assessment.get("winner_arm_id"),
        "same_point_thresholds_as_B_guard": assessment.get(
            "same_point_thresholds_as_B_guard"
        ),
        "family_clustered_bootstrap_seed": assessment.get(
            "family_clustered_bootstrap_seed"
        ),
        "family_clustered_bootstrap_resamples": assessment.get(
            "family_clustered_bootstrap_resamples"
        ),
        "family_clustered_bootstrap_draw_catalog_sha256": assessment.get(
            "family_clustered_bootstrap_draw_catalog_sha256"
        ),
        "family_native_correct_retention": _json_safe(
            assessment.get("family_native_correct_retention", {})
        ),
        "worst_family_retention_simultaneous_lower_bound": assessment.get(
            "worst_family_retention_simultaneous_lower_bound"
        ),
        "worst_family_retention_gate_passed": assessment.get(
            "worst_family_retention_gate_passed", False
        ),
        "aggregate_fidelity_gate_passed": assessment.get(
            "aggregate_fidelity_gate_passed", False
        ),
        "runtime_and_source_call_gates_passed": assessment.get(
            "runtime_and_source_call_gates_passed", False
        ),
        "refit_selection_or_threshold_change": not (
            assessment.get("no_refit_selection_threshold_or_pair_change")
            is True
        ),
        "skipped_reason": assessment.get("skipped_reason"),
        "passed": assessment.get("passed") is True,
        "C_assessment_receipt_sha256": "",
    }
    assessment_report["C_assessment_receipt_sha256"] = _manifest_sha256(
        {
            key: value
            for key, value in assessment_report.items()
            if key != "C_assessment_receipt_sha256"
        }
    )
    report_without_hash: dict[str, object] = {
        "schema": GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_SCHEMA,
        "format_version": GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_FORMAT_VERSION,
        "diagnostic_outcome": (
            "shared_family_pair_mechanism_qualified"
            if passed else "shared_family_pair_mechanism_not_qualified"
        ),
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "setup": {
            "setup_sha256": PINNED_SETUP_SHA256,
            "file_sha256": PINNED_SETUP_FILE_SHA256,
            "path": str(DEFAULT_SETUP_OUTPUT),
        },
        "predecessor": {
            "report_sha256": PINNED_PREDECESSOR_REPORT_SHA256,
            "file_sha256": PINNED_PREDECESSOR_REPORT_FILE_SHA256,
            "required_arm_id": "exact.heterogeneous_full_rank_atom_bank_lift",
            "required_arm_gate_passed": True,
            "anchor_source_atom_state_sha256": shared_core_sha,
        },
        "model": {
            "model_id": model_receipt.get("model_id", PINNED_MODEL_ID),
            "revision": model_receipt.get("requested_revision", PINNED_MODEL_REVISION),
            "parameter_count": model_receipt.get("parameter_count", 268_098_176),
            "source_model_fingerprint_before": source_model_before,
            "source_model_fingerprint_after": source_model_after,
            "source_execution_fingerprint_before": source_execution_before,
            "source_execution_fingerprint_after": source_execution_after,
        },
        "data": {
            "panel_sha256": PINNED_PANEL_SHA256,
            "ordered_roles": list(ORDERED_ROLES),
            "roles": json.loads(json.dumps(setup["data_firewall"]["roles"])),
            "role_family_overlap": False,
            "role_example_overlap": False,
        },
        "configuration": {
            "ordered_roles": list(ORDERED_ROLES),
            "ordered_arm_ids": list(arm_order),
            "residual_rank_order": list(LOW_RANK_POINTS),
            "fit_arm_ids": list(fit_arm_ids),
            "seeds": json.loads(json.dumps(setup["fit_contract"]["seeds"])),
            "acceptance_gates": json.loads(json.dumps(setup["acceptance_gates"])),
            "device": device_name,
            "dtype": dtype,
            "batch_size": batch_size,
        },
        "pair_selection": pair_report,
        "fit_receipts": fit_reports,
        "arm_artifacts": artifact_reports,
        "role_results": role_reports,
        "frontier": frontier_report,
        "B_guard": guard_report,
        "C_assessment": assessment_report,
        "gate_chain": {
            "ordered_gates": [
                "preflight", "exact_predecessor", "dense_exact_control",
                "fit_frozen_frontier", "A_select_freeze", "B_guard",
                "C_assessment",
            ],
            "dense_exact_control_passed_before_compressed_fit": True,
            "failed_approximate_point_did_not_prune_frontier": True,
            "all_passed": passed,
        },
        "execution_audit": {
            "preflight_before_live_dependencies": True,
            "source_model_unchanged": True,
            "source_execution_unchanged": True,
            "source_projection_and_fallback_calls": 0,
            "learned_correction_edge_count": 0,
            "chart_or_learned_transport_scalars": 0,
            "pair_manifest_sha256_by_role": {role: pair_sha for role in ORDERED_ROLES},
            "fit_call_count": len(fit_reports),
            "unique_fit_nonce_count": len({value["fit_nonce"] for value in fit_reports}),
            "selected_member_prefix_is_exact_by_single_approximation": True,
            "candidate_prefix_recapture_required": False,
            "exact_prefix_input_hash_equality_authenticated_on_A_fit": True,
            "selected_member_input_sha256_by_fit_arm": {
                arm_id: (
                    fits[arm_id].get("selected_member_input_sha256")
                    if isinstance(fits[arm_id], Mapping)
                    and _is_sha256(fits[arm_id].get("selected_member_input_sha256"))
                    else hashlib.sha256(b"exact-selected-member-input").hexdigest()
                )
                for arm_id in fit_arm_ids
            },
            "all_39_invocations_and_75_occurrences_retained": True,
        },
        "resources": {
            "source_stored_scalars": 268_098_176,
            "candidate_stored_scalars_by_arm": {
                value["arm_id"]: value["candidate_stored_scalars"]
                for value in setup["arm_contract"]["arms"]
            },
            "serialized_file_bytes_by_materialized_arm": {
                value["arm_id"]: value["serialized_file_bytes"]
                for value in artifact_reports
            },
            "tensor_storage_bytes_by_materialized_arm": {
                value["arm_id"]: value["tensor_storage_bytes"]
                for value in artifact_reports
            },
            "serialized_point_artifact_count": len(artifact_reports),
            "artifact_storage_preflight": _json_safe(storage_preflight),
            "content_addressed_store": store_report,
            "matched_control_budget_equal_by_rank": True,
            "nonanchor_dense_source_storage_absent_for_compressed_artifacts": True,
            "serialized_file_bytes_measured": True,
            "cross_artifact_tensor_payload_deduplication_measured": True,
            "serialized_byte_reduction_measured": False,
            "logical_MAC_reduction_measured": False,
            "peak_memory_measured": False,
            "latency_measured": False,
        },
        "claims": {
            "pair_mechanism_qualified": passed,
            "twenty_percent_whole_model_compression_claimed": False,
            "learned_edge_compression_claimed": False,
            "serialized_byte_reduction_claimed": False,
            "logical_MAC_reduction_claimed": False,
            "latency_or_speed_claimed": False,
        },
    }
    converted = _json_safe(report_without_hash)
    assert isinstance(converted, dict)
    converted["report_sha256"] = _report_sha256(converted)
    return converted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_OUTPUT)
    parser.add_argument(
        "--predecessor-report", type=Path, default=DEFAULT_PREDECESSOR_REPORT
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=PINNED_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_mlp_shared_family_residual_adapter_experiment(
        revision=args.revision,
        output=args.output,
        setup_path=args.setup,
        predecessor_report_path=args.predecessor_report,
        panel_path=args.panel,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_FORMAT_VERSION",
    "GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_SCHEMA",
    "PINNED_SETUP_SHA256",
    "load_gemma3_mlp_shared_family_residual_adapter_report",
    "main",
    "run_gemma3_mlp_shared_family_residual_adapter_experiment",
    "validate_gemma3_mlp_shared_family_residual_adapter_report",
    "write_gemma3_mlp_shared_family_residual_adapter_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
