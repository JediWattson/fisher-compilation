"""Real one-fold A5c broader-row selected-generator runner.

This rung preserves the canonical A5b v1 artifact as immutable failure
evidence, widens outer-fold-zero training capture to four examples per each of
the seven permitted families, solves downstream-sensitive coordinates for
every valid captured row, and performs nested family-disjoint ridge selection.

No outer-held-family batch is selected until the winning learned executable or
the exact source-graph fallback has been reduced to immutable graph, lowering,
composition, lineage, and selection-freeze hashes.  The final assessment is a
bounded full-model-logit development result, not held-out confirmation and not
a whole-model compilation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .computational_modes import ComputationalModeBasis
from .downstream_affine_coordinate_solver import DownstreamAffineSolverConfig
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l10_l17_a4_oracle_attribution import (
    DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT,
)
from .gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT,
    _authenticate_a4_oracle_chain,
    _file_sha256,
    _take_first_examples,
    build_frozen_affine_image,
    load_a5_frozen_affine_capacity_report,
)
from .gemma3_l10_l17_a5b_batched_capacity import (
    TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW,
    solve_batched_frozen_affine_capacity_rows,
)
from .gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    a5b_tensor_sha256,
    build_a5b_downstream_coordinate_target_bridge,
)
from .gemma3_l10_l17_a5b_generator_microcanary import (
    DEFAULT_GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_OUTPUT,
    _EXPECTED_A5A_FILE_SHA256,
    _EXPECTED_A5A_REPORT_SHA256,
    _EXPECTED_MODEL_FINGERPRINT,
    _EXPECTED_MODEL_REVISION,
    _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256,
    _REVISION,
    _TARGET_RECEIPT_DOMAIN,
    load_a5b_generator_microcanary_report,
)
from .gemma3_l10_l17_a5c_breadth_split import (
    build_a5c_breadth_split_from_bridge,
)
from .gemma3_l10_l17_a5c_family_ridge_cv import (
    A5C_FIXED_GENERATOR_RANK,
    A5C_RIDGE_GRID,
    A5cFamilyRidgeCvSelection,
    select_a5c_family_disjoint_ridge,
)
from .gemma3_l10_l17_a5c_report import (
    DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT,
    a5c_outer_evaluation_sha256,
    a5c_resource_accounting_sha256,
    a5c_selection_freeze_sha256,
    a5c_source_scorer_evaluation_sha256,
    a5c_source_scorer_receipt_sha256,
    a5c_target_token_locality_lineage_sha256,
)
from .gemma3_l10_l17_a5c_prepublication_bundle import (
    default_a5c_prepublication_bundle_path,
    finalize_a5c_prepublication_bundle,
    publish_a5c_report_with_prepublication_bundle,
)
from .gemma3_l10_l17_full_block_closure_bundle import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
)
from .gemma3_l10_l17_full_block_closure_lofo import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    _build_source_runtime_catalog,
    _canonical_json_bytes,
    _fold_catalog,
    _full_block_capture_audit_receipt,
    _ordered_restored_lowerings,
    _POST_DELTA_BOUNDARY,
    _source_lowering_maps,
    _validate_frozen_selection,
    _validate_source_decoder_contract,
    _validate_source_runtime_catalog,
)
from .gemma3_l10_l17_open_a_progressive_evaluation import (
    DEFAULT_COMPOSITION_BUNDLE_PATH,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    project_joint_target_to_frozen_bases,
    replace_layer_nodes_in_composed_graph,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    _authenticate_before_fit_access,
    _merge_corrected_composition_lowerings,
    _shared_compiled_input,
    _validate_fold_evaluation,
    score_trajectory_correction_fold,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_family_lofo_authority import (
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_full_block_closure_capture import (
    capture_gemma3_layer17_full_block_closure,
)
from .gemma3_layer17_v8_fit_lofo import _blocks_to_device, _family_blocks
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    select_top_fisher_same_layer_fragments,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering


__all__ = [
    "A5cBroaderTrainingWorkspace",
    "run_gemma3_l10_l17_a5c_broader_selected_generator",
]


_EXPECTED_A5B_FILE_SHA256 = (
    "c0ad54d30ac192c7ac8002aa8f9cf0fe78ba83edfed9c74502bc0b50ca1dd023"
)
_EXPECTED_A5B_REPORT_SHA256 = (
    "27397857c1707c774b5d78ead1ca0770e6a3d5eaa43a0b59e502c6b190bb2cd5"
)
_A5B_LEARNED_COMPOSITION = {
    "nll_per_token": 11.708115005493164,
    "delta_nll_per_token": 4.550132043021066,
    "native_to_candidate_kl_per_token": 3.4907681586316577,
    "top1_agreement_to_native": 0.4857142857142857,
}
_OUTER_FOLD_INDEX = 0
_TRAINING_FAMILY_COUNT = 7
_TRAINING_EXAMPLES_PER_FAMILY = 4
_INNER_AUDIT_EXAMPLES_PER_FAMILY = 1
_TARGET_SOLVER_STEPS = 64
# Each solve materializes full-vocabulary teacher/candidate logits and their
# float64 KL views.  Eight rows is the already-audited A5b token-local batch
# and bounds those dominant tensors while retaining useful CPU throughput.
_TARGET_BATCH_ROWS = 8
_TARGET_LEARNING_RATE_FRACTION = 1.0e-2
_TOKEN_LOCALITY_ATOL = 1.0e-6
_TOKEN_LOCALITY_RTOL = 2.0e-6
_FINAL_HEAD_CHUNK_ROWS = 8
_TARGET_RIDGE = 0.0
_TARGET_TRUST_RADIUS = None
_BREADTH_SPLIT_DOMAIN = b"fisher-graph:a5c-breadth-split-binding:v1\0"
_BRIDGE_SPLIT_DOMAIN = b"fisher-graph:a5c-bridge-split-binding:v1\0"
_ROW_CATALOG_DOMAIN = b"fisher-graph:a5c-all-valid-row-catalog:v1\0"
_CAPTURE_AUDIT_DOMAIN = b"fisher-graph:a5c-capture-audit:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_value(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _progress(message: str) -> None:
    print(f"[a5c-broader] {message}", flush=True)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_tensor(child) for child in value)
    return False


@dataclass(frozen=True, slots=True)
class A5cBroaderTrainingWorkspace:
    """Ephemeral authenticated state at the post-breadth continuation seam.

    This workspace intentionally contains live tensors and runtime objects and
    must never be serialized as a report.  It exists only long enough for an
    internal continuation to consume the already-authenticated A5c capture,
    solve, bridge, and breadth orchestration before any A5c CV or outer-held
    batch access occurs.
    """

    adapter: Gemma3CausalLMAdapter
    model: object = field(repr=False)
    tokenizer: object = field(repr=False)
    device: torch.device
    blocks: Mapping[str, object] = field(repr=False)
    held_family_alias: str
    training_family_aliases: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    family_alias_by_example: Mapping[str, str]
    fold: Mapping[str, object]
    materialization: Mapping[str, object]

    source_composition_graph: ModalGeneratorGraphPlan
    source_composition_lowerings: tuple[ModalGeneratorLowering, ...]
    layer10_graph: ModalGeneratorGraphPlan
    layer17_graph: ModalGeneratorGraphPlan
    layer10_lowerings_by_node: Mapping[str, ModalGeneratorLowering]
    layer17_lowerings_by_node: Mapping[str, ModalGeneratorLowering]
    bases_by_node: Mapping[str, ComputationalModeBasis]
    fragment_id_by_node: Mapping[str, str]
    fragment_selection: SameLayerFragmentSelection
    frozen_affine_image: object = field(repr=False)

    capture: object = field(repr=False)
    capture_metadata: Mapping[str, object]
    capture_audit: Mapping[str, object]
    capture_sha256: str
    capture_audit_sha256: str
    all_row_keys: tuple[tuple[str, int], ...]
    row_catalog_sha256: str
    compiled_inputs: Tensor = field(repr=False)
    native_block_states: Tensor = field(repr=False)
    frozen_compiled_block_states: Tensor = field(repr=False)
    compiled_correction_base_states: Tensor = field(repr=False)
    compiled_post_attention_residual: Tensor = field(repr=False)
    compiled_compact_retained_delta: Tensor = field(repr=False)
    a4_full_block_closure_target: Tensor = field(repr=False)

    target_solution: object = field(repr=False)
    target_receipt: Mapping[str, object]
    target_report_receipt: Mapping[str, object]
    target_compact: Mapping[str, object]
    target_hashes: Mapping[str, object]
    target_token_locality_lineage_sha256: str
    bridge: object = field(repr=False)
    bridge_receipt: Mapping[str, object]
    coordinate_row_bank_compact: Mapping[str, object]
    breadth: object = field(repr=False)
    breadth_receipt: Mapping[str, object]
    breadth_compact: Mapping[str, object]

    runtime_metadata: Mapping[str, object]
    source_bindings: Mapping[str, object]
    comparison_to_a5b: Mapping[str, object]
    configuration_metadata: Mapping[str, object]
    source_runtime_catalog: Mapping[str, object]
    source_runtime_metadata: Mapping[str, object]
    source_authorization: Mapping[str, object]
    fit_authorization: Mapping[str, object]
    protocol: Mapping[str, object]
    a5b_report: Mapping[str, object]
    a5a_report: Mapping[str, object]
    a4_oracle_report: Mapping[str, object]
    source_a4_report: Mapping[str, object]
    fold_bundle: Mapping[str, object]
    output_path: Path


def _dispatch_a5c_training_continuation(
    workspace: A5cBroaderTrainingWorkspace,
    continuation: Callable[
        [A5cBroaderTrainingWorkspace], dict[str, object]
    ],
) -> dict[str, object]:
    """Invoke one opt-in post-breadth continuation and validate its result."""

    if not isinstance(workspace, A5cBroaderTrainingWorkspace):
        raise TypeError("A5c continuation workspace is invalid")
    if not callable(continuation):
        raise TypeError("A5c continuation must be callable")
    result = continuation(workspace)
    if not isinstance(result, dict):
        raise TypeError("A5c continuation must return a report dictionary")
    return result


def _authenticate_canonical_a5b_failure(
    path: Path | str,
) -> tuple[dict[str, object], dict[str, object]]:
    source = Path(path)
    report = load_a5b_generator_microcanary_report(source)
    conclusion = report.get("conclusion")
    evaluation = report.get("evaluation")
    if (
        _file_sha256(source) != _EXPECTED_A5B_FILE_SHA256
        or report.get("report_sha256") != _EXPECTED_A5B_REPORT_SHA256
        or not isinstance(conclusion, Mapping)
        or conclusion.get("learned_generator_improves_frozen_kl") is not False
        or conclusion.get("learned_generator_improves_frozen_delta_nll")
        is not False
        or conclusion.get("learned_generator_improves_frozen_top1") is not False
        or not isinstance(evaluation, Mapping)
    ):
        raise ValueError("A5c requires canonical A5b v1 failure evidence")
    conditions = evaluation.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError("canonical A5b conditions are unavailable")
    learned = conditions.get("trajectory_corrected_composition")
    if learned != _A5B_LEARNED_COMPOSITION:
        raise ValueError("canonical A5b learned-composition metrics drifted")
    evidence = {
        "a5b_report_sha256": _EXPECTED_A5B_REPORT_SHA256,
        "same_outer_fold": True,
        "same_held_example_policy": True,
        "a5b_learned_composition": dict(_A5B_LEARNED_COMPOSITION),
    }
    return report, evidence


def _graph_descriptor(
    *,
    layer17_graph: ModalGeneratorGraphPlan,
    layer17_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    composition_graph: ModalGeneratorGraphPlan,
    layer17_post_feedforward_delta_layer_ordinals: Sequence[int],
    composition_post_feedforward_delta_layer_ordinals: Sequence[int],
) -> dict[str, object]:
    layer17_graph.validate_integrity()
    composition_graph.validate_integrity()
    if (
        set(layer17_graph.traversal_order) != set(layer17_lowerings_by_node)
        or len(layer17_lowerings_by_node) != 4
    ):
        raise ValueError("A5c descriptor Layer17 lowering catalog drifted")
    node_by_name = {node.name: node for node in layer17_graph.nodes}
    authenticated_lowerings: dict[str, ModalGeneratorLowering] = {}
    for name in layer17_graph.traversal_order:
        lowering = layer17_lowerings_by_node[name]
        if not isinstance(lowering, ModalGeneratorLowering):
            raise TypeError("A5c descriptor lowering has an invalid type")
        restored = ModalGeneratorLowering.from_state_dict(lowering.state_dict())
        if restored.artifact_sha256 != lowering.artifact_sha256:
            raise ValueError("A5c descriptor lowering roundtrip identity drifted")
        node = node_by_name.get(name)
        if (
            node is None
            or node.weights.artifact_sha256
            != restored.graph_weights.artifact_sha256
        ):
            raise ValueError(
                "A5c descriptor graph node and lowering weights are unpaired"
            )
        authenticated_lowerings[name] = restored
    layer17_post_delta = tuple(layer17_post_feedforward_delta_layer_ordinals)
    composition_post_delta = tuple(
        composition_post_feedforward_delta_layer_ordinals
    )
    if layer17_post_delta not in ((), (17,)) or composition_post_delta not in (
        (),
        (17,),
    ):
        raise ValueError("A5c descriptor post-feed-forward policy is invalid")
    return {
        "layer17_graph_sha256": layer17_graph.artifact_sha256,
        "layer17_lowering_sha256_by_node": {
            name: authenticated_lowerings[name].artifact_sha256
            for name in layer17_graph.traversal_order
        },
        "composition_graph_sha256": composition_graph.artifact_sha256,
        "layer17_parameter_count": layer17_graph.parameter_count,
        "layer17_macs_per_token": layer17_graph.macs_per_token,
        "composition_parameter_count": composition_graph.parameter_count,
        "composition_macs_per_token": composition_graph.macs_per_token,
        "layer17_post_feedforward_delta_layer_ordinals": list(
            layer17_post_delta
        ),
        "composition_post_feedforward_delta_layer_ordinals": list(
            composition_post_delta
        ),
    }


@dataclass(frozen=True, slots=True)
class _FrozenA5cExecutable:
    kind: str
    selected_ridge: float | None
    layer17_graph: ModalGeneratorGraphPlan = field(repr=False)
    layer17_lowerings_by_node: Mapping[str, ModalGeneratorLowering] = field(
        repr=False
    )
    composition_graph: ModalGeneratorGraphPlan = field(repr=False)
    composition_lowerings: tuple[ModalGeneratorLowering, ...] = field(repr=False)
    selected_descriptor: Mapping[str, object]
    frozen_descriptor: Mapping[str, object]
    lineage: Mapping[str, object]
    selection_freeze_sha256: str

    def report_section(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "selected_ridge": self.selected_ridge,
            "lineage": dict(self.lineage),
            "selected": dict(self.selected_descriptor),
            "frozen_reference": dict(self.frozen_descriptor),
            "selection_freeze_sha256": self.selection_freeze_sha256,
        }


def _freeze_selected_executable(
    *,
    selection: A5cFamilyRidgeCvSelection,
    source_layer17_graph: ModalGeneratorGraphPlan,
    source_layer17_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    source_composition_graph: ModalGeneratorGraphPlan,
    source_composition_lowerings: tuple[ModalGeneratorLowering, ...],
    layer10_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    lineage: Mapping[str, object],
) -> _FrozenA5cExecutable:
    """Reduce a CV decision to immutable executable identities."""

    receipt = selection.receipt()
    cv_selection = receipt.get("selection")
    if not isinstance(cv_selection, Mapping):
        raise TypeError("A5c CV selection receipt is unavailable")
    frozen_descriptor = _graph_descriptor(
        layer17_graph=source_layer17_graph,
        layer17_lowerings_by_node=source_layer17_lowerings_by_node,
        composition_graph=source_composition_graph,
        layer17_post_feedforward_delta_layer_ordinals=(),
        composition_post_feedforward_delta_layer_ordinals=(),
    )
    if selection.use_frozen_fallback:
        if selection.selected_ridge is not None or selection.correction_fit is not None:
            raise ValueError("A5c frozen fallback contains a learned correction")
        kind = "frozen_source_fallback"
        layer17_graph = source_layer17_graph
        layer17_lowerings = dict(source_layer17_lowerings_by_node)
        composition_graph = source_composition_graph
        composition_lowerings = source_composition_lowerings
        selected_descriptor = dict(frozen_descriptor)
    else:
        correction = selection.correction_fit
        if correction is None or selection.selected_ridge is None:
            raise ValueError("A5c learned selection lacks its post-CV fit")
        kind = "learned_correction"
        layer17_graph = correction.graph_plan
        layer17_lowerings = dict(correction.lowerings_by_node)
        composition_graph = replace_layer_nodes_in_composed_graph(
            source_composition_graph,
            layer17_graph,
            layer_ordinal=17,
        )
        composition_lowerings = _merge_corrected_composition_lowerings(
            composition_graph,
            layer10_lowerings_by_node=layer10_lowerings_by_node,
            corrected_layer17_lowerings_by_node=layer17_lowerings,
        )
        selected_descriptor = _graph_descriptor(
            layer17_graph=layer17_graph,
            layer17_lowerings_by_node=layer17_lowerings,
            composition_graph=composition_graph,
            layer17_post_feedforward_delta_layer_ordinals=(17,),
            composition_post_feedforward_delta_layer_ordinals=(17,),
        )
    for descriptor, label in (
        (frozen_descriptor, "frozen"),
        (selected_descriptor, "selected"),
    ):
        if (
            descriptor["layer17_parameter_count"] != 163_094
            or descriptor["layer17_macs_per_token"] != 160_352
            or descriptor["composition_parameter_count"] != 295_129
            or descriptor["composition_macs_per_token"] != 289_600
        ):
            raise RuntimeError(f"A5c {label} executable resources drifted")
    expected_input_lineage_fields = {
        "target_solve_receipt_sha256",
        "coordinate_row_bank_receipt_sha256",
        "breadth_split_receipt_sha256",
        "ridge_cv_receipt_sha256",
        "layer10_graph_sha256",
        "layer10_lowering_sha256_by_node",
    }
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != expected_input_lineage_fields
    ):
        raise ValueError("A5c selection lineage fields are invalid")
    layer10_lineage = lineage["layer10_lowering_sha256_by_node"]
    if not isinstance(layer10_lineage, Mapping) or len(layer10_lineage) != 4:
        raise ValueError("A5c selection lineage lacks four Layer10 lowerings")
    frozen_lineage: dict[str, object] = {
        name: _require_sha256(value, label=f"A5c lineage {name}")
        for name, value in lineage.items()
        if name != "layer10_lowering_sha256_by_node"
    }
    frozen_lineage["layer10_lowering_sha256_by_node"] = {
        str(name): _require_sha256(
            digest, label=f"A5c lineage Layer10 lowering {name}"
        )
        for name, digest in layer10_lineage.items()
    }
    frozen_lineage["matched_double_deletion_graph_sha256"] = str(
        selected_descriptor["composition_graph_sha256"]
    )
    if frozen_lineage["ridge_cv_receipt_sha256"] != receipt.get(
        "receipt_sha256"
    ):
        raise ValueError("A5c executable does not bind its CV receipt")
    selected_ridge = selection.selected_ridge
    if (
        cv_selection.get("selected_ridge") != selected_ridge
        or cv_selection.get("use_frozen_fallback")
        is not selection.use_frozen_fallback
    ):
        raise ValueError("A5c in-memory selection contradicts its receipt")
    freeze_sha = a5c_selection_freeze_sha256(
        kind=kind,
        selected_ridge=selected_ridge,
        lineage=frozen_lineage,
        selected=selected_descriptor,
        frozen_reference=frozen_descriptor,
    )
    return _FrozenA5cExecutable(
        kind=kind,
        selected_ridge=selected_ridge,
        layer17_graph=layer17_graph,
        layer17_lowerings_by_node=layer17_lowerings,
        composition_graph=composition_graph,
        composition_lowerings=tuple(composition_lowerings),
        selected_descriptor=selected_descriptor,
        frozen_descriptor=frozen_descriptor,
        lineage=frozen_lineage,
        selection_freeze_sha256=freeze_sha,
    )


def _select_held_scoring_batch_after_freeze(
    *,
    blocks: Mapping[str, object],
    held_family_alias: str,
    executable: _FrozenA5cExecutable,
) -> tuple[object, ...]:
    """Select the held scoring batch only after the executable is frozen.

    The sealed authority materializes all eight family blocks earlier.  This
    boundary therefore makes no claim that the held collection was never
    accessed; it guarantees only that no held batch was selected for capture,
    target solving, fitting, CV, or scoring before this point.
    """

    expected = a5c_selection_freeze_sha256(
        kind=executable.kind,
        selected_ridge=executable.selected_ridge,
        lineage=executable.lineage,
        selected=executable.selected_descriptor,
        frozen_reference=executable.frozen_descriptor,
    )
    if expected != executable.selection_freeze_sha256:
        raise RuntimeError("A5c executable is not frozen before held access")
    if held_family_alias not in blocks:
        raise KeyError("outer-held family is unavailable")
    return tuple(_take_first_examples(blocks[held_family_alias], 1))


def _build_scoring_executors(
    *,
    adapter: Gemma3CausalLMAdapter,
    layer10_graph: ModalGeneratorGraphPlan,
    layer10_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    source_composition_graph: ModalGeneratorGraphPlan,
    source_composition_lowerings: tuple[ModalGeneratorLowering, ...],
    executable: _FrozenA5cExecutable,
) -> tuple[
    Gemma3ModalGeneratorGraphExecutor,
    Gemma3ModalGeneratorGraphExecutor,
    Gemma3ModalGeneratorGraphExecutor,
    Gemma3ModalGeneratorGraphExecutor,
]:
    """Instantiate four distinct executors, including on source fallback."""

    layer10_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer10_graph,
        tuple(
            layer10_lowerings_by_node[name]
            for name in layer10_graph.traversal_order
        ),
    )
    selected_post_delta = (17,) if executable.kind == "learned_correction" else ()
    selected_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        executable.layer17_graph,
        tuple(
            executable.layer17_lowerings_by_node[name]
            for name in executable.layer17_graph.traversal_order
        ),
        post_feedforward_delta_layer_ordinals=selected_post_delta,
    )
    frozen_composition_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        source_composition_graph,
        source_composition_lowerings,
    )
    selected_composition_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        executable.composition_graph,
        executable.composition_lowerings,
        post_feedforward_delta_layer_ordinals=selected_post_delta,
    )
    executors = (
        layer10_executor,
        selected_layer17_executor,
        frozen_composition_executor,
        selected_composition_executor,
    )
    if len({id(value) for value in executors}) != len(executors):
        raise RuntimeError("A5c scoring executors are not distinct")
    expected_layer17_post_delta = tuple(
        executable.selected_descriptor[
            "layer17_post_feedforward_delta_layer_ordinals"
        ]
    )
    expected_composition_post_delta = tuple(
        executable.selected_descriptor[
            "composition_post_feedforward_delta_layer_ordinals"
        ]
    )
    if (
        layer10_executor.post_feedforward_delta_layer_ordinals != ()
        or selected_layer17_executor.post_feedforward_delta_layer_ordinals
        != expected_layer17_post_delta
        or frozen_composition_executor.post_feedforward_delta_layer_ordinals
        != ()
        or selected_composition_executor.post_feedforward_delta_layer_ordinals
        != expected_composition_post_delta
    ):
        raise RuntimeError("A5c scoring executor post-delta policy is not frozen")
    if executable.kind == "frozen_source_fallback" and (
        selected_layer17_executor.graph_plan.artifact_sha256
        != executable.frozen_descriptor["layer17_graph_sha256"]
        or selected_composition_executor.graph_plan.artifact_sha256
        != executable.frozen_descriptor["composition_graph_sha256"]
    ):
        raise RuntimeError("A5c fallback executors differ from frozen source")
    return executors


def _metric_with_graph(
    value: object,
    *,
    graph_sha256: str,
) -> dict[str, object]:
    fields = {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("A5c source evaluation metric fields are invalid")
    parsed: dict[str, object] = {"graph_sha256": _require_sha256(
        graph_sha256, label="A5c evaluation graph"
    )}
    for name in fields:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"A5c evaluation {name} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"A5c evaluation {name} must be finite")
        parsed[name] = number
    return parsed


def _adapt_outer_evaluation(
    raw: Mapping[str, object],
    *,
    outer_fold_index: int,
    layer10_graph_sha256: str,
    executable: _FrozenA5cExecutable,
) -> dict[str, object]:
    conditions = raw.get("conditions")
    native = raw.get("native")
    resources = raw.get("resource_accounting")
    if not isinstance(conditions, Mapping) or not isinstance(native, Mapping):
        raise TypeError("A5c raw outer evaluation is unavailable")
    if not isinstance(resources, Mapping):
        raise TypeError("A5c raw scorer resource accounting is unavailable")
    expected = {
        "layer10_only",
        "trajectory_corrected_layer17_only",
        "frozen_uncorrected_composition",
        "trajectory_corrected_composition",
        "matched_double_deletion",
    }
    if set(conditions) != expected or set(native) != {"nll_per_token"}:
        raise ValueError("A5c raw evaluation condition panel drifted")
    frozen_composition = _metric_with_graph(
        conditions["frozen_uncorrected_composition"],
        graph_sha256=str(
            executable.frozen_descriptor["composition_graph_sha256"]
        ),
    )
    selected_composition = _metric_with_graph(
        conditions["trajectory_corrected_composition"],
        graph_sha256=str(
            executable.selected_descriptor["composition_graph_sha256"]
        ),
    )
    if executable.kind == "frozen_source_fallback":
        if selected_composition != frozen_composition:
            raise RuntimeError(
                "A5c fallback selected composition is not metric/hash-identical"
            )
        selected_composition = dict(frozen_composition)
    compact_conditions = {
        "layer10_only": _metric_with_graph(
            conditions["layer10_only"],
            graph_sha256=layer10_graph_sha256,
        ),
        "selected_layer17_only": _metric_with_graph(
            conditions["trajectory_corrected_layer17_only"],
            graph_sha256=str(
                executable.selected_descriptor["layer17_graph_sha256"]
            ),
        ),
        "frozen_uncorrected_composition": frozen_composition,
        "selected_composition": selected_composition,
        "matched_double_deletion": _metric_with_graph(
            conditions["matched_double_deletion"],
            graph_sha256=str(
                executable.selected_descriptor["composition_graph_sha256"]
            ),
        ),
    }
    condition_graphs = {
        name: value["graph_sha256"] for name, value in compact_conditions.items()
    }
    scorer_evaluation_sha256 = a5c_source_scorer_evaluation_sha256(raw)
    resource_accounting_sha256 = a5c_resource_accounting_sha256(resources)
    scorer_receipt_sha256 = a5c_source_scorer_receipt_sha256(
        source_scorer_evaluation=raw,
        outer_fold_index=outer_fold_index,
        condition_graph_sha256_by_name=condition_graphs,
    )
    result = {
        "assessment_role": "calibration_a_outer_family_bounded_development",
        "outer_fold_index": outer_fold_index,
        "logical_valid_tokens": raw.get("logical_valid_tokens"),
        "supervised_tokens": raw.get("supervised_tokens"),
        "native": {"nll_per_token": native["nll_per_token"]},
        "conditions": compact_conditions,
        "source_scorer_evaluation": json.loads(
            json.dumps(raw, allow_nan=False)
        ),
        "source_scorer_receipt_sha256": scorer_receipt_sha256,
        "source_scorer_evaluation_sha256": scorer_evaluation_sha256,
        "resource_accounting_sha256": resource_accounting_sha256,
        "resource_accounting_reference": (
            "score_trajectory_correction_fold.resource_accounting"
        ),
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
    }
    if (
        type(result["logical_valid_tokens"]) is not int
        or int(result["logical_valid_tokens"]) <= 0
        or type(result["supervised_tokens"]) is not int
        or int(result["supervised_tokens"]) <= 0
    ):
        raise ValueError("A5c outer evaluation token counts are invalid")
    return result


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} is unavailable")
    return value


def _compact_target_solve_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    hashes = _required_mapping(receipt.get("hashes"), label="A5c target hashes")
    compact = {
        "receipt_schema": receipt.get("schema"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "row_count": receipt.get("row_count"),
        "selected_coordinate_sha256": hashes.get(
            "selected_coefficient_sha256"
        ),
        "contains_tensor_payloads": receipt.get("contains_tensor_payloads"),
    }
    if compact["contains_tensor_payloads"] is not False or _contains_tensor(compact):
        raise ValueError("A5c compact target receipt is not tensor-free")
    return compact


def _compact_coordinate_row_bank(
    *,
    bridge: object,
    bridge_receipt: Mapping[str, object],
) -> dict[str, object]:
    authentication = _required_mapping(
        bridge_receipt.get("authentication"), label="A5c bridge authentication"
    )
    all_rows = getattr(bridge, "all_rows", None)
    ownership = getattr(bridge, "family_alias_by_example", None)
    receipt_sha256 = getattr(bridge, "receipt_sha256", None)
    if (
        all_rows is None
        or not isinstance(ownership, Mapping)
        or bridge_receipt.get("receipt_sha256") != receipt_sha256
    ):
        raise ValueError("A5c coordinate bridge identity drifted")
    compact = {
        "receipt_schema": bridge_receipt.get("schema"),
        "receipt_sha256": receipt_sha256,
        "row_count": getattr(all_rows, "observations", None),
        "example_count": len(ownership),
        "row_key_sha256": getattr(all_rows, "row_key_sha256", None),
        "compiled_inputs_sha256": authentication.get(
            "compiled_inputs_sha256"
        ),
        "selected_coordinates_sha256": authentication.get(
            "selected_joint_coordinates_sha256"
        ),
        "outer_held_family_rows_present": False,
        "contains_tensor_payloads": False,
    }
    if _contains_tensor(compact):
        raise RuntimeError("A5c compact coordinate row bank contains tensors")
    return compact


def _compact_breadth_split(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    source = _required_mapping(receipt.get("source"), label="A5c breadth source")
    final = _required_mapping(
        receipt.get("final_split"), label="A5c breadth final split"
    )
    fit = _required_mapping(final.get("fit"), label="A5c breadth fit")
    audit = _required_mapping(final.get("audit"), label="A5c breadth audit")
    quarantine = _required_mapping(
        receipt.get("collision_quarantine"),
        label="A5c breadth collision quarantine",
    )
    ownership = _required_mapping(
        receipt.get("ownership"), label="A5c breadth ownership"
    )
    safety = _required_mapping(receipt.get("safety"), label="A5c breadth safety")
    compact = {
        "receipt_schema": receipt.get("schema"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "all_row_count": source.get("observations"),
        "fit_row_count": fit.get("observations"),
        "audit_row_count": audit.get("observations"),
        "all_example_count": source.get("examples"),
        "fit_example_count": fit.get("examples"),
        "audit_example_count": audit.get("examples"),
        "fit_examples_fully_removed_for_signature_overlap": quarantine.get(
            "fit_examples_fully_removed"
        ),
        "removed_fit_rows_for_signature_overlap": quarantine.get(
            "fit_rows_removed"
        ),
        "fit_audit_example_overlap_count": final.get("example_overlap_count"),
        "post_purge_input_signature_overlap_count": final.get(
            "compiled_input_signature_overlap_count"
        ),
        "outer_held_family_rows_present": ownership.get(
            "outer_held_family_present"
        ),
        "contains_tensor_payloads": safety.get("contains_tensors"),
    }
    if compact["contains_tensor_payloads"] is not False or _contains_tensor(compact):
        raise ValueError("A5c compact breadth receipt is not tensor-free")
    return compact


def _compact_ridge_cv(receipt: Mapping[str, object]) -> dict[str, object]:
    configuration = _required_mapping(
        receipt.get("configuration"), label="A5c CV configuration"
    )
    ownership = _required_mapping(
        receipt.get("ownership"), label="A5c CV ownership"
    )
    selection = _required_mapping(
        receipt.get("selection"), label="A5c CV selection"
    )
    candidates = receipt.get("candidates")
    safety = _required_mapping(receipt.get("safety"), label="A5c CV safety")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("A5c CV candidates are unavailable")
    compact = {
        "receipt_schema": receipt.get("schema"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "candidate_count": len(candidates),
        "inner_fold_count": configuration.get("inner_fold_count"),
        "selected_ridge": selection.get("selected_ridge"),
        "use_frozen_fallback": selection.get("use_frozen_fallback"),
        "outer_held_family_accessed": ownership.get(
            "outer_held_family_states_or_rows_accessed"
        ),
        "contains_tensor_payloads": safety.get("contains_tensors"),
    }
    if compact["contains_tensor_payloads"] is not False or _contains_tensor(compact):
        raise ValueError("A5c compact CV receipt is not tensor-free")
    return compact


def _chronology(
    *,
    ridge_cv_receipt_sha256: str,
    selection_freeze_sha256: str,
    outer_evaluation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "ridge_cv_completed_event": 1,
        "executable_frozen_event": 2,
        "outer_held_batch_selected_event": 3,
        "outer_held_model_evaluated_event": 4,
        "outer_held_batch_selected_or_scored_before_freeze": False,
        "executable_frozen_before_outer_held_batch_selection": True,
        "executable_frozen_before_outer_held_model_evaluation": True,
        "ridge_cv_receipt_sha256": _require_sha256(
            ridge_cv_receipt_sha256, label="A5c chronology CV receipt"
        ),
        "selection_freeze_sha256": _require_sha256(
            selection_freeze_sha256, label="A5c chronology selection freeze"
        ),
        "outer_evaluation_sha256": a5c_outer_evaluation_sha256(
            outer_evaluation
        ),
    }


def run_gemma3_l10_l17_a5c_broader_selected_generator(
    *,
    revision: str,
    output: Path | str = DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT,
    a5b_path: Path | str = DEFAULT_GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_OUTPUT,
    a5a_path: Path | str = DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT,
    a4_oracle_path: Path | str = DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT,
    source_a4_report_path: Path | str = DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    fold_bundle_path: Path | str = DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    composition_bundle_path: Path | str = DEFAULT_COMPOSITION_BUNDLE_PATH,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    _continuation: Callable[
        [A5cBroaderTrainingWorkspace], dict[str, object]
    ]
    | None = None,
) -> dict[str, object]:
    """Run one fixed outer-fold-zero A5c broader-row selection rung."""

    if (
        not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or revision != _EXPECTED_MODEL_REVISION
        or model_id != DEFAULT_MODEL_ID
        or device_name != "cpu"
        or dtype != "float32"
    ):
        raise ValueError("A5c must replay the canonical pinned CPU float32 runtime")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A5c report")
    prepublication_bundle = default_a5c_prepublication_bundle_path(destination)
    if _continuation is None and (
        prepublication_bundle.exists() or prepublication_bundle.is_symlink()
    ):
        _progress("recover: publish surviving tensor-free prepublication bundle")
        return finalize_a5c_prepublication_bundle(
            prepublication_bundle, output=destination
        )

    a5b_file = Path(a5b_path)
    a5a_file = Path(a5a_path)
    a4_oracle_file = Path(a4_oracle_path)
    a4_file = Path(source_a4_report_path)
    fold_file = Path(fold_bundle_path)
    composition_file = Path(composition_bundle_path)

    _progress("preflight: authenticate immutable A5b failure and A5a/A4 sources")
    a5b_report, a5b_comparison = _authenticate_canonical_a5b_failure(a5b_file)
    a5a = load_a5_frozen_affine_capacity_report(a5a_file)
    if (
        _file_sha256(a5a_file) != _EXPECTED_A5A_FILE_SHA256
        or a5a.get("report_sha256") != _EXPECTED_A5A_REPORT_SHA256
        or a5a.get("conclusion", {}).get(
            "bounded_canary_resolves_affine_capacity"
        )
        is not True
    ):
        raise ValueError("A5c requires the canonical capacity-passing A5a result")
    a4_oracle, source_report, fold_bundle = _authenticate_a4_oracle_chain(
        a4_oracle_path=a4_oracle_file,
        a4_report_path=a4_file,
        fold_bundle_path=fold_file,
        composition_bundle_path=composition_file,
    )
    source_runtime = _required_mapping(
        source_report.get("runtime"), label="published A4 runtime"
    )
    source_authorization = _required_mapping(
        source_report.get("authorization"), label="published A4 authorization"
    )
    protocol = _required_mapping(
        source_report.get("protocol"), label="published A4 protocol"
    )
    if (
        source_runtime.get("model_id") != model_id
        or source_runtime.get("requested_revision") != revision
        or source_runtime.get("model_fingerprint") != _EXPECTED_MODEL_FINGERPRINT
        or source_runtime.get("device") != device_name
        or source_runtime.get("dtype") != dtype
    ):
        raise ValueError("A5c runtime must exactly replay published A4")

    bundle, authority, _, fit_authorization = _authenticate_before_fit_access(
        bundle_path=composition_file,
        corpus_receipt_path=corpus_receipt_path,
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
    )
    if (
        _canonical_json_bytes(source_authorization.get("bundle"))
        != _canonical_json_bytes(fit_authorization.get("bundle"))
        or _canonical_json_bytes(source_authorization.get("fit_authority"))
        != _canonical_json_bytes(fit_authorization.get("fit_authority"))
        or source_authorization.get("fit_authority_sha256")
        != fit_authorization.get("fit_authority_sha256")
    ):
        raise ValueError("live A-fit authority differs from published A4")
    bundle_binding = getattr(bundle, "binding", None)
    primary_graph = getattr(bundle, "primary", None)
    bundle_lowerings = getattr(bundle, "lowerings", None)
    if (
        not isinstance(bundle_binding, Mapping)
        or not isinstance(primary_graph, ModalGeneratorGraphPlan)
        or not isinstance(bundle_lowerings, tuple)
    ):
        raise TypeError("authenticated composition runtime is unavailable")
    layer10_graph, layer17_graph, layer10_lowerings, layer17_lowerings = (
        _source_lowering_maps(bundle)
    )
    _, fragment_by_node = _validate_source_decoder_contract(
        layer17_graph, layer17_lowerings, protocol
    )
    fragment_plans = {
        lowering.fragment_plan.artifact_sha256: lowering.fragment_plan
        for lowering in layer17_lowerings.values()
    }
    if len(fragment_plans) != 1:
        raise ValueError("Layer17 source lowerings use different fragment plans")
    fragment_selection: SameLayerFragmentSelection = (
        select_top_fisher_same_layer_fragments(
            next(iter(fragment_plans.values())),
            count=4,
            minimum_fragment_modes=32,
            layer_ordinal=17,
        )
    )
    _validate_frozen_selection(fragment_selection)
    if tuple(fragment_selection.fragment_ids) != tuple(fragment_by_node.values()):
        raise ValueError("Layer17 selected fragment order differs from A4")
    catalog = _build_source_runtime_catalog(
        bundle_binding=bundle_binding,
        primary_graph=primary_graph,
        layer10_graph=layer10_graph,
        layer17_graph=layer17_graph,
        layer10_lowerings_by_node=layer10_lowerings,
        layer17_lowerings_by_node=layer17_lowerings,
        selection=fragment_selection,
    )
    if catalog.get("catalog_sha256") != _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256:
        raise ValueError("live source runtime catalog is not frozen")
    _validate_source_runtime_catalog(
        catalog, protocol=protocol, bundle_binding=bundle_binding
    )
    if (
        _canonical_json_bytes(source_authorization.get("source_runtime_catalog"))
        != _canonical_json_bytes(catalog)
        or a4_oracle.get("source_bindings", {}).get(
            "source_runtime_catalog_sha256"
        )
        != catalog.get("catalog_sha256")
    ):
        raise ValueError("live source runtime catalog differs from A4 oracle")

    bases_by_node: dict[str, ComputationalModeBasis] = {
        name: layer17_lowerings[name].computational_mode_basis
        for name in layer17_graph.traversal_order
    }
    image = build_frozen_affine_image(
        bases_by_node, node_order=layer17_graph.traversal_order
    )
    randomness = source_runtime.get("randomness")
    inherited = (
        randomness.get("inherits_seed_and_execution_recipe_from")
        if isinstance(randomness, Mapping)
        else None
    )
    seed = inherited.get("torch_seed") if isinstance(inherited, Mapping) else None
    if type(seed) is not int:
        raise ValueError("published A4 deterministic seed is unavailable")
    torch.manual_seed(seed)
    device = resolve_torch_device(device_name)

    _progress("model: load pinned local Gemma checkpoint")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != _EXPECTED_MODEL_FINGERPRINT:
        raise ValueError("live Gemma fingerprint differs from published A4")

    _progress("tokenize: replay sealed family authority")
    raw_blocks, materialization = materialize_gemma3_layer17_family_lofo(
        authority, tokenizer
    )
    validate_gemma3_layer17_family_lofo_materialization_metadata(materialization)
    fit_collection = _required_mapping(
        source_report.get("fit_collection"), label="published A4 fit collection"
    )
    if _canonical_json_bytes(materialization) != _canonical_json_bytes(
        fit_collection.get("materialization")
    ):
        raise ValueError("live A-fit materialization differs from published A4")
    blocks = dict(_blocks_to_device(_family_blocks(raw_blocks), device))
    fold = _fold_catalog(protocol)[_OUTER_FOLD_INDEX]
    held_alias = str(fold["held_family_alias"])
    training_aliases = tuple(str(value) for value in fold["training_family_aliases"])
    if (
        len(training_aliases) != _TRAINING_FAMILY_COUNT
        or len(set(training_aliases)) != _TRAINING_FAMILY_COUNT
        or held_alias in training_aliases
    ):
        raise ValueError("A5c outer-fold family ownership drifted")

    training_batches: list[object] = []
    family_alias_by_example: dict[str, str] = {}
    training_example_ids: list[str] = []
    for alias in training_aliases:
        selected_batches = _take_first_examples(
            blocks[alias], _TRAINING_EXAMPLES_PER_FAMILY
        )
        training_batches.extend(selected_batches)
        for batch in selected_batches:
            example_ids = getattr(batch, "example_ids", None)
            if example_ids is None or len(example_ids) != 1:
                raise ValueError("A5c bounded training examples require identities")
            example_id = example_ids[0]
            if example_id in family_alias_by_example:
                raise ValueError("A5c training example identity is duplicated")
            family_alias_by_example[example_id] = alias
            training_example_ids.append(example_id)
    expected_examples = _TRAINING_FAMILY_COUNT * _TRAINING_EXAMPLES_PER_FAMILY
    if (
        len(training_batches) != expected_examples
        or len(training_example_ids) != expected_examples
        or held_alias in family_alias_by_example.values()
    ):
        raise RuntimeError("A5c broader training selection leaked the held family")

    capture_layer10_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer10_graph,
        tuple(
            layer10_lowerings[name] for name in layer10_graph.traversal_order
        ),
    )
    capture_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer17_graph,
        tuple(
            layer17_lowerings[name] for name in layer17_graph.traversal_order
        ),
    )
    source_lowering_by_name = {
        node.name: lowering
        for node, lowering in zip(primary_graph.nodes, bundle_lowerings, strict=True)
    }
    source_composition_lowerings = _ordered_restored_lowerings(
        primary_graph, source_lowering_by_name
    )

    _progress("capture: all valid rows from four examples in seven families")
    capture = capture_gemma3_layer17_full_block_closure(
        adapter,
        tuple(training_batches),
        selection=fragment_selection,
        leaf_activation_site=fragment_selection.execution_order[0].input_site,
        layer10_executor=capture_layer10_executor,
        layer17_executor=capture_layer17_executor,
    )
    capture_metadata = capture.metadata()
    capture_audit = _full_block_capture_audit_receipt(capture_metadata, protocol)
    if capture_audit.get("all_required_capture_audits_pass") is not True:
        raise RuntimeError("A5c broader capture audits failed")
    all_row_keys = tuple(capture.native_rows.row_keys)
    observed_examples = {example_id for example_id, _ in all_row_keys}
    if (
        not all_row_keys
        or observed_examples != set(training_example_ids)
        or capture.native_rows.observations != len(all_row_keys)
        or capture.compiled_rows.observations != len(all_row_keys)
    ):
        raise RuntimeError("A5c all-valid-row capture ownership drifted")
    fragment_ids = tuple(capture.trajectory_rows.fragment_ids)
    compiled_inputs = _shared_compiled_input(
        capture.compiled_rows, fragment_ids
    ).contiguous()
    native_state = capture.native_block_output.to(
        dtype=torch.float32
    ).contiguous()
    post_attention = capture.compiled_post_attention_residual.to(
        dtype=torch.float32
    ).contiguous()
    retained_delta = capture.compiled_compact_retained_post_feedforward_delta.to(
        dtype=torch.float32
    ).contiguous()
    frozen_state = capture.compiled_block_output.to(
        dtype=torch.float32
    ).contiguous()
    correction_base = (post_attention + retained_delta).contiguous()
    target = capture.a4_full_block_closure_target.contiguous()
    a4_projection = project_joint_target_to_frozen_bases(
        target,
        bases_by_node,
        node_order=layer17_graph.traversal_order,
    )
    a4_initial_correction = a4_projection.prediction.to(
        dtype=torch.float32
    ).contiguous()
    del a4_projection

    _progress("solve: all-row batched downstream coordinate targets")
    target_solution = solve_batched_frozen_affine_capacity_rows(
        adapter=adapter,
        image=image,
        native_state=native_state,
        compiled_post_attention_residual=post_attention,
        compiled_compact_retained_delta=retained_delta,
        target_correction=target,
        a4_float64_projection_correction=a4_initial_correction,
        row_chunk_size=_TARGET_BATCH_ROWS,
        token_locality_atol=_TOKEN_LOCALITY_ATOL,
        token_locality_rtol=_TOKEN_LOCALITY_RTOL,
        token_locality_policy=(
            TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW
        ),
        solver_config=DownstreamAffineSolverConfig(
            steps=_TARGET_SOLVER_STEPS,
            learning_rate=_TARGET_LEARNING_RATE_FRACTION,
            ridge=_TARGET_RIDGE,
            trust_radius=_TARGET_TRUST_RADIUS,
        ),
    )
    target_receipt = target_solution.receipt
    target_hashes = _required_mapping(
        target_receipt.get("hashes"), label="A5c target solver hashes"
    )
    target_report_receipt = {
        **target_receipt,
        "receipt_sha256": _sha256_value(_TARGET_RECEIPT_DOMAIN, target_receipt),
    }

    if (
        target_report_receipt.get("row_count") != len(all_row_keys)
        or target_report_receipt.get("row_chunk_size") != _TARGET_BATCH_ROWS
        or target_report_receipt.get("contains_tensor_payloads") is not False
    ):
        raise RuntimeError("A5c all-row target-solve accounting drifted")
    token_locality_lineage_sha256 = a5c_target_token_locality_lineage_sha256(
        target_report_receipt
    )

    row_catalog_sha256 = _sha256_value(_ROW_CATALOG_DOMAIN, all_row_keys)
    bridge_split_binding = _sha256_value(
        _BRIDGE_SPLIT_DOMAIN,
        {
            "a5a_report_sha256": a5a["report_sha256"],
            "protocol_fold_sha256": fold["artifact_sha256"],
            "capture_sha256": capture.capture_sha256,
            "all_valid_row_catalog_sha256": row_catalog_sha256,
            "configuration": {
                "examples_per_family": _TRAINING_EXAMPLES_PER_FAMILY,
                "row_selection_policy": "all_valid_captured_rows_per_example",
                "audit_examples_per_family": _INNER_AUDIT_EXAMPLES_PER_FAMILY,
            },
        },
    )
    fisher_by_node = {
        name: capture.native_rows.rows_by_fragment[
            fragment_by_node[name]
        ].fisher_weights.contiguous()
        for name in layer17_graph.traversal_order
    }
    bridge = build_a5b_downstream_coordinate_target_bridge(
        compiled_inputs=compiled_inputs,
        authenticated_compiled_inputs_sha256=a5b_tensor_sha256(compiled_inputs),
        selected_joint_coordinates=target_solution.selected_coefficients,
        authenticated_joint_coordinates_sha256=str(
            target_hashes["selected_coefficient_sha256"]
        ),
        bases_by_node=bases_by_node,
        node_order=layer17_graph.traversal_order,
        fisher_weights_by_node=fisher_by_node,
        fragment_id_by_node=fragment_by_node,
        row_keys=all_row_keys,
        family_alias_by_example=family_alias_by_example,
        training_family_aliases=training_aliases,
        held_family_alias=held_alias,
        inner_split_binding_sha256=bridge_split_binding,
        inner_audit_examples_per_family=_INNER_AUDIT_EXAMPLES_PER_FAMILY,
    )
    bridge_receipt = bridge.receipt()
    breadth_binding = _sha256_value(
        _BREADTH_SPLIT_DOMAIN,
        {
            "bridge_receipt_sha256": bridge.receipt_sha256,
            "capture_sha256": capture.capture_sha256,
            "all_valid_row_catalog_sha256": row_catalog_sha256,
            "examples_per_family": _TRAINING_EXAMPLES_PER_FAMILY,
            "audit_examples_per_family": _INNER_AUDIT_EXAMPLES_PER_FAMILY,
        },
    )
    breadth = build_a5c_breadth_split_from_bridge(
        bridge=bridge,
        split_binding_sha256=breadth_binding,
        audit_examples_per_family=_INNER_AUDIT_EXAMPLES_PER_FAMILY,
    )
    breadth_receipt = breadth.receipt()

    if _continuation is not None:
        # Keep the default A5c path byte-for-byte in its historical order:
        # these compact views are built early only for an explicit internal
        # continuation.  No A5c CV or held-family batch access has occurred.
        target_compact = _compact_target_solve_receipt(
            target_report_receipt
        )
        row_bank_compact = _compact_coordinate_row_bank(
            bridge=bridge, bridge_receipt=bridge_receipt
        )
        breadth_compact = _compact_breadth_split(breadth_receipt)
        capture_audit_sha256 = _sha256_value(
            _CAPTURE_AUDIT_DOMAIN, capture_audit
        )
        runtime_metadata = {
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": adapter.model_fingerprint(),
            "device": device_name,
            "dtype": dtype,
            "local_files_only": True,
        }
        source_bindings = {
            "a5b_file_sha256": _file_sha256(a5b_file),
            "a5b_report_sha256": _EXPECTED_A5B_REPORT_SHA256,
            **dict(
                _required_mapping(
                    a5b_report.get("source_bindings"),
                    label="canonical A5b source bindings",
                )
            ),
        }
        configuration_metadata = {
            "outer_fold_index": _OUTER_FOLD_INDEX,
            "training_family_count": _TRAINING_FAMILY_COUNT,
            "training_examples_per_family": _TRAINING_EXAMPLES_PER_FAMILY,
            "row_selection_policy": "all_valid_captured_rows_per_example",
            "inner_audit_examples_per_family": (
                _INNER_AUDIT_EXAMPLES_PER_FAMILY
            ),
            "target_solver_steps": _TARGET_SOLVER_STEPS,
            "target_batch_rows": _TARGET_BATCH_ROWS,
            "target_learning_rate_fraction": (
                _TARGET_LEARNING_RATE_FRACTION
            ),
            "target_ridge": _TARGET_RIDGE,
            "target_trust_radius": _TARGET_TRUST_RADIUS,
            "generator_rank": A5C_FIXED_GENERATOR_RANK,
            "ridge_grid": list(A5C_RIDGE_GRID),
            "held_examples_selected_or_scored": 0,
            "continuation_seam": "post_breadth_pre_cv_pre_held_access",
        }
        workspace = A5cBroaderTrainingWorkspace(
            adapter=adapter,
            model=model,
            tokenizer=tokenizer,
            device=device,
            blocks=blocks,
            held_family_alias=held_alias,
            training_family_aliases=training_aliases,
            training_example_ids=tuple(training_example_ids),
            family_alias_by_example=family_alias_by_example,
            fold=fold,
            materialization=materialization,
            source_composition_graph=primary_graph,
            source_composition_lowerings=source_composition_lowerings,
            layer10_graph=layer10_graph,
            layer17_graph=layer17_graph,
            layer10_lowerings_by_node=layer10_lowerings,
            layer17_lowerings_by_node=layer17_lowerings,
            bases_by_node=bases_by_node,
            fragment_id_by_node=fragment_by_node,
            fragment_selection=fragment_selection,
            frozen_affine_image=image,
            capture=capture,
            capture_metadata=capture_metadata,
            capture_audit=capture_audit,
            capture_sha256=capture.capture_sha256,
            capture_audit_sha256=capture_audit_sha256,
            all_row_keys=all_row_keys,
            row_catalog_sha256=row_catalog_sha256,
            compiled_inputs=compiled_inputs,
            native_block_states=native_state,
            frozen_compiled_block_states=frozen_state,
            compiled_correction_base_states=correction_base,
            compiled_post_attention_residual=post_attention,
            compiled_compact_retained_delta=retained_delta,
            a4_full_block_closure_target=target,
            target_solution=target_solution,
            target_receipt=target_receipt,
            target_report_receipt=target_report_receipt,
            target_compact=target_compact,
            target_hashes=target_hashes,
            target_token_locality_lineage_sha256=(
                token_locality_lineage_sha256
            ),
            bridge=bridge,
            bridge_receipt=bridge_receipt,
            coordinate_row_bank_compact=row_bank_compact,
            breadth=breadth,
            breadth_receipt=breadth_receipt,
            breadth_compact=breadth_compact,
            runtime_metadata=runtime_metadata,
            source_bindings=source_bindings,
            comparison_to_a5b=a5b_comparison,
            configuration_metadata=configuration_metadata,
            source_runtime_catalog=catalog,
            source_runtime_metadata=source_runtime,
            source_authorization=source_authorization,
            fit_authorization=fit_authorization,
            protocol=protocol,
            a5b_report=a5b_report,
            a5a_report=a5a,
            a4_oracle_report=a4_oracle,
            source_a4_report=source_report,
            fold_bundle=fold_bundle,
            output_path=destination,
        )
        return _dispatch_a5c_training_continuation(
            workspace, _continuation
        )

    _progress("select: seven family-disjoint ridge folds and final refit")
    cv_selection = select_a5c_family_disjoint_ridge(
        bridge=breadth,
        source_graph=layer17_graph,
        source_lowerings_by_node=layer17_lowerings,
        adapter=adapter,
        native_block_states=native_state,
        frozen_compiled_block_states=frozen_state,
        compiled_correction_base_states=correction_base,
        output_boundary=_POST_DELTA_BOUNDARY,
        ridge_grid=A5C_RIDGE_GRID,
        final_head_chunk_rows=_FINAL_HEAD_CHUNK_ROWS,
        final_head_token_locality_lineage_sha256=(
            token_locality_lineage_sha256
        ),
    )
    cv_receipt = cv_selection.receipt()

    target_compact = _compact_target_solve_receipt(target_report_receipt)
    row_bank_compact = _compact_coordinate_row_bank(
        bridge=bridge, bridge_receipt=bridge_receipt
    )
    breadth_compact = _compact_breadth_split(breadth_receipt)
    cv_compact = _compact_ridge_cv(cv_receipt)
    lineage = {
        "target_solve_receipt_sha256": target_compact["receipt_sha256"],
        "coordinate_row_bank_receipt_sha256": row_bank_compact[
            "receipt_sha256"
        ],
        "breadth_split_receipt_sha256": breadth_compact["receipt_sha256"],
        "ridge_cv_receipt_sha256": cv_compact["receipt_sha256"],
        "layer10_graph_sha256": layer10_graph.artifact_sha256,
        "layer10_lowering_sha256_by_node": {
            name: layer10_lowerings[name].artifact_sha256
            for name in layer10_graph.traversal_order
        },
    }
    executable = _freeze_selected_executable(
        selection=cv_selection,
        source_layer17_graph=layer17_graph,
        source_layer17_lowerings_by_node=layer17_lowerings,
        source_composition_graph=primary_graph,
        source_composition_lowerings=source_composition_lowerings,
        layer10_lowerings_by_node=layer10_lowerings,
        lineage=lineage,
    )
    scoring_executors = _build_scoring_executors(
        adapter=adapter,
        layer10_graph=layer10_graph,
        layer10_lowerings_by_node=layer10_lowerings,
        source_composition_graph=primary_graph,
        source_composition_lowerings=source_composition_lowerings,
        executable=executable,
    )

    # All sealed family blocks were materialized above, but this is the first
    # held-batch selection.  The helper re-authenticates the complete
    # executable freeze before selecting that batch.
    held_batches = _select_held_scoring_batch_after_freeze(
        blocks=blocks,
        held_family_alias=held_alias,
        executable=executable,
    )
    _progress("score: one untouched outer-family example through full logits")
    raw_evaluation = score_trajectory_correction_fold(
        adapter=adapter,
        layer10_executor=scoring_executors[0],
        corrected_layer17_executor=scoring_executors[1],
        frozen_composition_executor=scoring_executors[2],
        corrected_composition_executor=scoring_executors[3],
        batches=held_batches,
    )
    _validate_fold_evaluation(raw_evaluation, label="A5c outer fold zero")
    outer_evaluation = _adapt_outer_evaluation(
        raw_evaluation,
        outer_fold_index=_OUTER_FOLD_INDEX,
        layer10_graph_sha256=layer10_graph.artifact_sha256,
        executable=executable,
    )
    chronology = _chronology(
        ridge_cv_receipt_sha256=str(cv_compact["receipt_sha256"]),
        selection_freeze_sha256=executable.selection_freeze_sha256,
        outer_evaluation=outer_evaluation,
    )

    report_inputs = {
        "source_bindings": {
            "a5b_file_sha256": _file_sha256(a5b_file),
            "a5b_report_sha256": _EXPECTED_A5B_REPORT_SHA256,
            **dict(
                _required_mapping(
                    a5b_report.get("source_bindings"),
                    label="canonical A5b source bindings",
                )
            ),
        },
        "runtime": {
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": adapter.model_fingerprint(),
            "device": device_name,
            "dtype": dtype,
            "local_files_only": True,
        },
        "configuration": {
            "outer_fold_index": _OUTER_FOLD_INDEX,
            "training_family_count": _TRAINING_FAMILY_COUNT,
            "training_examples_per_family": _TRAINING_EXAMPLES_PER_FAMILY,
            "row_selection_policy": "all_valid_captured_rows_per_example",
            "inner_audit_examples_per_family": _INNER_AUDIT_EXAMPLES_PER_FAMILY,
            "target_solver_steps": _TARGET_SOLVER_STEPS,
            "target_batch_rows": _TARGET_BATCH_ROWS,
            "target_learning_rate_fraction": _TARGET_LEARNING_RATE_FRACTION,
            "target_ridge": _TARGET_RIDGE,
            "target_trust_radius": _TARGET_TRUST_RADIUS,
            "generator_rank": A5C_FIXED_GENERATOR_RANK,
            "ridge_grid": list(A5C_RIDGE_GRID),
            "held_examples_scored": 1,
        },
        "capture": {
            "capture_sha256": capture.capture_sha256,
            "capture_audit_sha256": _sha256_value(
                _CAPTURE_AUDIT_DOMAIN, capture_audit
            ),
            "source_row_catalog_sha256": capture.native_rows.row_key_sha256,
            "training_family_count": _TRAINING_FAMILY_COUNT,
            "training_example_count": len(training_example_ids),
            "captured_observation_count": len(all_row_keys),
            "selected_target_row_count": len(all_row_keys),
            "outer_held_family_rows_present": False,
            "all_required_capture_audits_pass": True,
        },
        "target_solve": target_compact,
        "coordinate_row_bank": row_bank_compact,
        "breadth_split": breadth_compact,
        "ridge_cv": cv_compact,
        "evidence_receipts": {
            "target_solve": target_report_receipt,
            "coordinate_row_bank": bridge_receipt,
            "breadth_split": breadth_receipt,
            "ridge_cv": cv_receipt,
        },
        "selected_executable": executable.report_section(),
        "chronology": chronology,
        "outer_evaluation": outer_evaluation,
        "comparison_to_a5b": a5b_comparison,
    }
    saved = publish_a5c_report_with_prepublication_bundle(
        output=destination,
        report_inputs=report_inputs,
    )
    _progress(f"published: {destination}")
    del capture, target_solution, bridge, breadth, cv_selection
    gc.collect()
    return saved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT
    )
    parser.add_argument(
        "--a5b-path",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_OUTPUT,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_l10_l17_a5c_broader_selected_generator(
        revision=args.revision,
        output=args.output,
        a5b_path=args.a5b_path,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
