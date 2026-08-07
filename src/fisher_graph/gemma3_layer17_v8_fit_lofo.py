"""Family-disjoint Calibration-A refit diagnostic for Gemma layer 17.

This runner asks whether the fixed cap-48/rank-16 edgeless layer-17 modal
topology transfers once every fitted quantity is estimated out of family.
It opens only the authenticated v8 ``calibration_a_fit`` authority, captures
the four native fragment streams once, and then performs eight deterministic
train-seven/hold-one refits.  The held family cannot affect the contribution
center, Fisher normalization, computational-mode basis, or modal generator.

The result is deliberately diagnostic.  It serializes scalar fidelity,
resource accounting, and hash receipts only.  It does not serialize fold
weights, authorize serving, choose a new hyperparameter, or open selection,
guard, Calibration-B, validation, or test data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import MappingProxyType

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_capped_node_fit import (
    _validate_frozen_selection,
    default_gemma3_layer17_capped_node_output,
    fit_layer17_capped_node_pilots,
    load_gemma3_layer17_capped_node_candidate,
    restore_gemma3_layer17_capped_node_runtime,
)
from .gemma3_layer17_family_lofo_authority import (
    Gemma3Layer17FamilyLOFOAuthority,
    load_gemma3_layer17_family_lofo_authority,
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_authority_metadata,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    V8_FAMILY_LOFO_FAMILY_ALIASES,
    build_authenticated_v8_layer17_family_lofo_protocol,
    build_default_v8_layer17_family_lofo_protocol,
    validate_v8_layer17_family_lofo_protocol,
)
from .gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_NATIVE_MODE_COUNTS,
    LAYER17_TOPOLOGY_SHA256,
    Layer17NodeRankResourceRow,
    build_layer17_node_rank_resource_row,
)
from .gemma3_modal_generator_dev_experiment import (
    load_gemma3_modal_generator_dev_artifact,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    DEFAULT_BASE_ARTIFACT,
    _restore_upstream_analysis,
    _validate_upstream_bindings,
)
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows
from .gemma3_same_layer_shape_flow import (
    build_edgeless_same_layer_graph,
    select_top_fisher_same_layer_fragments,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _collect_same_layer_native_rows,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_graph_rung_evaluation import (
    _GRAPH_LOGICAL_FIELDS,
    _GRAPH_STATIC_FIELDS,
    _candidate_comparison,
    _execution_fields,
    _model_logits,
    _native_nll,
    _selected_logits_and_targets,
    _validate_graph_execution,
)
from .gemma3_modal_generator_dev_experiment import LayerFragmentRows


__all__ = [
    "DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT",
    "DEFAULT_MODE_RANK_CAP",
    "GEMMA3_LAYER17_V8_FIT_LOFO_FORMAT_VERSION",
    "GEMMA3_LAYER17_V8_FIT_LOFO_SCHEMA",
    "Layer17FamilyFold",
    "aggregate_layer17_lofo_fold_metrics",
    "build_layer17_family_folds",
    "build_layer17_v8_fit_lofo_report",
    "build_parser",
    "evaluate_layer17_lofo_protocol_gates",
    "evaluate_layer17_lofo_fold",
    "load_gemma3_layer17_v8_fit_lofo_report",
    "make_layer17_lofo_fold_rows",
    "partition_aligned_fragment_rows_by_family",
    "run_gemma3_layer17_v8_fit_lofo",
    "save_gemma3_layer17_v8_fit_lofo_report",
    "validate_gemma3_layer17_v8_fit_lofo_report",
]


GEMMA3_LAYER17_V8_FIT_LOFO_SCHEMA = (
    "fisher_graph.gemma3_layer17_v8_fit_family_lofo"
)
GEMMA3_LAYER17_V8_FIT_LOFO_FORMAT_VERSION = 1
DEFAULT_MODE_RANK_CAP = 48
_GENERATOR_RANK = 16
_RIDGE = 0.0
_EXPECTED_FAMILIES = 8
_EXPECTED_EXAMPLES = 256
_VOCABULARY_CHUNK_SIZE = 16384
_CONDITIONS = ("lofo_refit", "frozen_v9_cap48", "matched_deletion")
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT = (
    _LOCAL_ROOT / "layer17-v8-fit-lofo-cap48-r16-edgeless-v1.json"
)
DEFAULT_FROZEN_CAP48_CANDIDATE = default_gemma3_layer17_capped_node_output(
    DEFAULT_MODE_RANK_CAP,
    _GENERATOR_RANK,
)
_LAYER17_NODE_NAMES = tuple(
    "gemma3.layer-17."
    f"cluster-{fragment_id.split('/', 1)[0].split('.', 1)[1]}."
    f"modal-generator.same-layer-{index}.graph-node"
    for index, fragment_id in enumerate(LAYER17_FRAGMENT_IDS)
)

_REPORT_DOMAIN = b"fisher-graph:gemma3-layer17-v8-fit-lofo:report:v1\0"
_FOLD_DOMAIN = b"fisher-graph:gemma3-layer17-v8-fit-lofo:fold:v1\0"
_SPLIT_DOMAIN = b"fisher-graph:gemma3-layer17-v8-fit-lofo:split:v1\0"
_FISHER_DOMAIN = b"fisher-graph:gemma3-layer17-v8-fit-lofo:fisher:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALIAS = re.compile(r"^family_[0-9]{2}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_FISHER_NORMALIZATION = (
    "raw_empirical_fisher_normalized_to_equal_total_mass_per_training_family"
)


def _progress(message: str) -> None:
    print(f"[layer17-v8-fit-lofo] {message}", file=sys.stderr, flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise TypeError("tensor hash input must be a strided Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_FISHER_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "shape": tuple(int(width) for width in canonical.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_aliases(
    values: Sequence[str],
    *,
    expected: int | None = None,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("family aliases must be a sequence")
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, str) or _ALIAS.fullmatch(value) is None for value in result)
        or result != tuple(sorted(set(result)))
        or (expected is not None and len(result) != expected)
    ):
        raise ValueError("family aliases must be unique, sorted family_NN values")
    return result


@dataclass(frozen=True, slots=True)
class Layer17FamilyFold:
    """One deterministic train-seven/hold-one family ownership receipt."""

    held_family_alias: str
    training_family_aliases: tuple[str, ...]
    protocol_fold_sha256: str
    held_membership_sha256: str
    training_membership_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if _ALIAS.fullmatch(self.held_family_alias) is None:
            raise ValueError("held_family_alias must be family_NN")
        training = _canonical_aliases(
            self.training_family_aliases,
            expected=_EXPECTED_FAMILIES - 1,
        )
        if self.held_family_alias in training:
            raise ValueError("held family cannot appear in training families")
        if set((*training, self.held_family_alias)) != set(
            V8_FAMILY_LOFO_FAMILY_ALIASES
        ):
            raise ValueError("fold does not cover the frozen eight aliases")
        for label, value in (
            ("protocol fold", self.protocol_fold_sha256),
            ("held membership", self.held_membership_sha256),
            ("training membership", self.training_membership_sha256),
        ):
            _require_sha256(value, label=label)
        expected = _sha256(self._payload(), domain=_FOLD_DOMAIN)
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="fold artifact_sha256",
            ) != expected:
                raise ValueError("family-fold hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    def _payload(self) -> dict[str, object]:
        return {
            "held_family_alias": self.held_family_alias,
            "training_family_aliases": self.training_family_aliases,
            "training_family_count": len(self.training_family_aliases),
            "held_family_excluded": True,
            "protocol_fold_sha256": self.protocol_fold_sha256,
            "held_membership_sha256": self.held_membership_sha256,
            "training_membership_sha256": self.training_membership_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def build_layer17_family_folds(
    family_aliases: Sequence[str],
    *,
    protocol: Mapping[str, object] | None = None,
) -> tuple[Layer17FamilyFold, ...]:
    """Restore the exact eight folds from the authenticated frozen protocol."""

    aliases = _canonical_aliases(
        family_aliases,
        expected=_EXPECTED_FAMILIES,
    )
    frozen = validate_v8_layer17_family_lofo_protocol(
        build_default_v8_layer17_family_lofo_protocol()
        if protocol is None
        else protocol
    )
    if frozen["artifact_sha256"] != (
        FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
    ):
        raise ValueError("family folds do not bind the frozen protocol")
    raw_folds = frozen.get("folds")
    if not isinstance(raw_folds, Sequence) or len(raw_folds) != len(aliases):
        raise ValueError("frozen protocol fold catalog is invalid")
    result: list[Layer17FamilyFold] = []
    for index, raw in enumerate(raw_folds):
        if not isinstance(raw, Mapping):
            raise TypeError("frozen protocol fold must be a mapping")
        held = aliases[index]
        training = tuple(alias for alias in aliases if alias != held)
        if (
            raw.get("fold_index") != index
            or raw.get("held_family_alias") != held
            or tuple(raw.get("training_family_aliases", ())) != training
            or raw.get("held_example_count") != 32
            or raw.get("training_example_count") != 224
        ):
            raise ValueError("runtime aliases differ from frozen protocol folds")
        result.append(
            Layer17FamilyFold(
                held_family_alias=held,
                training_family_aliases=training,
                protocol_fold_sha256=_require_sha256(
                    raw.get("artifact_sha256"),
                    label="protocol fold",
                ),
                held_membership_sha256=_require_sha256(
                    raw.get("held_membership_sha256"),
                    label="held membership",
                ),
                training_membership_sha256=_require_sha256(
                    raw.get("training_membership_sha256"),
                    label="training membership",
                ),
            )
        )
    return tuple(result)


def _selected_rows(
    rows: AlignedFragmentRows,
    indices: Tensor,
) -> AlignedFragmentRows:
    if (
        not isinstance(indices, Tensor)
        or indices.dtype != torch.long
        or indices.ndim != 1
        or indices.numel() <= 0
    ):
        raise ValueError("aligned-row selection requires nonempty long indices")
    selected_keys = tuple(rows.row_keys[int(index)] for index in indices.tolist())
    sequences = len({example_id for example_id, _ in selected_keys})
    return AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=fragment_rows.inputs.index_select(0, indices),
                contributions=fragment_rows.contributions.index_select(0, indices),
                fisher_weights=fragment_rows.fisher_weights.index_select(0, indices),
                sequences=sequences,
            )
            for fragment_id, fragment_rows in rows.rows_by_fragment.items()
        },
        row_keys=selected_keys,
    )


def partition_aligned_fragment_rows_by_family(
    rows: AlignedFragmentRows,
    family_alias_by_example: Mapping[str, str],
) -> Mapping[str, AlignedFragmentRows]:
    """Partition one captured aligned row set without recomputing the model."""

    if not isinstance(rows, AlignedFragmentRows):
        raise TypeError("rows must be AlignedFragmentRows")
    if not isinstance(family_alias_by_example, Mapping):
        raise TypeError("family_alias_by_example must be a mapping")
    examples = {example_id for example_id, _ in rows.row_keys}
    if set(family_alias_by_example) != examples:
        raise ValueError("family ownership must exactly cover captured examples")
    aliases = _canonical_aliases(
        tuple(sorted(set(family_alias_by_example.values()))),
        expected=_EXPECTED_FAMILIES,
    )
    partitions: dict[str, AlignedFragmentRows] = {}
    covered: list[tuple[str, int]] = []
    for alias in aliases:
        indices = torch.tensor(
            [
                index
                for index, (example_id, _) in enumerate(rows.row_keys)
                if family_alias_by_example[example_id] == alias
            ],
            dtype=torch.long,
        )
        partitions[alias] = _selected_rows(rows, indices)
        covered.extend(partitions[alias].row_keys)
    if set(covered) != set(rows.row_keys) or len(covered) != len(rows.row_keys):
        raise RuntimeError("family row partition lost or duplicated rows")
    return MappingProxyType(partitions)


def _equal_family_fisher_concat(
    family_rows: Mapping[str, AlignedFragmentRows],
    aliases: Sequence[str],
) -> AlignedFragmentRows:
    selected = _canonical_aliases(tuple(aliases))
    if not isinstance(family_rows, Mapping) or not set(selected).issubset(family_rows):
        raise ValueError("selected aliases are absent from family rows")
    fragment_ids = tuple(family_rows[selected[0]].rows_by_fragment)
    if set(fragment_ids) != set(LAYER17_FRAGMENT_IDS):
        raise ValueError("family rows do not cover the frozen layer-17 fragments")
    row_keys = tuple(
        key for alias in selected for key in family_rows[alias].row_keys
    )
    sequences = sum(family_rows[alias].sequences for alias in selected)
    normalized: dict[str, LayerFragmentRows] = {}
    for fragment_id in fragment_ids:
        input_parts: list[Tensor] = []
        contribution_parts: list[Tensor] = []
        weight_parts: list[Tensor] = []
        for alias in selected:
            fragment = family_rows[alias].rows_by_fragment[fragment_id]
            total = fragment.fisher_weights.sum()
            if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
                raise ValueError(
                    f"{alias}/{fragment_id} has no positive Fisher mass"
                )
            input_parts.append(fragment.inputs)
            contribution_parts.append(fragment.contributions)
            weight_parts.append(
                fragment.fisher_weights / total / len(selected)
            )
        weights = torch.cat(weight_parts, dim=0)
        if not torch.allclose(
            weights.sum(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=0.0,
            atol=2e-12,
        ):
            raise RuntimeError("fold Fisher weights failed unit normalization")
        normalized[fragment_id] = LayerFragmentRows(
            inputs=torch.cat(input_parts, dim=0),
            contributions=torch.cat(contribution_parts, dim=0),
            fisher_weights=weights,
            sequences=sequences,
        )
    return AlignedFragmentRows(
        rows_by_fragment=normalized,
        row_keys=row_keys,
    )


def _split_sha256(
    *,
    authority_sha256: str,
    fold: Layer17FamilyFold,
    role: str,
    rows: AlignedFragmentRows,
    family_aliases: Sequence[str],
) -> str:
    aliases = _canonical_aliases(tuple(family_aliases))
    if role not in ("fit", "held"):
        raise ValueError("fold split role must be fit or held")
    return _sha256(
        {
            "authority_sha256": _require_sha256(
                authority_sha256,
                label="authority_sha256",
            ),
            "fold_sha256": fold.artifact_sha256,
            "role": role,
            "family_aliases": aliases,
            "row_key_sha256": rows.row_key_sha256,
            "observations": rows.observations,
            "sequences": rows.sequences,
            "fisher_normalization": _FISHER_NORMALIZATION,
            "fisher_weight_sha256_by_fragment": {
                fragment_id: _tensor_sha256(fragment.fisher_weights)
                for fragment_id, fragment in sorted(
                    rows.rows_by_fragment.items()
                )
            },
        },
        domain=_SPLIT_DOMAIN,
    )


def make_layer17_lofo_fold_rows(
    family_rows: Mapping[str, AlignedFragmentRows],
    fold: Layer17FamilyFold,
    *,
    authority_sha256: str,
) -> tuple[AlignedFragmentRows, AlignedFragmentRows, dict[str, object]]:
    """Construct normalized train-seven and held-one rows for one fold."""

    if not isinstance(fold, Layer17FamilyFold):
        raise TypeError("fold must be Layer17FamilyFold")
    train = _equal_family_fisher_concat(
        family_rows,
        fold.training_family_aliases,
    )
    held = _equal_family_fisher_concat(
        family_rows,
        (fold.held_family_alias,),
    )
    if set(train.row_keys) & set(held.row_keys):
        raise RuntimeError("LOFO fit and held row keys overlap")
    if set(train.row_keys) | set(held.row_keys) != {
        key for value in family_rows.values() for key in value.row_keys
    }:
        raise RuntimeError("LOFO fit and held rows do not cover the panel")
    fit_sha256 = _split_sha256(
        authority_sha256=authority_sha256,
        fold=fold,
        role="fit",
        rows=train,
        family_aliases=fold.training_family_aliases,
    )
    held_sha256 = _split_sha256(
        authority_sha256=authority_sha256,
        fold=fold,
        role="held",
        rows=held,
        family_aliases=(fold.held_family_alias,),
    )
    if fit_sha256 == held_sha256:
        raise RuntimeError("LOFO fit and held split bindings collide")
    receipt = {
        **fold.metadata(),
        "fit_split_sha256": fit_sha256,
        "held_split_sha256": held_sha256,
        "fit_observations": train.observations,
        "held_observations": held.observations,
        "fit_sequences": train.sequences,
        "held_sequences": held.sequences,
        "fisher_normalization": _FISHER_NORMALIZATION,
        "fit_fisher_total_mass_per_training_family": 1.0
        / (_EXPECTED_FAMILIES - 1),
        "held_fisher_total_mass": 1.0,
        "held_family_excluded_from_center_basis_fisher_and_generator": True,
        "held_rows_used_only_for_fixed_rank_descriptive_metrics": True,
    }
    return train, held, receipt


def _new_metric_totals() -> dict[str, object]:
    return {
        "supervised_tokens": 0,
        "native_nll_sum": 0.0,
        "conditions": {
            name: {
                "nll_sum": 0.0,
                "native_to_candidate_kl_sum": 0.0,
                "top1_matches": 0,
            }
            for name in _CONDITIONS
        },
    }


def _finish_metric_totals(totals: Mapping[str, object]) -> dict[str, object]:
    tokens = totals.get("supervised_tokens")
    raw_conditions = totals.get("conditions")
    if type(tokens) is not int or tokens <= 0 or not isinstance(
        raw_conditions, Mapping
    ):
        raise ValueError("LOFO metric totals are incomplete")
    native_nll = float(totals["native_nll_sum"]) / tokens
    conditions: dict[str, object] = {}
    for name in _CONDITIONS:
        raw = raw_conditions[name]
        if not isinstance(raw, Mapping):
            raise TypeError("LOFO condition totals are invalid")
        nll = float(raw["nll_sum"]) / tokens
        conditions[name] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": (
                float(raw["native_to_candidate_kl_sum"]) / tokens
            ),
            "top1_agreement_to_native": int(raw["top1_matches"]) / tokens,
        }
    return {
        "supervised_tokens": tokens,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
    }


def _graph_node_topology_signature(plan: object) -> tuple[tuple[object, ...], ...]:
    nodes = getattr(plan, "nodes", None)
    if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Sequence):
        raise TypeError("modal graph plan nodes must be a sequence")
    return tuple(
        (
            node.name,
            node.input_boundary,
            node.output_boundary,
            node.input_width,
            node.latent_width,
            node.output_width,
        )
        for node in nodes
    )


def evaluate_layer17_lofo_fold(
    adapter: Gemma3CausalLMAdapter,
    lofo_executor: Gemma3ModalGeneratorGraphExecutor,
    frozen_executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    """Score one held family with one native and one matched deletion path."""

    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(batch, CalibrationBatch) for batch in materialized
    ):
        raise ValueError("held batches must contain CalibrationBatch values")
    expected_ids = tuple(
        example_id
        for batch in materialized
        for example_id in (
            batch.example_ids if batch.example_ids is not None else ()
        )
    )
    if (
        any(batch.example_ids is None for batch in materialized)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise ValueError("held batches require unique example identities")
    lofo_plan = lofo_executor.graph_plan
    frozen_plan = frozen_executor.graph_plan
    if lofo_plan.interactions or frozen_plan.interactions:
        raise ValueError("LOFO comparison requires edgeless graph plans")
    if (
        lofo_plan.model_fingerprint != frozen_plan.model_fingerprint
        or lofo_plan.parameter_cluster_plan_sha256
        != frozen_plan.parameter_cluster_plan_sha256
        or _graph_node_topology_signature(lofo_plan)
        != _graph_node_topology_signature(frozen_plan)
    ):
        raise ValueError("LOFO and frozen graph physical topologies differ")

    totals = _new_metric_totals()
    logical_valid_tokens = 0
    static_resources: dict[str, object] | None = None
    logical_totals = {name: 0 for name in _GRAPH_LOGICAL_FIELDS}
    native_model = adapter.module
    with ExitStack() as stack:
        stack.enter_context(lofo_executor.validated_transaction())
        stack.enter_context(frozen_executor.validated_transaction())
        for batch in materialized:
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = native_model(**call_inputs)
                lofo_execution = lofo_executor.run(
                    batch.model_inputs,
                    condition="generated",
                )
                frozen_execution = frozen_executor.run(
                    batch.model_inputs,
                    condition="generated",
                )
                deletion_execution = lofo_executor.run(
                    batch.model_inputs,
                    condition="deletion",
                )
            _validate_graph_execution(
                lofo_execution,
                lofo_plan,
                condition="generated",
                label="LOFO refit",
            )
            _validate_graph_execution(
                frozen_execution,
                frozen_plan,
                condition="generated",
                label="frozen v9 cap48",
            )
            _validate_graph_execution(
                deletion_execution,
                lofo_plan,
                condition="deletion",
                label="matched deletion",
            )
            current_static = _execution_fields(
                lofo_execution,
                _GRAPH_STATIC_FIELDS,
                label="LOFO refit",
            )
            if current_static != _execution_fields(
                frozen_execution,
                _GRAPH_STATIC_FIELDS,
                label="frozen v9 cap48",
            ) or current_static != _execution_fields(
                deletion_execution,
                _GRAPH_STATIC_FIELDS,
                label="matched deletion",
            ):
                raise RuntimeError("LOFO condition resource scopes differ")
            if static_resources is None:
                static_resources = current_static
            elif static_resources != current_static:
                raise RuntimeError("LOFO static resources changed by batch")
            if (
                deletion_execution.logical_executed_modal_graph_macs != 0
                or deletion_execution.logical_executed_modal_graph_additions
                != 0
                or deletion_execution.peak_live_modal_width != 0
            ):
                raise RuntimeError("matched deletion executed modal graph work")

            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output),
                batch,
            )
            candidates = {
                "lofo_refit": _selected_logits_and_targets(
                    _model_logits(lofo_execution.model_output), batch
                ),
                "frozen_v9_cap48": _selected_logits_and_targets(
                    _model_logits(frozen_execution.model_output), batch
                ),
                "matched_deletion": _selected_logits_and_targets(
                    _model_logits(deletion_execution.model_output), batch
                ),
            }
            token_count = targets.numel()
            totals["supervised_tokens"] = int(totals["supervised_tokens"]) + token_count
            totals["native_nll_sum"] = float(totals["native_nll_sum"]) + _native_nll(
                native_logits,
                targets,
            )
            condition_totals = totals["conditions"]
            assert isinstance(condition_totals, dict)
            for name, (candidate_logits, candidate_targets) in candidates.items():
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{name} held targets drifted")
                comparison = _candidate_comparison(
                    native_logits,
                    candidate_logits,
                    targets,
                    vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                )
                accumulator = condition_totals[name]
                for metric, value in comparison.items():
                    accumulator[metric] += value
            logical_valid_tokens += lofo_execution.valid_tokens
            for name in _GRAPH_LOGICAL_FIELDS:
                logical_totals[name] += int(getattr(lofo_execution, name))

    if static_resources is None:
        raise RuntimeError("held-family evaluation produced no resources")
    metrics = _finish_metric_totals(totals)
    return {
        "execution_path": "paired_layer17_lofo_and_frozen_edgeless_graphs",
        **metrics,
        "logical_valid_tokens": logical_valid_tokens,
        "graph": {
            "node_count": len(lofo_plan.nodes),
            "interaction_count": 0,
            "traversal_order": lofo_plan.traversal_order,
        },
        "resource_accounting": {
            **static_resources,
            "lofo_generated": logical_totals,
            "matched_deletion_executed_graph_macs": 0,
            "latency_or_kernel_speed_claim": False,
        },
    }


def _condition_summary(
    fold_metrics: Sequence[Mapping[str, object]],
    condition: str,
) -> dict[str, object]:
    rows = [value["conditions"][condition] for value in fold_metrics]  # type: ignore[index]
    metrics = (
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    )
    macro: dict[str, float] = {}
    median: dict[str, float] = {}
    worst: dict[str, float] = {}
    for metric in metrics:
        values = sorted(float(row[metric]) for row in rows)  # type: ignore[index]
        macro[metric] = math.fsum(values) / len(values)
        middle = len(values) // 2
        median[metric] = (values[middle - 1] + values[middle]) / 2.0
        worst[metric] = (
            min(values)
            if metric == "top1_agreement_to_native"
            else max(values)
        )
    return {"macro": macro, "median": median, "worst": worst}


def aggregate_layer17_lofo_fold_metrics(
    fold_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate eight held-family rows as micro, macro, median, and worst."""

    values = tuple(fold_metrics)
    if len(values) != _EXPECTED_FAMILIES:
        raise ValueError("LOFO aggregation requires exactly eight folds")
    totals = _new_metric_totals()
    for value in values:
        tokens = value.get("supervised_tokens")
        native = value.get("native")
        conditions = value.get("conditions")
        if (
            type(tokens) is not int
            or tokens <= 0
            or not isinstance(native, Mapping)
            or not isinstance(conditions, Mapping)
            or set(conditions) != set(_CONDITIONS)
        ):
            raise ValueError("fold metrics are incomplete")
        totals["supervised_tokens"] = int(totals["supervised_tokens"]) + tokens
        totals["native_nll_sum"] = float(totals["native_nll_sum"]) + (
            float(native["nll_per_token"]) * tokens
        )
        aggregate_conditions = totals["conditions"]
        assert isinstance(aggregate_conditions, dict)
        for name in _CONDITIONS:
            row = conditions[name]
            if not isinstance(row, Mapping):
                raise TypeError("fold condition metrics are invalid")
            aggregate_conditions[name]["nll_sum"] += float(row["nll_per_token"]) * tokens
            aggregate_conditions[name]["native_to_candidate_kl_sum"] += (
                float(row["native_to_candidate_kl_per_token"]) * tokens
            )
            aggregate_conditions[name]["top1_matches"] += round(
                float(row["top1_agreement_to_native"]) * tokens
            )
    micro = _finish_metric_totals(totals)
    native_values = sorted(
        float(value["native"]["nll_per_token"])  # type: ignore[index]
        for value in values
    )
    return {
        "micro": micro,
        "equal_family": {
            "native": {
                "macro_nll_per_token": math.fsum(native_values) / len(native_values),
                "median_nll_per_token": (
                    native_values[3] + native_values[4]
                )
                / 2.0,
                "worst_nll_per_token": max(native_values),
            },
            "conditions": {
                name: _condition_summary(values, name) for name in _CONDITIONS
            },
        },
    }


def _gate_row(
    gate_id: str,
    *,
    observed: int | float | bool | None,
    operator: str,
    threshold: int | float | bool,
) -> dict[str, object]:
    if observed is None:
        passed = False
    elif operator == "<=":
        passed = float(observed) <= float(threshold)
    elif operator == ">=":
        passed = float(observed) >= float(threshold)
    elif operator == "==":
        passed = observed == threshold
    else:
        raise ValueError("unsupported LOFO gate operator")
    return {
        "gate_id": gate_id,
        "required": True,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
    }


def _deletion_recovery_by_fold(
    fold_metrics: Sequence[Mapping[str, object]],
) -> tuple[float | None, ...]:
    result: list[float | None] = []
    for value in fold_metrics:
        conditions = value.get("conditions")
        if not isinstance(conditions, Mapping):
            raise TypeError("fold conditions are unavailable")
        generated = conditions.get("lofo_refit")
        deletion = conditions.get("matched_deletion")
        if not isinstance(generated, Mapping) or not isinstance(
            deletion, Mapping
        ):
            raise TypeError("fold generated/deletion metrics are unavailable")
        generated_delta = float(generated["delta_nll_per_token"])
        deletion_delta = float(deletion["delta_nll_per_token"])
        if not math.isfinite(generated_delta) or not math.isfinite(
            deletion_delta
        ):
            raise ValueError("fold deletion-recovery inputs must be finite")
        result.append(
            None
            if deletion_delta <= 0.0
            else (deletion_delta - generated_delta) / deletion_delta
        )
    return tuple(result)


def evaluate_layer17_lofo_protocol_gates(
    *,
    protocol: Mapping[str, object],
    fold_metrics: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    """Apply only the exact thresholds frozen before the LOFO run."""

    frozen = validate_v8_layer17_family_lofo_protocol(protocol)
    metric_rows = tuple(
        _validate_fold_evaluation(value, label=f"gate fold {index}")
        for index, value in enumerate(fold_metrics)
    )
    recomputed_aggregate = aggregate_layer17_lofo_fold_metrics(metric_rows)
    if _canonical_json_bytes(aggregate) != _canonical_json_bytes(
        recomputed_aggregate
    ):
        raise ValueError(
            "supplied LOFO aggregate differs from the exact fold metrics"
        )
    if not isinstance(resources, Mapping):
        raise TypeError("LOFO gate resources must be a mapping")
    restored_resources = Layer17NodeRankResourceRow.from_state_dict(resources)
    expected_resources = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=DEFAULT_MODE_RANK_CAP,
        generator_rank=_GENERATOR_RANK,
        edge_policy="edgeless",
    )
    if _canonical_json_bytes(
        restored_resources.state_dict()
    ) != _canonical_json_bytes(expected_resources.state_dict()):
        raise ValueError("LOFO gate resources differ from fixed cap48/r16/edgeless")
    resources = restored_resources.state_dict()
    static_reference: dict[str, object] | None = None
    additions_reference: int | None = None
    for index, row in enumerate(metric_rows):
        static, additions = _validate_fixed_arm_fold_resources(
            row,
            restored_resources,
            label=f"gate fold {index}",
        )
        if static_reference is None:
            static_reference = static
        elif static != static_reference:
            raise ValueError("LOFO gate fold static resources differ")
        if additions_reference is None:
            additions_reference = additions
        elif additions != additions_reference:
            raise ValueError("LOFO gate fold additions per token differ")
    gates = frozen.get("gates")
    evaluation_contract = frozen.get("evaluation_contract")
    equal_family = recomputed_aggregate.get("equal_family")
    if (
        not isinstance(gates, Mapping)
        or not isinstance(evaluation_contract, Mapping)
        or not isinstance(equal_family, Mapping)
    ):
        raise TypeError("LOFO gates or equal-family aggregate are unavailable")
    recovery_contract = evaluation_contract.get("deletion_nll_recovery")
    if not isinstance(recovery_contract, Mapping):
        raise TypeError("frozen deletion-recovery contract is unavailable")
    conditions = equal_family.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError("equal-family condition metrics are unavailable")
    generated = conditions.get("lofo_refit")
    if not isinstance(generated, Mapping):
        raise TypeError("LOFO refit aggregate is unavailable")
    macro = generated.get("macro")
    worst = generated.get("worst")
    if not isinstance(macro, Mapping) or not isinstance(worst, Mapping):
        raise TypeError("LOFO macro/worst metrics are unavailable")
    recovery = _deletion_recovery_by_fold(metric_rows)
    valid_recovery = tuple(value for value in recovery if value is not None)
    invalid_recovery_count = len(recovery) - len(valid_recovery)
    recovery_macro = (
        None
        if invalid_recovery_count
        else math.fsum(valid_recovery) / len(valid_recovery)
    )
    recovery_worst = (
        None if invalid_recovery_count else min(valid_recovery)
    )
    completed = len(metric_rows)
    failed = sum(
        1
        for value in metric_rows
        if value.get("supervised_tokens", 0) in (None, 0)
    )
    gate_rows = (
        _gate_row(
            "completed_fold_count",
            observed=completed,
            operator=">=",
            threshold=int(gates["required_completed_fold_count"]),
        ),
        _gate_row(
            "failed_fold_count",
            observed=failed,
            operator="<=",
            threshold=int(gates["maximum_failed_fold_count"]),
        ),
        _gate_row(
            "family_macro_delta_nll_per_token",
            observed=float(macro["delta_nll_per_token"]),
            operator="<=",
            threshold=float(
                gates["maximum_family_macro_delta_nll_per_token"]
            ),
        ),
        _gate_row(
            "worst_family_delta_nll_per_token",
            observed=float(worst["delta_nll_per_token"]),
            operator="<=",
            threshold=float(
                gates["maximum_worst_family_delta_nll_per_token"]
            ),
        ),
        _gate_row(
            "family_macro_native_to_candidate_kl_per_token",
            observed=float(macro["native_to_candidate_kl_per_token"]),
            operator="<=",
            threshold=float(
                gates[
                    "maximum_family_macro_native_to_candidate_kl_per_token"
                ]
            ),
        ),
        _gate_row(
            "family_macro_top1_agreement_to_native",
            observed=float(macro["top1_agreement_to_native"]),
            operator=">=",
            threshold=float(
                gates["minimum_family_macro_top1_agreement_to_native"]
            ),
        ),
        _gate_row(
            "family_macro_deletion_nll_recovery_fraction",
            observed=recovery_macro,
            operator=">=",
            threshold=float(
                gates[
                    "minimum_family_macro_deletion_nll_recovery_fraction"
                ]
            ),
        ),
        _gate_row(
            "worst_family_deletion_nll_recovery_fraction",
            observed=recovery_worst,
            operator=">=",
            threshold=float(
                gates[
                    "minimum_worst_family_deletion_nll_recovery_fraction"
                ]
            ),
        ),
        _gate_row(
            "positive_exact_parameter_savings",
            observed=int(resources["net_parameter_savings"]) > 0,
            operator="==",
            threshold=bool(gates["require_positive_exact_parameter_savings"]),
        ),
        _gate_row(
            "positive_logical_mac_savings",
            observed=int(resources["net_dense_macs_saved_per_token"]) > 0,
            operator="==",
            threshold=bool(gates["require_positive_logical_mac_savings"]),
        ),
    )
    passed = all(bool(row["passed"]) for row in gate_rows)
    return {
        "protocol_sha256": frozen["artifact_sha256"],
        "decision_policy": gates["decision_policy"],
        "deletion_nll_recovery_contract": dict(recovery_contract),
        "deletion_nll_recovery_fraction_by_family_alias": {
            f"family_{index:02d}": value
            for index, value in enumerate(recovery)
        },
        "deletion_nll_recovery_denominator_valid": (
            invalid_recovery_count == 0
        ),
        "deletion_nll_recovery_invalid_denominator_count": (
            invalid_recovery_count
        ),
        "gate_table": list(gate_rows),
        "all_required_gates_pass": passed,
        "next_action": (
            "freeze_full_eight_family_refit_then_replay_eligible_open_"
            "development_assessment"
            if passed
            else "stop_keep_other_roles_closed_and_revise_a_fit_recipe"
        ),
    }


def _forbidden_output_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized.startswith("contains_"):
        return False
    return normalized in {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "prompt_sha256",
        "prompt_sha256s",
        "ordered_prompt_sha256s",
        "family_id",
        "family_ids",
        "ordered_family_ids",
        "input_ids",
        "token_ids",
        "tokens",
        "logits",
        "weights",
        "state_dict",
        "raw_rows",
        "activation_rows",
        "gradient_rows",
    }


def _reject_forbidden_output_fields(value: object, *, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if _forbidden_output_key(key):
                raise ValueError(f"{path}.{key} is a forbidden source field")
            _reject_forbidden_output_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_forbidden_output_fields(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite scalar")


_REPORT_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "experiment",
    "authority",
    "protocol",
    "fit_collection",
    "folds",
    "aggregate",
    "decision",
    "resources",
    "lineage",
    "scope",
    "safety",
    "heldout_confirmation",
    "serving_authorized",
    "compression_claim",
    "report_sha256",
}
_FOLD_REPORT_FIELDS = {
    "held_family_alias",
    "training_family_aliases",
    "training_family_count",
    "held_family_excluded",
    "protocol_fold_sha256",
    "held_membership_sha256",
    "training_membership_sha256",
    "artifact_sha256",
    "fit_split_sha256",
    "held_split_sha256",
    "fit_observations",
    "held_observations",
    "fit_sequences",
    "held_sequences",
    "fisher_normalization",
    "fit_fisher_total_mass_per_training_family",
    "held_fisher_total_mass",
    "held_family_excluded_from_center_basis_fisher_and_generator",
    "held_rows_used_only_for_fixed_rank_descriptive_metrics",
    "lowering_sha256_by_node",
    "evaluation",
}
_MATERIALIZATION_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "heldout_confirmation",
    "authority_sha256",
    "tokenization",
    "access",
    "safety",
    "materialization_sha256",
}
_FIT_COLLECTION_EXTRA_FIELDS = {
    "capture_count",
    "captured_examples",
    "captured_observations",
    "captured_sequences",
    "captured_row_key_sha256",
    "family_observations",
    "model_rows_recollected_per_fold",
}
_REPORT_SCOPE = {
    "whole_model_logits_evaluated": True,
    "compiled_layer_count": 1,
    "compiled_layer_ordinal": 17,
    "whole_model_compiled": False,
    "source_model_parameters_retained": True,
    "latency_or_kernel_speed_claim": False,
}
_REPORT_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_semantic_family_identifiers": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_or_gradient_rows": False,
    "contains_fold_model_tensors": False,
    "selection_opened": False,
    "guard_opened": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "held_family_excluded_from_every_fold_fit": True,
    "source_safe": True,
}


def _json_clone(value: object) -> object:
    """Return the strict-JSON canonical representation used on disk."""

    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _json_equal(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _embedded_frozen_protocol() -> dict[str, object]:
    frozen = build_default_v8_layer17_family_lofo_protocol()
    return {
        "artifact_sha256": frozen["artifact_sha256"],
        "corpus_authority": frozen["corpus_authority"],
        "first_arm": frozen["first_arm"],
        "folds": frozen["folds"],
        "gates": frozen["gates"],
        "evaluation_contract": frozen["evaluation_contract"],
        "claim_boundary": frozen["claim_boundary"],
        "runtime_fisher_normalization": _FISHER_NORMALIZATION,
    }


def _validate_fold_evaluation(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "execution_path",
        "supervised_tokens",
        "native",
        "conditions",
        "logical_valid_tokens",
        "graph",
        "resource_accounting",
    }:
        raise ValueError(f"{label} evaluation fields are invalid")
    tokens = value.get("supervised_tokens")
    logical_tokens = value.get("logical_valid_tokens")
    native = value.get("native")
    conditions = value.get("conditions")
    graph = value.get("graph")
    accounting = value.get("resource_accounting")
    expected_accounting_fields = set(_GRAPH_STATIC_FIELDS) | {
        "lofo_generated",
        "matched_deletion_executed_graph_macs",
        "latency_or_kernel_speed_claim",
    }
    raw_traversal_order = (
        graph.get("traversal_order") if isinstance(graph, Mapping) else None
    )
    traversal_order = (
        tuple(raw_traversal_order)
        if isinstance(raw_traversal_order, Sequence)
        and not isinstance(raw_traversal_order, (str, bytes))
        else ()
    )
    if (
        value.get("execution_path")
        != "paired_layer17_lofo_and_frozen_edgeless_graphs"
        or type(tokens) is not int
        or tokens <= 0
        or type(logical_tokens) is not int
        or logical_tokens <= 0
        or not isinstance(native, Mapping)
        or set(native) != {"nll_per_token"}
        or not isinstance(conditions, Mapping)
        or set(conditions) != set(_CONDITIONS)
        or not isinstance(graph, Mapping)
        or set(graph) != {"node_count", "interaction_count", "traversal_order"}
        or graph.get("node_count") != 4
        or graph.get("interaction_count") != 0
        or traversal_order != _LAYER17_NODE_NAMES
        or not isinstance(accounting, Mapping)
        or set(accounting) != expected_accounting_fields
        or accounting.get("matched_deletion_executed_graph_macs") != 0
        or accounting.get("latency_or_kernel_speed_claim") is not False
    ):
        raise ValueError(f"{label} evaluation contract drifted")
    logical = accounting.get("lofo_generated")
    if (
        not isinstance(logical, Mapping)
        or set(logical) != set(_GRAPH_LOGICAL_FIELDS)
        or any(
            type(value) is not int or value < 0 for value in logical.values()
        )
    ):
        raise ValueError(f"{label} logical resource accounting is invalid")
    native_raw = native["nll_per_token"]
    if (
        isinstance(native_raw, bool)
        or not isinstance(native_raw, (int, float))
        or not math.isfinite(float(native_raw))
    ):
        raise ValueError(f"{label} native NLL must be finite")
    native_nll = float(native_raw)
    metric_fields = {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
    for name in _CONDITIONS:
        row = conditions[name]
        if not isinstance(row, Mapping) or set(row) != metric_fields:
            raise ValueError(f"{label}/{name} metric fields are invalid")
        if any(
            isinstance(row[key], bool)
            or not isinstance(row[key], (int, float))
            for key in metric_fields
        ):
            raise ValueError(f"{label}/{name} metrics must be JSON numbers")
        metrics = {key: float(row[key]) for key in metric_fields}
        if (
            any(not math.isfinite(metric) for metric in metrics.values())
            or metrics["native_to_candidate_kl_per_token"] < 0.0
            or not 0.0 <= metrics["top1_agreement_to_native"] <= 1.0
            or not math.isclose(
                metrics["delta_nll_per_token"],
                metrics["nll_per_token"] - native_nll,
                rel_tol=0.0,
                abs_tol=2e-12,
            )
        ):
            raise ValueError(f"{label}/{name} metrics are inconsistent")
    return value


def _validate_fixed_arm_fold_resources(
    evaluation: Mapping[str, object],
    resources: Layer17NodeRankResourceRow,
    *,
    label: str,
) -> tuple[dict[str, object], int]:
    """Reconcile one fold's runtime receipt with the fixed analytic arm."""

    accounting = evaluation["resource_accounting"]
    assert isinstance(accounting, Mapping)
    static_resources = {
        name: accounting[name] for name in _GRAPH_STATIC_FIELDS
    }
    source_whole = accounting["source_whole_model_learned_parameters"]
    if (
        accounting["replacement_scope"]
        != "partial_native_mlp_mode_replacement"
        or accounting["replaced_layer_count"] != 1
        or accounting["graph_node_count"] != 4
        or accounting["fragment_count"] != 4
        or accounting["removed_mode_count"] != sum(LAYER17_NATIVE_MODE_COUNTS)
        or type(source_whole) is not int
        or source_whole <= resources.source_parameter_count
        or accounting["native_removed_learned_parameters"]
        != resources.source_parameter_count
        or accounting["modal_graph_learned_parameters"]
        != resources.graph_parameter_count
        or accounting["net_stored_parameter_savings"]
        != resources.net_parameter_savings
        or accounting["candidate_whole_model_learned_parameters"]
        != source_whole - resources.net_parameter_savings
        or accounting["graph_runtime_storage"]
        != "registered_copied_device_local_graph_parameters"
    ):
        raise ValueError(f"{label} static resources contradict the fixed arm")
    valid_tokens = int(evaluation["logical_valid_tokens"])
    logical = accounting["lofo_generated"]
    assert isinstance(logical, Mapping)
    expected_native_macs = valid_tokens * resources.source_macs_per_token
    expected_graph_macs = valid_tokens * resources.graph_dense_macs_per_token
    additions = logical["logical_modal_graph_additions"]
    if (
        logical["logical_linear_macs_native_removed"] != expected_native_macs
        or logical["logical_modal_graph_macs"] != expected_graph_macs
        or logical["logical_executed_modal_graph_macs"] != expected_graph_macs
        or logical["net_logical_macs_saved"]
        != expected_native_macs - expected_graph_macs
        or logical["logical_executed_modal_graph_additions"] != additions
        or type(additions) is not int
        or additions < 0
        or additions % valid_tokens != 0
    ):
        raise ValueError(f"{label} logical resources contradict the fixed arm")
    return static_resources, additions // valid_tokens


def _validate_report_scientific_integrity(report: Mapping[str, object]) -> None:
    """Replay every public decision instead of trusting a replaceable hash."""

    if set(report) != _REPORT_FIELDS:
        raise ValueError("LOFO report fields differ from the frozen schema")
    if (
        report.get("scientific_status")
        != "opened_calibration_a_fit_outer_lofo_diagnostic"
        or report.get("heldout_confirmation") is not False
        or report.get("serving_authorized") is not False
        or report.get("compression_claim") is not False
        or report.get("scope") != _REPORT_SCOPE
        or report.get("safety") != _REPORT_SAFETY
    ):
        raise ValueError("LOFO report claim boundary or safety flags drifted")

    embedded_protocol = report.get("protocol")
    if not isinstance(embedded_protocol, Mapping) or not _json_equal(
        embedded_protocol,
        _embedded_frozen_protocol(),
    ):
        raise ValueError("LOFO report protocol differs from the frozen plan")
    frozen_protocol = build_default_v8_layer17_family_lofo_protocol()

    authority = report.get("authority")
    if not isinstance(authority, Mapping):
        raise TypeError("LOFO report authority must be a mapping")
    validate_gemma3_layer17_family_lofo_authority_metadata(authority)

    fit_collection = report.get("fit_collection")
    if not isinstance(fit_collection, Mapping) or set(fit_collection) != (
        _MATERIALIZATION_FIELDS | _FIT_COLLECTION_EXTRA_FIELDS
    ):
        raise ValueError("LOFO fit collection fields are invalid")
    materialization = {
        key: fit_collection[key] for key in _MATERIALIZATION_FIELDS
    }
    validate_gemma3_layer17_family_lofo_materialization_metadata(
        materialization
    )
    family_observations = fit_collection.get("family_observations")
    if (
        materialization.get("authority_sha256")
        != authority.get("authority_sha256")
        or fit_collection.get("capture_count") != 1
        or fit_collection.get("captured_examples") != _EXPECTED_EXAMPLES
        or fit_collection.get("captured_sequences") != _EXPECTED_EXAMPLES
        or fit_collection.get("model_rows_recollected_per_fold") is not False
        or not isinstance(tokenization := materialization.get("tokenization"), Mapping)
        or tokenization.get("device") != "cpu"
        or not isinstance(family_observations, Mapping)
        or set(family_observations) != set(V8_FAMILY_LOFO_FAMILY_ALIASES)
        or any(
            type(count) is not int or count <= 0
            for count in family_observations.values()
        )
        or type(fit_collection.get("captured_observations")) is not int
        or fit_collection["captured_observations"]
        != sum(int(value) for value in family_observations.values())
    ):
        raise ValueError("LOFO fit collection ownership or capture drifted")
    _require_sha256(
        fit_collection.get("captured_row_key_sha256"),
        label="captured row-key",
    )
    token_blocks = (
        tokenization.get("blocks")
        if isinstance(tokenization, Mapping)
        else None
    )
    if (
        not isinstance(token_blocks, Mapping)
        or set(token_blocks) != set(V8_FAMILY_LOFO_FAMILY_ALIASES)
    ):
        raise ValueError("LOFO tokenized family blocks are unavailable")

    experiment = report.get("experiment")
    if not isinstance(experiment, Mapping) or set(experiment) != {
        "experiment_kind",
        "scientific_role",
        "model_id",
        "requested_revision",
        "adapter_model_fingerprint",
        "source_model_unchanged",
    }:
        raise ValueError("LOFO experiment fields are invalid")
    if (
        experiment.get("experiment_kind")
        != "gemma3_layer17_v8_fit_family_lofo_v1"
        or experiment.get("scientific_role")
        != "calibration_a_fit_cross_fitted_diagnostic"
        or experiment.get("model_id") != DEFAULT_MODEL_ID
        or not isinstance(experiment.get("requested_revision"), str)
        or _REVISION.fullmatch(experiment["requested_revision"]) is None
        or experiment.get("source_model_unchanged") is not True
    ):
        raise ValueError("LOFO experiment contract drifted")
    _require_sha256(
        experiment.get("adapter_model_fingerprint"),
        label="adapter model fingerprint",
    )

    resources = report.get("resources")
    if not isinstance(resources, Mapping):
        raise TypeError("LOFO resources must be a mapping")
    restored_resources = Layer17NodeRankResourceRow.from_state_dict(resources)
    expected_resources = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=DEFAULT_MODE_RANK_CAP,
        generator_rank=_GENERATOR_RANK,
        edge_policy="edgeless",
    )
    if not _json_equal(
        restored_resources.state_dict(),
        expected_resources.state_dict(),
    ):
        raise ValueError("LOFO resources differ from fixed cap48/r16/edgeless")

    raw_folds = report.get("folds")
    if isinstance(raw_folds, (str, bytes)) or not isinstance(
        raw_folds, Sequence
    ) or len(raw_folds) != _EXPECTED_FAMILIES:
        raise ValueError("LOFO report must contain eight fold results")
    expected_folds = build_layer17_family_folds(
        V8_FAMILY_LOFO_FAMILY_ALIASES,
        protocol=frozen_protocol,
    )
    evaluations: list[Mapping[str, object]] = []
    static_resource_reference: dict[str, object] | None = None
    additions_per_token_reference: int | None = None
    for index, (raw, expected_fold) in enumerate(
        zip(raw_folds, expected_folds, strict=True)
    ):
        label = f"fold {index}"
        if not isinstance(raw, Mapping) or set(raw) != _FOLD_REPORT_FIELDS:
            raise ValueError(f"{label} report fields are invalid")
        expected_metadata = expected_fold.metadata()
        if any(
            not _json_equal(raw.get(key), expected_metadata[key])
            for key in expected_metadata
        ):
            raise ValueError(f"{label} protocol identity or ownership drifted")
        for key in (
            "fit_split_sha256",
            "held_split_sha256",
        ):
            _require_sha256(raw.get(key), label=f"{label} {key}")
        lowerings = raw.get("lowering_sha256_by_node")
        evaluation = _validate_fold_evaluation(
            raw.get("evaluation"),
            label=label,
        )
        evaluation_graph = evaluation["graph"]
        assert isinstance(evaluation_graph, Mapping)
        traversal_order = tuple(evaluation_graph["traversal_order"])  # type: ignore[arg-type]
        held_alias = expected_fold.held_family_alias
        training_aliases = expected_fold.training_family_aliases
        held_block = token_blocks[held_alias]
        if not isinstance(held_block, Mapping):
            raise TypeError(f"{label} tokenized family block is invalid")
        if (
            raw.get("fit_split_sha256") == raw.get("held_split_sha256")
            or type(raw.get("fit_observations")) is not int
            or int(raw["fit_observations"]) <= 0
            or type(raw.get("held_observations")) is not int
            or int(raw["held_observations"]) <= 0
            or raw.get("fit_sequences") != 224
            or raw.get("held_sequences") != 32
            or raw.get("fisher_normalization") != _FISHER_NORMALIZATION
            or not math.isclose(
                float(raw.get("fit_fisher_total_mass_per_training_family", 0.0)),
                1.0 / 7.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or float(raw.get("held_fisher_total_mass", 0.0)) != 1.0
            or raw.get(
                "held_family_excluded_from_center_basis_fisher_and_generator"
            )
            is not True
            or raw.get(
                "held_rows_used_only_for_fixed_rank_descriptive_metrics"
            )
            is not True
            or not isinstance(lowerings, Mapping)
            or len(lowerings) != len(LAYER17_FRAGMENT_IDS)
            or set(lowerings) != set(traversal_order)
            or raw.get("held_observations")
            != family_observations[held_alias]
            or raw.get("fit_observations")
            != sum(
                int(family_observations[alias])
                for alias in training_aliases
            )
            or int(raw["fit_observations"])
            + int(raw["held_observations"])
            != fit_collection["captured_observations"]
            or evaluation.get("supervised_tokens")
            != held_block.get("supervised_tokens")
            or evaluation.get("logical_valid_tokens")
            != held_block.get("logical_valid_tokens")
        ):
            raise ValueError(f"{label} fitting receipt is invalid")
        for node_name, digest in lowerings.items():
            _require_sha256(
                digest,
                label=f"{label} lowering {node_name}",
            )
        static_resources, additions_per_token = (
            _validate_fixed_arm_fold_resources(
                evaluation,
                expected_resources,
                label=label,
            )
        )
        if static_resource_reference is None:
            static_resource_reference = static_resources
        elif static_resources != static_resource_reference:
            raise ValueError("LOFO static resources differ across folds")
        if additions_per_token_reference is None:
            additions_per_token_reference = additions_per_token
        elif additions_per_token != additions_per_token_reference:
            raise ValueError("LOFO graph additions per token differ across folds")
        evaluations.append(evaluation)

    recomputed_aggregate = aggregate_layer17_lofo_fold_metrics(evaluations)
    if not _json_equal(report.get("aggregate"), recomputed_aggregate):
        raise ValueError("LOFO aggregate differs from replayed fold metrics")
    recomputed_decision = evaluate_layer17_lofo_protocol_gates(
        protocol=frozen_protocol,
        fold_metrics=evaluations,
        aggregate=recomputed_aggregate,
        resources=expected_resources.state_dict(),
    )
    if not _json_equal(report.get("decision"), recomputed_decision):
        raise ValueError("LOFO decision differs from replayed protocol gates")

    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != {
        "frozen_v9_candidate_file",
        "frozen_v9_candidate_file_sha256",
        "frozen_v9_candidate_scientific_sha256",
        "base_artifact_file",
        "base_artifact_file_sha256",
        "frozen_topology_sha256",
        "fragment_plan_sha256",
        "fragment_selection_sha256",
    }:
        raise ValueError("LOFO lineage fields are invalid")
    if (
        lineage.get("frozen_topology_sha256") != LAYER17_TOPOLOGY_SHA256
        or lineage.get("frozen_v9_candidate_file")
        != Path(DEFAULT_FROZEN_CAP48_CANDIDATE).name
        or lineage.get("base_artifact_file") != Path(DEFAULT_BASE_ARTIFACT).name
    ):
        raise ValueError("LOFO lineage topology or filenames drifted")
    for name in (
        "frozen_v9_candidate_file_sha256",
        "frozen_v9_candidate_scientific_sha256",
        "base_artifact_file_sha256",
        "fragment_plan_sha256",
        "fragment_selection_sha256",
    ):
        _require_sha256(lineage.get(name), label=f"lineage {name}")


def build_layer17_v8_fit_lofo_report(
    *,
    protocol: Mapping[str, object],
    authority: Mapping[str, object],
    experiment: Mapping[str, object],
    fit_collection: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    resources: Mapping[str, object],
    decision: Mapping[str, object],
    lineage: Mapping[str, object],
) -> dict[str, object]:
    """Build and hash one source-safe scalar-only diagnostic report."""

    fold_rows = tuple(dict(value) for value in folds)
    if len(fold_rows) != _EXPECTED_FAMILIES:
        raise ValueError("report requires exactly eight fold rows")
    held = tuple(value.get("held_family_alias") for value in fold_rows)
    if held != tuple(f"family_{index:02d}" for index in range(8)):
        raise ValueError("report folds must follow canonical family order")
    frozen_protocol = validate_v8_layer17_family_lofo_protocol(protocol)
    if frozen_protocol["artifact_sha256"] != (
        FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
    ):
        raise ValueError("report is not bound to the frozen LOFO protocol")
    first_arm = frozen_protocol["first_arm"]
    frozen_folds = frozen_protocol["folds"]
    assert isinstance(first_arm, Mapping)
    assert isinstance(frozen_folds, Sequence)
    if (
        first_arm.get("mode_rank_cap") != DEFAULT_MODE_RANK_CAP
        or first_arm.get("generator_rank") != _GENERATOR_RANK
        or first_arm.get("edge_policy") != "edgeless"
        or float(first_arm.get("ridge", -1.0)) != _RIDGE
        or tuple(
            value.get("protocol_fold_sha256") for value in fold_rows
        )
        != tuple(
            value.get("artifact_sha256")  # type: ignore[union-attr]
            for value in frozen_folds
        )
    ):
        raise ValueError("report arm or folds differ from frozen protocol")
    decision_row = dict(decision)
    if decision_row.get("protocol_sha256") != frozen_protocol[
        "artifact_sha256"
    ]:
        raise ValueError("gate decision does not bind the frozen protocol")
    report: dict[str, object] = {
        "schema": GEMMA3_LAYER17_V8_FIT_LOFO_SCHEMA,
        "format_version": GEMMA3_LAYER17_V8_FIT_LOFO_FORMAT_VERSION,
        "scientific_status": "opened_calibration_a_fit_outer_lofo_diagnostic",
        "experiment": dict(experiment),
        "authority": dict(authority),
        "protocol": {
            "artifact_sha256": frozen_protocol["artifact_sha256"],
            "corpus_authority": frozen_protocol["corpus_authority"],
            "first_arm": first_arm,
            "folds": frozen_folds,
            "gates": frozen_protocol["gates"],
            "evaluation_contract": frozen_protocol[
                "evaluation_contract"
            ],
            "claim_boundary": frozen_protocol["claim_boundary"],
            "runtime_fisher_normalization": _FISHER_NORMALIZATION,
        },
        "fit_collection": dict(fit_collection),
        "folds": list(fold_rows),
        "aggregate": dict(aggregate),
        "decision": decision_row,
        "resources": dict(resources),
        "lineage": dict(lineage),
        "scope": {
            "whole_model_logits_evaluated": True,
            "compiled_layer_count": 1,
            "compiled_layer_ordinal": 17,
            "whole_model_compiled": False,
            "source_model_parameters_retained": True,
            "latency_or_kernel_speed_claim": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_prompt_identities": False,
            "contains_semantic_family_identifiers": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_or_gradient_rows": False,
            "contains_fold_model_tensors": False,
            "selection_opened": False,
            "guard_opened": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "held_family_excluded_from_every_fold_fit": True,
            "source_safe": True,
        },
        "heldout_confirmation": False,
        "serving_authorized": False,
        "compression_claim": False,
    }
    _reject_forbidden_output_fields(report)
    canonical = _json_clone(report)
    assert isinstance(canonical, dict)
    canonical["report_sha256"] = _sha256(
        canonical,
        domain=_REPORT_DOMAIN,
    )
    return validate_gemma3_layer17_v8_fit_lofo_report(canonical)


def validate_gemma3_layer17_v8_fit_lofo_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("LOFO report must be a mapping")
    canonical = _json_clone(value)
    if not isinstance(canonical, dict):
        raise TypeError("LOFO report must canonicalize to one object")
    report = canonical
    if (
        report.get("schema") != GEMMA3_LAYER17_V8_FIT_LOFO_SCHEMA
        or type(report.get("format_version")) is not int
        or report.get("format_version")
        != GEMMA3_LAYER17_V8_FIT_LOFO_FORMAT_VERSION
    ):
        raise ValueError("unsupported layer-17 v8 fit LOFO report")
    supplied = _require_sha256(
        report.pop("report_sha256", None),
        label="report_sha256",
    )
    _reject_forbidden_output_fields(report)
    expected = _sha256(report, domain=_REPORT_DOMAIN)
    if supplied != expected:
        raise ValueError("layer-17 v8 fit LOFO report hash mismatch")
    report["report_sha256"] = supplied
    _validate_report_scientific_integrity(report)
    return report


def save_gemma3_layer17_v8_fit_lofo_report(
    path: Path | str,
    report: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("LOFO output must be JSON under .local-runs")
    if destination.exists():
        raise FileExistsError("refusing to overwrite layer-17 LOFO report")
    validated = validate_gemma3_layer17_v8_fit_lofo_report(report)
    encoded = _canonical_json_bytes(validated) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded)
    restored = load_gemma3_layer17_v8_fit_lofo_report(destination)
    if restored != validated:
        raise RuntimeError("saved layer-17 LOFO report roundtrip drifted")
    return restored


def load_gemma3_layer17_v8_fit_lofo_report(
    path: Path | str,
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("layer-17 LOFO report is not strict JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("layer-17 LOFO report must contain one object")
    return validate_gemma3_layer17_v8_fit_lofo_report(raw)


def _authority_metadata(authority: object) -> tuple[str, dict[str, object]]:
    digest = _require_sha256(
        getattr(authority, "authority_sha256", None),
        label="LOFO authority",
    )
    metadata = getattr(authority, "metadata", None)
    if not callable(metadata):
        raise TypeError("LOFO authority does not expose metadata()")
    safe = metadata()
    if not isinstance(safe, Mapping):
        raise TypeError("LOFO authority metadata must be a mapping")
    _reject_forbidden_output_fields(safe, path="authority")
    return digest, dict(safe)


def _family_blocks(
    value: object,
) -> tuple[tuple[str, tuple[CalibrationBatch, ...]], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("materialized authority blocks must be a sequence")
    blocks: list[tuple[str, tuple[CalibrationBatch, ...]]] = []
    for raw in value:
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
        ):
            raise TypeError("family block must contain alias and batches")
        alias, batches = raw
        if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
            raise ValueError("family block alias is invalid")
        materialized = tuple(batches)  # type: ignore[arg-type]
        if not materialized or any(
            not isinstance(batch, CalibrationBatch) for batch in materialized
        ):
            raise TypeError("family block batches are invalid")
        blocks.append((alias, materialized))
    aliases = _canonical_aliases(
        tuple(alias for alias, _ in blocks),
        expected=_EXPECTED_FAMILIES,
    )
    if aliases != tuple(alias for alias, _ in blocks):
        raise ValueError("family blocks must use canonical order")
    examples = [
        example_id
        for _, batches in blocks
        for batch in batches
        for example_id in (
            batch.example_ids if batch.example_ids is not None else ()
        )
    ]
    if (
        any(batch.example_ids is None for _, batches in blocks for batch in batches)
        or len(examples) != _EXPECTED_EXAMPLES
        or len(set(examples)) != _EXPECTED_EXAMPLES
    ):
        raise ValueError("family blocks must cover 256 unique fit examples")
    return tuple(blocks)


def _candidate_lineage(
    candidate: Mapping[str, object],
    *,
    path: Path | str,
) -> dict[str, object]:
    config = candidate.get("config")
    graph = candidate.get("edgeless_graph")
    if (
        not isinstance(config, Mapping)
        or config.get("mode_rank_cap") != DEFAULT_MODE_RANK_CAP
        or config.get("generator_rank") != _GENERATOR_RANK
        or config.get("edge_policy") != "edgeless"
        or tuple(config.get("fragment_ids", ())) != LAYER17_FRAGMENT_IDS
        or not isinstance(graph, Mapping)
        or graph.get("interactions") not in ([], ())
    ):
        raise ValueError("frozen candidate is not the fixed cap48/r16 graph")
    return {
        "frozen_v9_candidate_file": Path(path).name,
        "frozen_v9_candidate_file_sha256": _file_sha256(path),
        "frozen_v9_candidate_scientific_sha256": _require_sha256(
            candidate.get("scientific_payload_sha256"),
            label="frozen candidate scientific payload",
        ),
    }


def _ordered_restored_lowerings(
    graph: object,
    lowerings: Mapping[str, object],
) -> tuple[object, ...]:
    """Convert a restored node map into the executor's traversal sequence."""

    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Sequence):
        raise TypeError("restored graph nodes must be a sequence")
    names = tuple(getattr(node, "name", None) for node in nodes)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(lowerings, Mapping)
        or set(lowerings) != set(names)
    ):
        raise ValueError("restored graph and lowering catalog differ")
    return tuple(lowerings[name] for name in names)


def _load_authenticated_protocol(
    corpus_artifact_path: Path | str,
    *,
    authority_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the frozen protocol against the prompt-free corpus."""

    source = Path(corpus_artifact_path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v8 prompt-free corpus artifact is not strict JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("v8 prompt-free corpus artifact must be an object")
    protocol = validate_v8_layer17_family_lofo_protocol(
        build_authenticated_v8_layer17_family_lofo_protocol(raw)
    )
    authority_corpus = authority_metadata.get("corpus")
    authority_protocol = authority_metadata.get("protocol")
    protocol_corpus = protocol.get("corpus_authority")
    role_bindings = protocol.get("role_bindings")
    protocol_folds = protocol.get("folds")
    if (
        not isinstance(authority_corpus, Mapping)
        or not isinstance(authority_protocol, Mapping)
        or not isinstance(protocol_corpus, Mapping)
        or not isinstance(role_bindings, Mapping)
        or not isinstance(role_bindings.get("fit"), Mapping)
        or isinstance(protocol_folds, (str, bytes))
        or not isinstance(protocol_folds, Sequence)
    ):
        raise TypeError("protocol/authority bindings are unavailable")
    fit_binding = role_bindings["fit"]
    assert isinstance(fit_binding, Mapping)
    expected_authority_protocol = {
        "protocol_artifact_sha256": protocol["artifact_sha256"],
        "fit_membership_sha256": protocol_corpus[
            "fit_membership_sha256"
        ],
        "family_alias_mapping_sha256": protocol_corpus[
            "family_alias_mapping_sha256"
        ],
        "fold_count": len(protocol_folds),
        "folds": [
            {
                "held_family_alias": raw["held_family_alias"],
                "training_family_aliases": raw[
                    "training_family_aliases"
                ],
                "held_example_count": raw["held_example_count"],
                "training_example_count": raw["training_example_count"],
                "held_membership_sha256": raw[
                    "held_membership_sha256"
                ],
                "training_membership_sha256": raw[
                    "training_membership_sha256"
                ],
            }
            for raw in protocol_folds
            if isinstance(raw, Mapping)
        ],
    }
    if (
        protocol.get("artifact_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or _canonical_json_bytes(authority_protocol)
        != _canonical_json_bytes(expected_authority_protocol)
        or authority_corpus.get("corpus_artifact_sha256")
        != protocol_corpus.get("artifact_sha256")
        or authority_corpus.get("tokenizer_contract_sha256")
        != protocol_corpus.get("tokenizer_contract_sha256")
        or authority_corpus.get("fit_manifest_sha256")
        != fit_binding.get("manifest_sha256")
        or authority_corpus.get("fit_role_file_sha256")
        != fit_binding.get("source_file_sha256")
        or tuple(authority_corpus.get("block_labels", ()))
        != tuple(fit_binding.get("family_aliases", ()))
    ):
        raise ValueError("frozen protocol and A-fit authority differ")
    return protocol


def _batch_to_device(
    batch: CalibrationBatch,
    device: torch.device,
) -> CalibrationBatch:
    return CalibrationBatch(
        model_inputs={
            name: value.to(device=device)
            for name, value in batch.model_inputs.items()
        },
        targets=batch.targets.to(device=device),
        valid_positions=batch.valid_positions.to(device=device),
        shared_input_names=batch.shared_input_names,
        example_ids=batch.example_ids,
    )


def _blocks_to_device(
    blocks: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
    device: torch.device,
) -> tuple[tuple[str, tuple[CalibrationBatch, ...]], ...]:
    return tuple(
        (
            alias,
            tuple(_batch_to_device(batch, device) for batch in batches),
        )
        for alias, batches in blocks
    )


def _validate_fold_pilots_are_fit_only(
    pilots: Mapping[str, object],
) -> None:
    """Fail closed unless held rows were descriptive fixed-rank inputs only."""

    if set(pilots) != set(LAYER17_FRAGMENT_IDS):
        raise ValueError("fold pilot catalog differs from frozen fragments")
    for fragment_id, value in pilots.items():
        modes = getattr(value, "computational_modes", None)
        generators = getattr(value, "modal_generators", None)
        mode_config = getattr(modes, "config", None)
        generator_config = getattr(generators, "config", None)
        if (
            getattr(modes, "evaluation_used_for_basis_fit", None) is not False
            or getattr(modes, "evaluation_used_for_rank_selection", None)
            is not False
            or getattr(mode_config, "selection_rule", None) != "fixed_rank"
            or tuple(getattr(mode_config, "ranks", ()))
            != (getattr(mode_config, "selected_rank", None),)
            or getattr(generator_config, "selection_rule", None)
            != "fixed_rank"
            or tuple(getattr(generator_config, "ranks", ()))
            != (getattr(generator_config, "selected_rank", None),)
        ):
            raise RuntimeError(
                f"{fragment_id} allowed held rows into fitting or selection"
            )


def run_gemma3_layer17_v8_fit_lofo(
    *,
    revision: str,
    output: Path | str = DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    base_artifact_path: Path | str = DEFAULT_BASE_ARTIFACT,
    frozen_candidate_path: Path | str = DEFAULT_FROZEN_CAP48_CANDIDATE,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Run the fixed eight-fold A-fit diagnostic and write scalar JSON."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite layer-17 LOFO report")

    _progress("preflight: authenticate the A-fit-only family authority")
    authority = load_gemma3_layer17_family_lofo_authority(
        corpus_receipt_path=corpus_receipt_path,
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
    )
    authority_sha256, authority_safe = _authority_metadata(authority)
    protocol = _load_authenticated_protocol(
        corpus_artifact_path,
        authority_metadata=authority_safe,
    )

    _progress("preflight: restore frozen topology and v9 cap48 comparator")
    upstream = load_gemma3_modal_generator_dev_artifact(base_artifact_path)
    fit_trace, catalog, fisher, clusters, fragment_plan = (
        _restore_upstream_analysis(upstream)
    )
    selection = select_top_fisher_same_layer_fragments(
        fragment_plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    _validate_frozen_selection(selection)
    frozen_candidate = load_gemma3_layer17_capped_node_candidate(
        frozen_candidate_path
    )
    frozen_lineage = _candidate_lineage(
        frozen_candidate,
        path=frozen_candidate_path,
    )
    frozen_graph, frozen_lowerings, _ = (
        restore_gemma3_layer17_capped_node_runtime(frozen_candidate)
    )

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load the pinned local Gemma checkpoint")
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
    fingerprint = adapter.model_fingerprint()
    _validate_upstream_bindings(
        upstream,
        model_id=model_id,
        revision=revision,
        model_fingerprint=fingerprint,
    )
    if frozen_graph.model_fingerprint != fingerprint:
        raise ValueError("frozen cap48 candidate model fingerprint differs")
    live_specs, leaf_site, _ = _whole_model_layer_specs(adapter)
    if tuple(spec.layer_id for spec in live_specs) != tuple(
        spec.layer_id for spec in fit_trace.layer_specs
    ):
        raise ValueError("live layer catalog differs from frozen topology")

    _progress("tokenize: materialize eight opaque A-fit family blocks")
    raw_blocks, materialization_safe = materialize_gemma3_layer17_family_lofo(
        authority,
        tokenizer,
    )
    authority_blocks = _family_blocks(raw_blocks)
    if not isinstance(materialization_safe, Mapping):
        raise TypeError("LOFO materialization metadata must be a mapping")
    _reject_forbidden_output_fields(
        materialization_safe,
        path="materialization",
    )
    blocks = _blocks_to_device(authority_blocks, device)
    all_batches = tuple(batch for _, batches in blocks for batch in batches)
    family_alias_by_example = {
        example_id: alias
        for alias, batches in blocks
        for batch in batches
        for example_id in batch.example_ids or ()
    }

    _progress("rows: collect four layer-17 fragment streams exactly once")
    all_rows = _collect_same_layer_native_rows(
        adapter,
        all_batches,
        selection=selection,
        leaf_activation_site=leaf_site,
    )
    family_rows = partition_aligned_fragment_rows_by_family(
        all_rows,
        family_alias_by_example,
    )
    folds = build_layer17_family_folds(
        tuple(family_rows),
        protocol=protocol,
    )
    resource_row = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=DEFAULT_MODE_RANK_CAP,
        generator_rank=_GENERATOR_RANK,
        edge_policy="edgeless",
    )
    frozen_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        frozen_graph,
        _ordered_restored_lowerings(frozen_graph, frozen_lowerings),
    )

    fold_reports: list[dict[str, object]] = []
    for index, fold in enumerate(folds, start=1):
        _progress(
            f"fold {index}/{len(folds)}: hold {fold.held_family_alias}; "
            "normalize train-seven Fisher and refit all four generators"
        )
        train_rows, held_rows, fold_receipt = make_layer17_lofo_fold_rows(
            family_rows,
            fold,
            authority_sha256=authority_sha256,
        )
        pilots = fit_layer17_capped_node_pilots(
            train_rows,
            held_rows,
            selection=selection,
            source_model_sha256=fingerprint,
            parameter_catalog_sha256=catalog.artifact_sha256,
            fisher_coupling_sha256=fisher.artifact_sha256,
            fragment_plan=fragment_plan,
            fit_split_sha256=str(fold_receipt["fit_split_sha256"]),
            selection_split_sha256=str(fold_receipt["held_split_sha256"]),
            mode_rank_cap=DEFAULT_MODE_RANK_CAP,
            generator_rank=_GENERATOR_RANK,
            ridge=_RIDGE,
        )
        _validate_fold_pilots_are_fit_only(pilots)
        graph = build_edgeless_same_layer_graph(
            selection,
            fragment_plan=fragment_plan,
            lowerings_by_fragment={
                fragment_id: pilot.lowering
                for fragment_id, pilot in pilots.items()
            },
        )
        if (
            graph.graph_plan.parameter_count != resource_row.graph_parameter_count
            or graph.graph_plan.macs_per_token
            != resource_row.graph_dense_macs_per_token
        ):
            raise RuntimeError("fold graph resources differ from fixed cap48 plan")
        executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            graph.graph_plan,
            graph.lowerings,
        )
        held_batches = dict(blocks)[fold.held_family_alias]
        _progress(
            f"fold {index}/{len(folds)}: score LOFO, frozen v9, and deletion"
        )
        evaluation = evaluate_layer17_lofo_fold(
            adapter,
            executor,
            frozen_executor,
            held_batches,
        )
        fold_reports.append(
            {
                **fold_receipt,
                "lowering_sha256_by_node": {
                    node: lowering.artifact_sha256
                    for node, lowering in sorted(
                        graph.lowerings_by_node.items()
                    )
                },
                "evaluation": evaluation,
            }
        )
        del train_rows, held_rows, pilots, graph, executor

    if adapter.model_fingerprint() != fingerprint:
        raise RuntimeError("LOFO diagnostic mutated the source model")
    aggregate = aggregate_layer17_lofo_fold_metrics(
        tuple(value["evaluation"] for value in fold_reports)  # type: ignore[arg-type]
    )
    resources = resource_row.state_dict()
    decision = evaluate_layer17_lofo_protocol_gates(
        protocol=protocol,
        fold_metrics=tuple(
            value["evaluation"] for value in fold_reports  # type: ignore[arg-type]
        ),
        aggregate=aggregate,
        resources=resources,
    )
    report = build_layer17_v8_fit_lofo_report(
        protocol=protocol,
        authority=authority_safe,
        experiment={
            "experiment_kind": "gemma3_layer17_v8_fit_family_lofo_v1",
            "scientific_role": "calibration_a_fit_cross_fitted_diagnostic",
            "model_id": model_id,
            "requested_revision": revision,
            "adapter_model_fingerprint": fingerprint,
            "source_model_unchanged": True,
        },
        fit_collection={
            **dict(materialization_safe),
            "capture_count": 1,
            "captured_examples": _EXPECTED_EXAMPLES,
            "captured_observations": all_rows.observations,
            "captured_sequences": all_rows.sequences,
            "captured_row_key_sha256": all_rows.row_key_sha256,
            "family_observations": {
                alias: rows.observations for alias, rows in family_rows.items()
            },
            "model_rows_recollected_per_fold": False,
        },
        folds=fold_reports,
        aggregate=aggregate,
        resources=resources,
        decision=decision,
        lineage={
            **frozen_lineage,
            "base_artifact_file": Path(base_artifact_path).name,
            "base_artifact_file_sha256": _file_sha256(base_artifact_path),
            "frozen_topology_sha256": LAYER17_TOPOLOGY_SHA256,
            "fragment_plan_sha256": fragment_plan.artifact_sha256,
            "fragment_selection_sha256": selection.artifact_sha256,
        },
    )
    _progress("report: write source-safe hash-authenticated JSON")
    return save_gemma3_layer17_v8_fit_lofo_report(destination, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed cap48/r16 layer-17 outer-LOFO refits on authenticated "
            "Calibration-A fit only."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
    )
    parser.add_argument(
        "--corpus-receipt",
        type=Path,
        default=DEFAULT_RECEIPT_OUTPUT,
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=DEFAULT_CORPUS_OUTPUT,
    )
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_BASE_ARTIFACT,
    )
    parser.add_argument(
        "--frozen-candidate",
        type=Path,
        default=DEFAULT_FROZEN_CAP48_CANDIDATE,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_layer17_v8_fit_lofo(
        revision=arguments.revision,
        output=arguments.output,
        corpus_receipt_path=arguments.corpus_receipt,
        corpus_artifact_path=arguments.corpus_artifact,
        fit_input_path=arguments.fit_input,
        base_artifact_path=arguments.base_artifact,
        frozen_candidate_path=arguments.frozen_candidate,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
