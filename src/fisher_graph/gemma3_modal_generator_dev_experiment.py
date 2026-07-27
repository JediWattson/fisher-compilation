"""Development-only Gemma 3 modal-generator compiler experiment.

This opt-in runner implements one deliberately narrow end-to-end pilot::

    weights -> prompt Fisher -> parameter clusters -> computational modes
            -> modal generator -> generator graph -> graph traversal

The parameter clustering pass is frozen from the fit split before the
evaluation split is inspected.  Evaluation is descriptive only: it cannot
select a cluster, computational-mode rank, or generator rank.  Both ranks are
explicit command-line inputs from predeclared ladders.

The saved artifact contains prompt-level Fisher summaries, authenticated
cluster/mode/generator artifacts, and source-free generator weights.  It does
not contain prompt text, token ids, raw token activation/gradient rows,
tokenizer state, or source-model weights.  Model loading is local-files-only,
and the default output lives under the repository's ignored ``.local-runs``
directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Protocol
import unicodedata

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .computational_modes import (
    ComputationalModeBinding,
    ComputationalModeRateCurve,
    fit_computational_mode_rate_curve,
)
from .fisher_prompt_clustering import (
    FisherPromptClusterConfig,
    FisherPromptClusterPlan,
    build_fisher_prompt_clusters,
)
from .gemma3_ablation_experiment import _update_payload_digest
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorExecutor,
    Gemma3ModalGeneratorModelExecution,
    Gemma3ModalGeneratorReplacement,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecution,
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_compiler_pipeline import (
    ModalCompilerPipeline,
    ModalSourceReplacementAccounting,
    build_modal_compiler_pipeline,
    build_modal_source_replacement_accounting,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorPlan,
    ModalGeneratorRateCurve,
    fit_modal_generator_rate_curve,
)
from .modal_generator_lowering import (
    ModalGeneratorLowering,
    lower_coordinate_modal_generator,
)
from .parameter_fisher_coupling import (
    GroupedVirtualGateFisher,
    NaturalMLPParameterGroupCatalog,
    NaturalMLPLayerParameterSpec,
    build_grouped_virtual_gate_fisher_from_trace,
    build_natural_mlp_parameter_group_catalog,
    natural_mlp_input_catalog_sha256,
)
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
    build_parameter_cluster_layer_fragments,
)
from .prompt_mode_tracing import (
    PromptModeTrace,
    PromptModeTraceProvenance,
    collect_prompt_mode_trace,
)
from .streaming_analysis import (
    ActivationScoreGradientRows,
    iter_activation_score_gradient_rows,
)
from .structured_mlp_cross_block_bundling import CrossBlockLayerSpec


GEMMA3_MODAL_GENERATOR_DEV_SCHEMA = (
    "fisher_graph.gemma3_modal_generator_development"
)
GEMMA3_MODAL_GENERATOR_DEV_FORMAT_VERSION = 3
DEFAULT_FIT_EXPORT = Path(
    ".local-runs/google--gemma-3-270m/"
    "dev-v9-a-fit-first40-export.json"
)
DEFAULT_EVAL_EXPORT = Path(
    ".local-runs/google--gemma-3-270m/"
    "dev-v9-a-fit-40-79-export.json"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-graph-dev-v3.pt"
)
DEFAULT_MAX_LENGTH = 256
DEFAULT_TOKENIZATION_BATCH_SIZE = 1
DEFAULT_CLUSTER_COUNT = 64
DEFAULT_MINIMUM_FRAGMENT_MODES = 8
DEFAULT_MODE_RANKS = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_SELECTED_MODE_RANK = 32
DEFAULT_GENERATOR_RANKS = (1, 2, 4, 8, 16, 32)
DEFAULT_SELECTED_GENERATOR_RANK = 16

_EXPORT_SCHEMA = "fisher_graph.local_v9_a_fit_development_export"
_EXPORT_FIELDS = {
    "calibration_b_exported",
    "family_ids",
    "fit_positions",
    "format_version",
    "guard_exported",
    "model_or_tokenizer_accessed",
    "prompt_sha256",
    "prompts",
    "schema",
    "scientific_status",
    "selection_rule",
    "source_corpus_id",
    "source_fit_prompt_index_sha256",
    "source_prompt_indices",
    "source_role",
    "test_exported",
    "validation_exported",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_EXPORT_DOMAIN = b"fisher_graph.gemma3_modal_generator.export.v1\0"
_OBJECTIVE_DOMAIN = b"fisher_graph.gemma3_modal_generator.objective.v1\0"
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_modal_generator.payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_modal_generator.report.v1\0"

_FORBIDDEN_ARTIFACT_KEYS = {
    "prompt",
    "prompts",
    "text",
    "token_ids",
    "input_ids",
    "targets",
    "score_gradients",
    "raw_token_rows",
    "raw_fit_rows",
    "raw_eval_rows",
    "model_state_dict",
    "source_state_dict",
    "tokenizer_state",
}
_ARTIFACT_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "protocol",
    "splits",
    "fit_prompt_trace",
    "eval_prompt_trace",
    "parameter_catalog",
    "fisher_coupling",
    "parameter_clusters",
    "parameter_cluster_fragments",
    "selected_layer_cluster",
    "computational_modes",
    "modal_generators",
    "modal_generator_lowering",
    "modal_generator_graph",
    "source_replacement_accounting",
    "modal_compiler_pipeline",
    "dense_fused_executable_generator",
    "computational_mode_metadata",
    "modal_generator_metadata",
    "modal_generator_lowering_metadata",
    "modal_generator_graph_metadata",
    "source_replacement_accounting_metadata",
    "modal_compiler_pipeline_metadata",
    "dense_generator_rate_curve",
    "evaluation",
    "contains_source_model_weights",
    "contains_prompt_text",
    "contains_token_ids",
    "contains_raw_token_rows",
    "contains_tokenizer_state",
    "contains_generator_weights",
    "scientific_payload_sha256",
}


def _progress(message: str) -> None:
    print(f"[gemma-modal-generator-dev] {message}", file=sys.stderr, flush=True)


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _raw_text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int_tuple(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            f"{label} must contain unique, strictly increasing positive ints"
        )
    return result


def _objective_sha256() -> str:
    return _json_sha256(
        {
            "objective": (
                "fisher_graph.compiler.calibration.CausalLanguageModelNLL"
            ),
            "ignore_index": -100,
            "reduction": "summed_per_independent_sequence",
            "virtual_gate_score": "sum_t_activation_times_d_nll_d_activation",
        },
        domain=_OBJECTIVE_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class DevelopmentPromptExport:
    """Strictly parsed self-attested local fit-only prompt declaration."""

    prompts: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]
    family_ids: tuple[str, ...]
    fit_positions: tuple[int, ...]
    source_prompt_indices: tuple[int, ...]
    source_corpus_id: str
    source_fit_prompt_index_sha256: str
    selection_rule: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        count = len(self.prompts)
        if count <= 0:
            raise ValueError("development prompt export cannot be empty")
        if any(
            not isinstance(prompt, str) or not prompt.strip()
            for prompt in self.prompts
        ):
            raise ValueError("development prompts must be nonempty strings")
        if len(set(self.prompts)) != count:
            raise ValueError("development prompts must be unique")
        if (
            len(self.prompt_sha256s) != count
            or len(self.family_ids) != count
            or len(self.fit_positions) != count
            or len(self.source_prompt_indices) != count
        ):
            raise ValueError("development export columns have unequal lengths")
        if tuple(
            _raw_text_sha256(prompt) for prompt in self.prompts
        ) != self.prompt_sha256s:
            raise ValueError("development prompt hashes do not match text")
        if len(set(self.prompt_sha256s)) != count:
            raise ValueError("development prompt hashes must be unique")
        for index, digest in enumerate(self.prompt_sha256s):
            _require_sha256(digest, label=f"prompt_sha256s[{index}]")
        if any(
            not isinstance(value, str) or not value for value in self.family_ids
        ):
            raise ValueError("family ids must be nonempty strings")
        for label, values in (
            ("fit_positions", self.fit_positions),
            ("source_prompt_indices", self.source_prompt_indices),
        ):
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{label} must contain nonnegative integers")
            if len(set(values)) != count:
                raise ValueError(f"{label} must contain unique values")
        if not isinstance(self.source_corpus_id, str) or not (
            self.source_corpus_id
        ):
            raise ValueError("source_corpus_id must be nonempty")
        _require_sha256(
            self.source_fit_prompt_index_sha256,
            label="source_fit_prompt_index_sha256",
        )
        if not isinstance(self.selection_rule, str) or not self.selection_rule:
            raise ValueError("selection_rule must be nonempty")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if self.artifact_sha256 != _json_sha256(
            self._safe_payload(),
            domain=_EXPORT_DOMAIN,
        ):
            raise ValueError("development prompt export hash mismatch")

    def _safe_payload(self) -> dict[str, object]:
        return {
            "schema": _EXPORT_SCHEMA,
            "format_version": 1,
            "scientific_status": "development_only",
            "source_role": "calibration_a_fit_only",
            "source_corpus_id": self.source_corpus_id,
            "selection_rule": self.selection_rule,
            "prompt_sha256s": self.prompt_sha256s,
            "family_ids": self.family_ids,
            "fit_positions": self.fit_positions,
            "source_prompt_indices": self.source_prompt_indices,
            "source_fit_prompt_index_sha256": (
                self.source_fit_prompt_index_sha256
            ),
            "guard_exported": False,
            "calibration_b_exported": False,
            "validation_exported": False,
            "test_exported": False,
            "model_or_tokenizer_accessed": False,
            "contains_prompt_text": False,
            "provenance_assurance": "declared_self_attested",
            "externally_authenticated": False,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._safe_payload(),
            "prompt_count": len(self.prompts),
            "family_count": len(set(self.family_ids)),
            "artifact_sha256": self.artifact_sha256,
        }


def load_development_prompt_export(
    path: Path | str,
) -> DevelopmentPromptExport:
    """Validate a self-attested fit-only export without external attestation."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != _EXPORT_FIELDS:
        raise ValueError("development prompt export fields are invalid")
    if (
        raw["schema"] != _EXPORT_SCHEMA
        or raw["format_version"] != 1
        or raw["scientific_status"] != "development_only"
        or raw["source_role"] != "calibration_a_fit_only"
        or any(
            raw[name] is not False
            for name in (
                "guard_exported",
                "calibration_b_exported",
                "validation_exported",
                "test_exported",
                "model_or_tokenizer_accessed",
            )
        )
    ):
        raise ValueError(
            "only development-only calibration-A fit exports are permitted"
        )
    sequence_fields = (
        "prompts",
        "prompt_sha256",
        "family_ids",
        "fit_positions",
        "source_prompt_indices",
    )
    if any(not isinstance(raw[name], list) for name in sequence_fields):
        raise TypeError("development export columns must be JSON lists")
    prompts = tuple(raw["prompts"])
    prompt_hashes = tuple(raw["prompt_sha256"])
    family_ids = tuple(raw["family_ids"])
    fit_positions = tuple(raw["fit_positions"])
    source_indices = tuple(raw["source_prompt_indices"])
    safe_payload = {
        "schema": _EXPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": "development_only",
        "source_role": "calibration_a_fit_only",
        "source_corpus_id": raw["source_corpus_id"],
        "selection_rule": raw["selection_rule"],
        "prompt_sha256s": prompt_hashes,
        "family_ids": family_ids,
        "fit_positions": fit_positions,
        "source_prompt_indices": source_indices,
        "source_fit_prompt_index_sha256": (
            raw["source_fit_prompt_index_sha256"]
        ),
        "guard_exported": False,
        "calibration_b_exported": False,
        "validation_exported": False,
        "test_exported": False,
        "model_or_tokenizer_accessed": False,
        "contains_prompt_text": False,
        "provenance_assurance": "declared_self_attested",
        "externally_authenticated": False,
    }
    return DevelopmentPromptExport(
        prompts=prompts,  # type: ignore[arg-type]
        prompt_sha256s=prompt_hashes,  # type: ignore[arg-type]
        family_ids=family_ids,  # type: ignore[arg-type]
        fit_positions=fit_positions,  # type: ignore[arg-type]
        source_prompt_indices=source_indices,  # type: ignore[arg-type]
        source_corpus_id=str(raw["source_corpus_id"]),
        source_fit_prompt_index_sha256=str(
            raw["source_fit_prompt_index_sha256"]
        ),
        selection_rule=str(raw["selection_rule"]),
        artifact_sha256=_json_sha256(
            safe_payload,
            domain=_EXPORT_DOMAIN,
        ),
    )


def validate_development_split_pair(
    fit: DevelopmentPromptExport,
    evaluation: DevelopmentPromptExport,
) -> dict[str, object]:
    """Check declared partitions for overlap without authenticating membership."""

    if not isinstance(fit, DevelopmentPromptExport) or not isinstance(
        evaluation,
        DevelopmentPromptExport,
    ):
        raise TypeError("fit and evaluation must be development exports")
    if (
        fit.source_corpus_id != evaluation.source_corpus_id
        or fit.source_fit_prompt_index_sha256
        != evaluation.source_fit_prompt_index_sha256
    ):
        raise ValueError("development splits must bind the same source corpus")
    prompt_overlap = set(fit.prompt_sha256s) & set(evaluation.prompt_sha256s)
    source_overlap = set(fit.source_prompt_indices) & set(
        evaluation.source_prompt_indices
    )
    if prompt_overlap or source_overlap:
        raise ValueError(
            "fit and development-evaluation declared memberships must not "
            "overlap"
        )
    family_overlap = set(fit.family_ids) & set(evaluation.family_ids)
    return {
        "fit_export_sha256": fit.artifact_sha256,
        "eval_export_sha256": evaluation.artifact_sha256,
        "prompt_disjoint": True,
        "source_prompt_index_disjoint": True,
        "family_disjoint": not family_overlap,
        "overlapping_family_count": len(family_overlap),
        "evaluation_role": "development_only_fit_partition_replay",
        "export_provenance_assurance": "declared_self_attested",
        "export_provenance_externally_authenticated": False,
        "split_membership_provenance": "caller_declared_self_attested",
        "split_membership_externally_authenticated": False,
        "declared_membership_overlap_checked": True,
        "heldout_guard_used": False,
        "calibration_b_used": False,
        "validation_used": False,
        "test_used": False,
    }


class ActivationRowFactory(Protocol):
    def __call__(
        self,
        model: Gemma3CausalLMAdapter,
        calibration_batches: Iterable[CalibrationBatch],
        *,
        activation_names: Sequence[str],
        score_objective: CausalLanguageModelNLL,
        leaf_activation_name: str | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> Iterable[ActivationScoreGradientRows]: ...


def _select_row_sites(
    rows: Iterable[ActivationScoreGradientRows],
    sites: tuple[str, ...],
) -> Iterable[ActivationScoreGradientRows]:
    """Yield an exact site subset and close the underlying gradient stream."""

    iterator = iter(rows)
    try:
        for row in iterator:
            if any(site not in row.activations for site in sites):
                raise ValueError("activation row is missing a selected site")
            yield ActivationScoreGradientRows(
                activations={site: row.activations[site] for site in sites},
                score_gradients={
                    site: row.score_gradients[site] for site in sites
                },
                logical_positions=row.logical_positions,
                loss=row.loss,
                example_id=row.example_id,
            )
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def collect_gemma_prompt_mode_trace(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    layer_specs: tuple[CrossBlockLayerSpec, ...],
    leaf_activation_site: str,
    split_sha256: str,
    objective_sha256: str,
    row_factory: ActivationRowFactory = iter_activation_score_gradient_rows,
) -> PromptModeTrace:
    """Collect a prompt summary while dropping the detached leaf site."""

    sites = tuple(spec.activation_site for spec in layer_specs)
    requested = tuple(dict.fromkeys((*sites, leaf_activation_site)))
    raw_rows = row_factory(
        adapter,
        batches,
        activation_names=requested,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=leaf_activation_site,
        accumulation_dtype=torch.float64,
    )
    return collect_prompt_mode_trace(
        _select_row_sites(raw_rows, sites),
        layer_specs=layer_specs,
        provenance=PromptModeTraceProvenance(
            source_model_fingerprint=adapter.model_fingerprint(),
            calibration_split_sha256=split_sha256,
            objective_sha256=objective_sha256,
        ),
    )


def select_top_fisher_layer_cluster(
    fisher: GroupedVirtualGateFisher,
    cluster_plan: FisherPromptClusterPlan,
    *,
    minimum_fragment_modes: int = DEFAULT_MINIMUM_FRAGMENT_MODES,
) -> tuple[
    ParameterClusterLayerFragmentPlan,
    ParameterClusterLayerFragment,
]:
    """Select the highest-mass executable single-layer cluster fragment."""

    if not isinstance(fisher, GroupedVirtualGateFisher):
        raise TypeError("fisher must be GroupedVirtualGateFisher")
    if not isinstance(cluster_plan, FisherPromptClusterPlan):
        raise TypeError("cluster_plan must be FisherPromptClusterPlan")
    if type(minimum_fragment_modes) is not int or minimum_fragment_modes <= 0:
        raise ValueError("minimum_fragment_modes must be positive")
    fisher.validate_integrity()
    cluster_plan.validate_integrity()
    if (
        cluster_plan.config.source_fisher_coupling_sha256
        != fisher.artifact_sha256
        or cluster_plan.mode_count != fisher.group_count
    ):
        raise ValueError("cluster plan does not bind the grouped Fisher")

    fragment_plan = build_parameter_cluster_layer_fragments(
        cluster_plan,
        fisher,
    )
    eligible = tuple(
        fragment
        for fragment in fragment_plan.top_by_fisher_mass(
            fragment_plan.fragment_count
        )
        if fragment.mode_count >= minimum_fragment_modes
    )
    if not eligible:
        raise ValueError(
            "no positive-Fisher layer-cluster fragment meets the mode minimum"
        )
    return fragment_plan, eligible[0]


def build_fit_fisher_cluster_pilot(
    fit_trace: PromptModeTrace,
    *,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    cluster_count: int,
    minimum_fragment_modes: int = DEFAULT_MINIMUM_FRAGMENT_MODES,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
    mode_chunk_size: int = 4096,
) -> tuple[
    GroupedVirtualGateFisher,
    FisherPromptClusterPlan,
    ParameterClusterLayerFragmentPlan,
    ParameterClusterLayerFragment,
]:
    """Build coupling/clusters from fit only and freeze one pilot fragment."""

    if not isinstance(fit_trace, PromptModeTrace):
        raise TypeError("fit_trace must be PromptModeTrace")
    if not isinstance(parameter_catalog, NaturalMLPParameterGroupCatalog):
        raise TypeError(
            "parameter_catalog must be NaturalMLPParameterGroupCatalog"
        )
    if (
        parameter_catalog.model_fingerprint
        != fit_trace.provenance.source_model_fingerprint
        or parameter_catalog.group_count != fit_trace.mode_count
    ):
        raise ValueError("fit trace and parameter catalog do not align")
    fisher = build_grouped_virtual_gate_fisher_from_trace(
        fit_trace,
        catalog=parameter_catalog,
        normalization="mean_over_prompts",
    )
    config = FisherPromptClusterConfig(
        model_fingerprint=fit_trace.provenance.source_model_fingerprint,
        calibration_split_sha256=(
            fit_trace.provenance.calibration_split_sha256
        ),
        objective_sha256=fit_trace.provenance.objective_sha256,
        source_fisher_coupling_sha256=fisher.artifact_sha256,
        layer_specs=fit_trace.layer_specs,
        mode_catalog=fisher.fisher_ranked_mode_catalog(),
        cluster_count=cluster_count,
        max_iterations=max_iterations,
        tolerance=tolerance,
        mode_chunk_size=mode_chunk_size,
    )
    plan = build_fisher_prompt_clusters(fit_trace.prompt_effects, config)
    fragment_plan, selection = select_top_fisher_layer_cluster(
        fisher,
        plan,
        minimum_fragment_modes=minimum_fragment_modes,
    )
    return fisher, plan, fragment_plan, selection


@dataclass(frozen=True, slots=True)
class LayerFragmentRows:
    """Ephemeral token rows; this object is never part of a saved artifact."""

    inputs: Tensor
    contributions: Tensor
    fisher_weights: Tensor
    sequences: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.inputs, Tensor)
            or not isinstance(self.contributions, Tensor)
            or not isinstance(self.fisher_weights, Tensor)
            or self.inputs.ndim != 2
            or self.contributions.ndim != 2
            or self.fisher_weights.ndim != 1
            or self.inputs.shape[0] != self.contributions.shape[0]
            or self.inputs.shape[0] != self.fisher_weights.shape[0]
            or self.inputs.shape[0] <= 0
            or not self.inputs.is_floating_point()
            or not self.contributions.is_floating_point()
            or not self.fisher_weights.is_floating_point()
        ):
            raise ValueError("layer-fragment row tensors are inconsistent")
        for name in ("inputs", "contributions", "fisher_weights"):
            value = getattr(self, name).detach().to(
                device="cpu",
                dtype=torch.float64,
            ).contiguous()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if bool((self.fisher_weights < 0).any()) or float(
            self.fisher_weights.sum().item()
        ) <= 0.0:
            raise ValueError("Fisher row weights must have positive total mass")
        if type(self.sequences) is not int or self.sequences <= 0:
            raise ValueError("sequences must be positive")


def collect_layer_fragment_rows(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    input_site: str,
    down_input_site: str,
    mode_indices: tuple[int, ...],
    down_projection_weight: Tensor,
) -> LayerFragmentRows:
    """Collect X, native cluster residual Y, and token Fisher weights."""

    if (
        not input_site
        or not down_input_site
        or input_site == down_input_site
        or not mode_indices
        or mode_indices != tuple(sorted(set(mode_indices)))
    ):
        raise ValueError("layer-fragment site/mode declaration is invalid")
    if (
        not isinstance(down_projection_weight, Tensor)
        or down_projection_weight.ndim != 2
        or not down_projection_weight.is_floating_point()
    ):
        raise ValueError("down_projection_weight must be a floating matrix")
    down = down_projection_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    index = torch.tensor(mode_indices, dtype=torch.long)
    if int(index.max().item()) >= down.shape[1]:
        raise ValueError("selected modes exceed the down projection width")
    selected_down = down.index_select(1, index).contiguous()

    inputs: list[Tensor] = []
    contributions: list[Tensor] = []
    fisher_weights: list[Tensor] = []
    sequences = 0
    iterator = iter(rows)
    try:
        for row in iterator:
            if set(row.activations) != {input_site, down_input_site}:
                raise ValueError(
                    "fragment row sites must equal input and down-input sites"
                )
            x = row.activations[input_site].to(dtype=torch.float64)
            z = row.activations[down_input_site].to(dtype=torch.float64)
            gradient = row.score_gradients[down_input_site].to(
                dtype=torch.float64
            )
            if (
                x.shape[0] != z.shape[0]
                or z.shape != gradient.shape
                or z.shape[1] != down.shape[1]
            ):
                raise ValueError("fragment activation row shapes disagree")
            selected_z = z.index_select(1, index)
            selected_gradient = gradient.index_select(1, index)
            inputs.append(x)
            contributions.append(selected_z @ selected_down.T)
            fisher_weights.append(
                (selected_z * selected_gradient).square().sum(dim=1)
            )
            sequences += 1
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if not inputs:
        raise ValueError("layer-fragment row stream cannot be empty")
    return LayerFragmentRows(
        inputs=torch.cat(inputs, dim=0),
        contributions=torch.cat(contributions, dim=0),
        fisher_weights=torch.cat(fisher_weights, dim=0),
        sequences=sequences,
    )


def _dense_metrics(
    target: Tensor,
    prediction: Tensor,
    weights: Tensor,
) -> dict[str, float | int]:
    target = target.to(dtype=torch.float64)
    prediction = prediction.to(dtype=torch.float64)
    weights = weights.to(dtype=torch.float64)
    error = prediction - target
    row_mse = error.square().mean(dim=1)
    target_row_energy = target.square().mean(dim=1)
    weight_sum = weights.sum()
    mse = float(row_mse.mean().item())
    weighted_mse = float((weights * row_mse).sum().div(weight_sum).item())
    target_rms = math.sqrt(float(target_row_energy.mean().item()))
    weighted_target_rms = math.sqrt(
        float((weights * target_row_energy).sum().div(weight_sum).item())
    )
    rmse = math.sqrt(mse)
    weighted_rmse = math.sqrt(weighted_mse)
    return {
        "observations": target.shape[0],
        "mse": mse,
        "nrmse": rmse / max(target_rms, 1e-30),
        "weighted_mse": weighted_mse,
        "weighted_nrmse": (
            weighted_rmse / max(weighted_target_rms, 1e-30)
        ),
        "max_abs_error": float(error.abs().max().item()),
        "target_rms": target_rms,
        "weighted_target_rms": weighted_target_rms,
    }


@dataclass(frozen=True, slots=True)
class FittedModalGeneratorPilot:
    computational_modes: ComputationalModeRateCurve
    modal_generators: ModalGeneratorRateCurve
    lowering: ModalGeneratorLowering
    graph_plan: ModalGeneratorGraphPlan
    dense_fused_executable_plan: ModalGeneratorPlan
    dense_generator_rate_curve: tuple[Mapping[str, object], ...]

    @property
    def executable_plan(self) -> ModalGeneratorPlan:
        """Compatibility alias for the separately fused dense comparison."""

        return self.dense_fused_executable_plan


def fit_layer_cluster_modal_generator(
    fit_rows: LayerFragmentRows,
    eval_rows: LayerFragmentRows,
    *,
    selection: ParameterClusterLayerFragment,
    source_model_sha256: str,
    parameter_catalog_sha256: str,
    fisher_coupling_sha256: str,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    fit_split_sha256: str,
    eval_split_sha256: str,
    input_site: str,
    output_site: str,
    mode_ranks: Sequence[int],
    selected_mode_rank: int,
    generator_ranks: Sequence[int],
    selected_generator_rank: int,
    ridge: float = 0.0,
) -> FittedModalGeneratorPilot:
    """Fit both explicit ladders, then fold the selected decoder into runtime."""

    if fit_rows.inputs.shape[1] != eval_rows.inputs.shape[1]:
        raise ValueError("fit/eval input widths differ")
    if fit_rows.contributions.shape[1] != eval_rows.contributions.shape[1]:
        raise ValueError("fit/eval residual widths differ")
    mode_ranks = _positive_int_tuple(mode_ranks, label="mode_ranks")
    generator_ranks = _positive_int_tuple(
        generator_ranks,
        label="generator_ranks",
    )
    if not isinstance(fragment_plan, ParameterClusterLayerFragmentPlan):
        raise TypeError(
            "fragment_plan must be ParameterClusterLayerFragmentPlan"
        )
    fragment_plan.validate_integrity()
    if selection.artifact_sha256 not in {
        fragment.artifact_sha256 for fragment in fragment_plan.fragments
    }:
        raise ValueError("selection is not a member of fragment_plan")
    if (
        input_site != selection.input_site
        or output_site != selection.output_site
    ):
        raise ValueError(
            "runtime generator sites do not match the selected fragment"
        )
    if selected_mode_rank not in mode_ranks:
        raise ValueError("selected_mode_rank must be in mode_ranks")
    if selected_generator_rank not in generator_ranks:
        raise ValueError("selected_generator_rank must be in generator_ranks")
    mode_binding = ComputationalModeBinding.create(
        mode_set_id=selection.fragment_id,
        source_kind="parameter_cluster",
        output_site=output_site,
        source_model_sha256=source_model_sha256,
        parameter_catalog_sha256=parameter_catalog_sha256,
        fisher_coupling_sha256=fisher_coupling_sha256,
        parameter_cluster_sha256=selection.artifact_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
    )
    mode_curve = fit_computational_mode_rate_curve(
        fit_rows.contributions,
        fit_rows.fisher_weights,
        eval_rows.contributions,
        eval_rows.fisher_weights,
        mode_ranks,
        binding=mode_binding,
        selection_rule="fixed_rank",
        selected_rank=selected_mode_rank,
    )
    basis = mode_curve.selected_basis
    if basis is None:
        raise RuntimeError("fixed computational-mode rank was not selected")

    input_catalog_sha256 = natural_mlp_input_catalog_sha256(
        source_model_sha256=source_model_sha256,
        input_site=input_site,
        input_width=fit_rows.inputs.shape[1],
    )
    if input_catalog_sha256 != selection.input_catalog_sha256:
        raise ValueError(
            "runtime generator input catalog does not match the fragment"
        )
    generator_binding = ModalGeneratorBinding.create(
        generator_id=(
            f"gemma3.layer-{selection.layer_ordinal}."
            f"cluster-{selection.cluster_id}.modal-generator"
        ),
        input_kind="native_layer_input",
        input_site=input_site,
        output_site=output_site,
        source_model_sha256=source_model_sha256,
        input_catalog_sha256=input_catalog_sha256,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=fragment_plan.artifact_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=fisher_coupling_sha256,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=selection.artifact_sha256,
    )
    fit_coordinates = basis.encode(fit_rows.contributions)
    eval_coordinates = basis.encode(eval_rows.contributions)
    generator_curve = fit_modal_generator_rate_curve(
        fit_rows.inputs,
        fit_coordinates,
        fit_rows.fisher_weights,
        eval_rows.inputs,
        eval_coordinates,
        generator_ranks,
        binding=generator_binding,
        fisher_weights_eval=eval_rows.fisher_weights,
        fit_intercept=True,
        ridge=ridge,
        bias_mac_policy="matrix_multiplies_only",
        selection_rule="fixed_rank",
        selected_rank=selected_generator_rank,
    )
    selected_coordinate_plan = generator_curve.selected_plan
    if selected_coordinate_plan is None:
        raise RuntimeError("fixed generator rank was not selected")
    lowering = lower_coordinate_modal_generator(
        selected_coordinate_plan,
        basis,
        fragment_plan,
    )
    graph_node = lowering.to_graph_node(
        name=f"{selected_coordinate_plan.binding.generator_id}.graph-node",
        causal_order=selection.layer_ordinal,
    )
    graph_plan = ModalGeneratorGraphPlan(
        model_fingerprint=source_model_sha256,
        parameter_cluster_plan_sha256=fragment_plan.artifact_sha256,
        nodes=(graph_node,),
        interactions=(),
    )

    dense_curve: list[Mapping[str, object]] = []
    for point in generator_curve.points:
        fit_prediction = basis.decode(point.plan.apply(fit_rows.inputs))
        eval_prediction = basis.decode(point.plan.apply(eval_rows.inputs))
        dense_curve.append(
            {
                "generator_rank": point.rank,
                "computational_mode_rank": basis.rank,
                "fit": _dense_metrics(
                    fit_rows.contributions,
                    fit_prediction,
                    fit_rows.fisher_weights,
                ),
                "eval": _dense_metrics(
                    eval_rows.contributions,
                    eval_prediction,
                    eval_rows.fisher_weights,
                ),
                "coordinate_plan_sha256": point.plan.artifact_sha256,
            }
        )
    return FittedModalGeneratorPilot(
        computational_modes=mode_curve,
        modal_generators=generator_curve,
        lowering=lowering,
        graph_plan=graph_plan,
        dense_fused_executable_plan=lowering.fused_residual_plan,
        dense_generator_rate_curve=tuple(dense_curve),
    )


def build_single_node_modal_compiler_pipeline(
    *,
    fit_prompt_trace: PromptModeTrace,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    fisher_coupling: GroupedVirtualGateFisher,
    parameter_clusters: FisherPromptClusterPlan,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    selection: ParameterClusterLayerFragment,
    fitted: FittedModalGeneratorPilot,
) -> tuple[ModalSourceReplacementAccounting, ModalCompilerPipeline]:
    """Bind the fitted pilot to exact native accounting and a strict manifest."""

    if not isinstance(fitted, FittedModalGeneratorPilot):
        raise TypeError("fitted must be a FittedModalGeneratorPilot")
    if (
        selection.artifact_sha256
        != fitted.lowering.selected_fragment_sha256
    ):
        raise ValueError("selected fragment does not match fitted lowering")
    if len(fitted.graph_plan.nodes) != 1 or (
        fitted.graph_plan.interactions
    ):
        raise ValueError("development pilot must contain one edgeless graph node")
    node = fitted.graph_plan.nodes[0]
    if (
        node.weights.artifact_sha256
        != fitted.lowering.graph_weights.artifact_sha256
    ):
        raise ValueError("fitted graph node does not match its lowering")
    accounting = build_modal_source_replacement_accounting(
        parameter_catalog,
        fragment_plan,
        (selection.fragment_id,),
    )
    if (
        accounting.source_parameter_count
        != selection.native_parameter_count
    ):
        raise RuntimeError("fragment and catalog source accounting disagree")
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=fit_prompt_trace,
        parameter_catalog=parameter_catalog,
        grouped_fisher=fisher_coupling,
        fisher_clusters=parameter_clusters,
        parameter_cluster_fragments=fragment_plan,
        lowerings_by_node={node.name: fitted.lowering},
        graph_plan=fitted.graph_plan,
        interaction_selection=None,
        source_replacement_accounting=accounting,
    )
    if (
        pipeline.graph_parameter_count
        != fitted.graph_plan.parameter_count
        or pipeline.source_parameter_count
        != selection.native_parameter_count
    ):
        raise RuntimeError("modal compiler resource accounting drifted")
    return accounting, pipeline


def _model_logits(output: object) -> Tensor:
    logits = (
        output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise TypeError("model output must expose [batch, sequence, vocab] logits")
    return logits


def _selected_logits_and_targets(
    logits: Tensor,
    batch: CalibrationBatch,
) -> tuple[Tensor, Tensor]:
    targets = batch.targets.to(device=logits.device)
    selected = targets != -100
    if not bool(selected.any()):
        raise ValueError("evaluation batch has no supervised tokens")
    return (
        logits[selected].detach().to(device="cpu", dtype=torch.float32),
        targets[selected].detach().to(device="cpu", dtype=torch.long),
    )


def _native_nll(logits: Tensor, targets: Tensor) -> float:
    return float(
        F.cross_entropy(logits, targets, reduction="sum").double().item()
    )


def _candidate_comparison(
    native_logits: Tensor,
    candidate_logits: Tensor,
    targets: Tensor,
    *,
    vocabulary_chunk_size: int = 16384,
) -> dict[str, float | int]:
    if candidate_logits.shape != native_logits.shape:
        raise ValueError("native and candidate supervised logits differ")
    native_lse = torch.logsumexp(native_logits, dim=-1)
    candidate_lse = torch.logsumexp(candidate_logits, dim=-1)
    row = torch.arange(targets.shape[0])
    nll = -(
        candidate_logits[row, targets] - candidate_lse
    ).double().sum()
    top1_matches = int(
        (
            candidate_logits.argmax(dim=-1)
            == native_logits.argmax(dim=-1)
        ).sum().item()
    )
    kl_sum = 0.0
    for start in range(0, native_logits.shape[1], vocabulary_chunk_size):
        stop = min(start + vocabulary_chunk_size, native_logits.shape[1])
        native_log_probability = (
            native_logits[:, start:stop] - native_lse[:, None]
        ).double()
        candidate_log_probability = (
            candidate_logits[:, start:stop] - candidate_lse[:, None]
        ).double()
        kl_sum += float(
            (
                native_log_probability.exp()
                * (native_log_probability - candidate_log_probability)
            ).sum().item()
        )
    return {
        "nll_sum": float(nll.item()),
        "native_to_candidate_kl_sum": max(kl_sum, 0.0),
        "top1_matches": top1_matches,
        "tokens": targets.numel(),
    }


def _execution_accounting(
    execution: Gemma3ModalGeneratorModelExecution,
) -> dict[str, int | str]:
    return {
        "replacement_scope": execution.replacement_scope,
        "replaced_layer_count": execution.replaced_layer_count,
        "removed_mode_count": execution.removed_mode_count,
        "source_whole_model_learned_parameters": (
            execution.source_whole_model_learned_parameters
        ),
        "candidate_whole_model_learned_parameters": (
            execution.candidate_whole_model_learned_parameters
        ),
        "native_removed_learned_parameters": (
            execution.native_removed_learned_parameters
        ),
        "modal_generator_learned_parameters": (
            execution.modal_generator_learned_parameters
        ),
        "net_stored_parameter_savings": (
            execution.net_stored_parameter_savings
        ),
    }


def _graph_execution_accounting(
    execution: Gemma3ModalGeneratorGraphExecution,
) -> dict[str, int | str]:
    return {
        "replacement_scope": execution.replacement_scope,
        "replaced_layer_count": execution.replaced_layer_count,
        "graph_node_count": execution.graph_node_count,
        "fragment_count": execution.fragment_count,
        "removed_mode_count": execution.removed_mode_count,
        "source_whole_model_learned_parameters": (
            execution.source_whole_model_learned_parameters
        ),
        "candidate_whole_model_learned_parameters": (
            execution.candidate_whole_model_learned_parameters
        ),
        "native_removed_learned_parameters": (
            execution.native_removed_learned_parameters
        ),
        "modal_graph_learned_parameters": (
            execution.modal_graph_learned_parameters
        ),
        "net_stored_parameter_savings": (
            execution.net_stored_parameter_savings
        ),
        "graph_runtime_storage": execution.graph_runtime_storage,
    }


def evaluate_modal_generator_graph_conditions(
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    """Evaluate the incremental graph traversal and matched graph deletion."""

    native_nll_sum = 0.0
    condition_totals: dict[str, dict[str, float | int]] = {
        condition: {
            "nll_sum": 0.0,
            "native_to_candidate_kl_sum": 0.0,
            "top1_matches": 0,
            "logical_linear_macs_native_removed": 0,
            "logical_modal_graph_macs": 0,
            "logical_executed_modal_graph_macs": 0,
            "logical_modal_graph_additions": 0,
            "logical_executed_modal_graph_additions": 0,
            "net_logical_macs_saved": 0,
            "executed_peak_live_modal_width": 0,
        }
        for condition in ("generated", "deletion")
    }
    supervised_tokens = 0
    logical_valid_tokens = 0
    accounting: dict[str, int | str] | None = None
    expected_traversal = executor.graph_plan.traversal_order
    native_model = adapter.module
    for batch in batches:
        call_inputs: dict[str, object] = dict(batch.model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        with torch.no_grad():
            native_output = native_model(**call_inputs)
        native_logits, targets = _selected_logits_and_targets(
            _model_logits(native_output),
            batch,
        )
        del native_output
        native_nll_sum += _native_nll(native_logits, targets)
        supervised_tokens += targets.numel()

        batch_removed_macs: int | None = None
        batch_potential_graph_macs: int | None = None
        for condition in ("generated", "deletion"):
            with torch.no_grad():
                execution = executor.run(
                    batch.model_inputs,
                    condition=condition,
                )
            actual_traversal = execution.graph_execution.traversal_order
            if condition == "generated":
                if actual_traversal != expected_traversal:
                    raise RuntimeError("generated graph traversal order drifted")
                logical_valid_tokens += execution.valid_tokens
            elif actual_traversal:
                raise RuntimeError("deletion condition traversed modal nodes")
            candidate_logits, candidate_targets = (
                _selected_logits_and_targets(
                    _model_logits(execution.model_output),
                    batch,
                )
            )
            if not torch.equal(targets, candidate_targets):
                raise RuntimeError("candidate evaluation targets drifted")
            comparison = _candidate_comparison(
                native_logits,
                candidate_logits,
                targets,
            )
            totals = condition_totals[condition]
            for name in (
                "nll_sum",
                "native_to_candidate_kl_sum",
                "top1_matches",
            ):
                totals[name] += comparison[name]

            current = _graph_execution_accounting(execution)
            if accounting is None:
                accounting = current
            elif current != accounting:
                raise RuntimeError(
                    "graph executor parameter accounting changed"
                )
            if (
                batch_removed_macs is not None
                and batch_removed_macs
                != execution.logical_linear_macs_native_removed
            ):
                raise RuntimeError("graph conditions removed different native MACs")
            if (
                batch_potential_graph_macs is not None
                and batch_potential_graph_macs
                != execution.logical_modal_graph_macs
            ):
                raise RuntimeError("graph conditions declared different graph MACs")
            batch_removed_macs = (
                execution.logical_linear_macs_native_removed
            )
            batch_potential_graph_macs = execution.logical_modal_graph_macs
            totals["logical_linear_macs_native_removed"] += (
                execution.logical_linear_macs_native_removed
            )
            totals["logical_modal_graph_macs"] += (
                execution.logical_modal_graph_macs
            )
            totals["logical_executed_modal_graph_macs"] += (
                execution.logical_executed_modal_graph_macs
            )
            totals["logical_modal_graph_additions"] += (
                execution.logical_modal_graph_additions
            )
            totals["logical_executed_modal_graph_additions"] += (
                execution.logical_executed_modal_graph_additions
            )
            totals["net_logical_macs_saved"] += (
                execution.net_logical_macs_saved
            )
            totals["executed_peak_live_modal_width"] = max(
                int(totals["executed_peak_live_modal_width"]),
                execution.peak_live_modal_width,
            )
            del candidate_logits, execution
        del native_logits
    if supervised_tokens <= 0 or accounting is None:
        raise ValueError("evaluation stream cannot be empty")

    native_nll = native_nll_sum / supervised_tokens
    conditions: dict[str, object] = {}
    for condition, totals in condition_totals.items():
        nll = float(totals["nll_sum"]) / supervised_tokens
        conditions[condition] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": (
                float(totals["native_to_candidate_kl_sum"])
                / supervised_tokens
            ),
            "top1_agreement_to_native": (
                int(totals["top1_matches"]) / supervised_tokens
            ),
        }
    generated_macs = condition_totals["generated"]
    deletion_macs = condition_totals["deletion"]
    return {
        "execution_path": "incremental_modal_generator_graph_traversal",
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": logical_valid_tokens,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
        "graph": {
            "node_count": len(executor.graph_plan.nodes),
            "interaction_count": len(executor.graph_plan.interactions),
            "traversal_order": expected_traversal,
        },
        "resource_accounting": {
            **accounting,
            "planned_peak_live_modal_width": (
                executor.peak_live_modal_width
            ),
            "generated": {
                "logical_linear_macs_native_removed": generated_macs[
                    "logical_linear_macs_native_removed"
                ],
                "logical_modal_graph_macs": generated_macs[
                    "logical_modal_graph_macs"
                ],
                "logical_executed_modal_graph_macs": generated_macs[
                    "logical_executed_modal_graph_macs"
                ],
                "logical_modal_graph_additions": generated_macs[
                    "logical_modal_graph_additions"
                ],
                "logical_executed_modal_graph_additions": generated_macs[
                    "logical_executed_modal_graph_additions"
                ],
                "executed_peak_live_modal_width": generated_macs[
                    "executed_peak_live_modal_width"
                ],
                "net_logical_macs_saved": generated_macs[
                    "net_logical_macs_saved"
                ],
            },
            "deletion": {
                "logical_linear_macs_native_removed": deletion_macs[
                    "logical_linear_macs_native_removed"
                ],
                "logical_modal_graph_macs": deletion_macs[
                    "logical_modal_graph_macs"
                ],
                "logical_executed_modal_graph_macs": deletion_macs[
                    "logical_executed_modal_graph_macs"
                ],
                "logical_modal_graph_additions": deletion_macs[
                    "logical_modal_graph_additions"
                ],
                "logical_executed_modal_graph_additions": deletion_macs[
                    "logical_executed_modal_graph_additions"
                ],
                "executed_peak_live_modal_width": deletion_macs[
                    "executed_peak_live_modal_width"
                ],
                "net_logical_macs_saved": deletion_macs[
                    "net_logical_macs_saved"
                ],
            },
            "parameter_savings_positive": (
                int(accounting["net_stored_parameter_savings"]) > 0
            ),
            "logical_mac_savings_positive_generated": (
                int(generated_macs["net_logical_macs_saved"]) > 0
            ),
            "latency_or_kernel_speed_claim": False,
        },
    }


def evaluate_modal_generator_conditions(
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorExecutor,
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    """Evaluate the separately fused dense optimization comparison."""

    native_nll_sum = 0.0
    condition_totals = {
        condition: {
            "nll_sum": 0.0,
            "native_to_candidate_kl_sum": 0.0,
            "top1_matches": 0,
            "logical_linear_macs_native_removed": 0,
            "logical_modal_generator_macs": 0,
            "logical_executed_modal_generator_macs": 0,
            "logical_modal_generator_bias_additions": 0,
            "logical_executed_modal_generator_bias_additions": 0,
            "net_logical_macs_saved": 0,
        }
        for condition in ("generated", "matched_deletion")
    }
    supervised_tokens = 0
    logical_valid_tokens = 0
    accounting: dict[str, int | str] | None = None
    native_model = adapter.module
    for batch in batches:
        call_inputs: dict[str, object] = dict(batch.model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        with torch.no_grad():
            native_output = native_model(**call_inputs)
        native_logits, targets = _selected_logits_and_targets(
            _model_logits(native_output),
            batch,
        )
        del native_output
        native_nll_sum += _native_nll(native_logits, targets)
        supervised_tokens += targets.numel()

        for condition in ("generated", "matched_deletion"):
            with torch.no_grad():
                execution = executor.run(
                    batch.model_inputs,
                    condition=condition,
                )
            candidate_logits, candidate_targets = (
                _selected_logits_and_targets(
                    _model_logits(execution.model_output),
                    batch,
                )
            )
            if not torch.equal(targets, candidate_targets):
                raise RuntimeError("candidate evaluation targets drifted")
            comparison = _candidate_comparison(
                native_logits,
                candidate_logits,
                targets,
            )
            for name in (
                "nll_sum",
                "native_to_candidate_kl_sum",
                "top1_matches",
            ):
                condition_totals[condition][name] += comparison[name]
            current = _execution_accounting(execution)
            if accounting is None:
                accounting = current
            elif current != accounting:
                raise RuntimeError(
                    "executor parameter accounting changed across batches"
                )
            if condition == "generated":
                logical_valid_tokens += execution.valid_tokens
            condition_totals[condition][
                "logical_linear_macs_native_removed"
            ] += execution.logical_linear_macs_native_removed
            condition_totals[condition][
                "logical_modal_generator_macs"
            ] += execution.logical_modal_generator_macs
            condition_totals[condition][
                "logical_executed_modal_generator_macs"
            ] += execution.logical_executed_modal_generator_macs
            condition_totals[condition][
                "logical_modal_generator_bias_additions"
            ] += execution.logical_modal_generator_bias_additions
            condition_totals[condition][
                "logical_executed_modal_generator_bias_additions"
            ] += execution.logical_executed_modal_generator_bias_additions
            condition_totals[condition][
                "net_logical_macs_saved"
            ] += execution.net_logical_macs_saved
            del candidate_logits, execution
        del native_logits
    if supervised_tokens <= 0 or accounting is None:
        raise ValueError("evaluation stream cannot be empty")

    native_nll = native_nll_sum / supervised_tokens
    conditions: dict[str, object] = {}
    for condition, totals in condition_totals.items():
        nll = float(totals["nll_sum"]) / supervised_tokens
        conditions[condition] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": (
                float(totals["native_to_candidate_kl_sum"])
                / supervised_tokens
            ),
            "top1_agreement_to_native": (
                int(totals["top1_matches"]) / supervised_tokens
            ),
        }
    generated_totals = condition_totals["generated"]
    deletion_totals = condition_totals["matched_deletion"]
    return {
        "execution_path": "dense_fused_generator_optimization_comparison",
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": logical_valid_tokens,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
        "resource_accounting": {
            **accounting,
            "logical_linear_macs_native_removed": generated_totals[
                "logical_linear_macs_native_removed"
            ],
            "logical_modal_generator_macs": generated_totals[
                "logical_modal_generator_macs"
            ],
            "logical_executed_modal_generator_macs": generated_totals[
                "logical_executed_modal_generator_macs"
            ],
            "logical_modal_generator_bias_additions": generated_totals[
                "logical_modal_generator_bias_additions"
            ],
            "logical_executed_modal_generator_bias_additions": (
                generated_totals[
                    "logical_executed_modal_generator_bias_additions"
                ]
            ),
            "net_logical_macs_saved": generated_totals[
                "net_logical_macs_saved"
            ],
            "matched_deletion": {
                "logical_linear_macs_native_removed": deletion_totals[
                    "logical_linear_macs_native_removed"
                ],
                "logical_modal_generator_macs": deletion_totals[
                    "logical_modal_generator_macs"
                ],
                "logical_executed_modal_generator_macs": deletion_totals[
                    "logical_executed_modal_generator_macs"
                ],
                "logical_modal_generator_bias_additions": deletion_totals[
                    "logical_modal_generator_bias_additions"
                ],
                "logical_executed_modal_generator_bias_additions": (
                    deletion_totals[
                        "logical_executed_modal_generator_bias_additions"
                    ]
                ),
                "net_logical_macs_saved": deletion_totals[
                    "net_logical_macs_saved"
                ],
            },
            "parameter_savings_positive": (
                int(accounting["net_stored_parameter_savings"]) > 0
            ),
            "logical_mac_savings_positive": (
                int(generated_totals["net_logical_macs_saved"]) > 0
            ),
            "latency_or_kernel_speed_claim": False,
        },
    }


def _layer_runtime_sites(
    adapter: Gemma3CausalLMAdapter,
    layer_ordinal: int,
) -> tuple[str, str, nn.Module, Tensor]:
    layer = adapter.layers[layer_ordinal]
    transformer = layer.transformer
    if transformer is None or transformer.feed_forward is None:
        raise ValueError("selected Gemma layer has no structured MLP")
    stages = tuple(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    if len(stages) != 1:
        raise ValueError("selected Gemma layer has no unique MLP residual stage")
    source_layer = adapter.source_module(layer.id)
    mlp = getattr(source_layer, "mlp", None)
    down = getattr(mlp, "down_proj", None)
    weight = getattr(down, "weight", None)
    if not isinstance(mlp, nn.Module) or not isinstance(weight, Tensor):
        raise TypeError("selected Gemma layer does not expose an MLP down weight")
    return (
        stages[0].normalized_input_site,
        stages[0].operator_output_site,
        mlp,
        weight,
    )


def _collect_live_fragment_rows(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    leaf_activation_site: str,
    input_site: str,
    down_input_site: str,
    mode_indices: tuple[int, ...],
    down_projection_weight: Tensor,
    row_factory: ActivationRowFactory = iter_activation_score_gradient_rows,
) -> LayerFragmentRows:
    sites = (input_site, down_input_site)
    requested = tuple(dict.fromkeys((*sites, leaf_activation_site)))
    raw_rows = row_factory(
        adapter,
        batches,
        activation_names=requested,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=leaf_activation_site,
        accumulation_dtype=torch.float64,
    )
    return collect_layer_fragment_rows(
        _select_row_sites(raw_rows, sites),
        input_site=input_site,
        down_input_site=down_input_site,
        mode_indices=mode_indices,
        down_projection_weight=down_projection_weight,
    )


def _safe_tokenized_stream_metadata(
    value: Mapping[str, object],
) -> dict[str, object]:
    examples = value.get("examples")
    if not isinstance(examples, list):
        raise ValueError("tokenized stream examples are invalid")
    valid_tokens = tuple(int(row["valid_tokens"]) for row in examples)
    supervised = tuple(
        int(row["supervised_positions"]) for row in examples
    )
    content_sha256s = tuple(
        _require_sha256(
            row.get("content_sha256")
            if isinstance(row, Mapping)
            else None,
            label="tokenized example content_sha256",
        )
        for row in examples
    )
    if len(content_sha256s) != len(set(content_sha256s)):
        raise ValueError("tokenized stream contains duplicate content hashes")
    return {
        "schema": value.get("schema"),
        "format_version": value.get("format_version"),
        "split": value.get("split"),
        "batches": value.get("batches"),
        "sequences": value.get("sequences"),
        "serialized_sha256": value.get("serialized_sha256"),
        "source_prompt_sha256": tuple(
            value.get("source_prompt_sha256", ())
        ),
        "content_sha256": content_sha256s,
        "valid_tokens": {
            "minimum": min(valid_tokens),
            "maximum": max(valid_tokens),
            "total": sum(valid_tokens),
        },
        "supervised_positions": {
            "minimum": min(supervised),
            "maximum": max(supervised),
            "total": sum(supervised),
        },
        "contains_prompt_text": False,
        "contains_token_ids": False,
    }


def _tokenized_content_disjointness(
    fit_stream: Mapping[str, object],
    eval_stream: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed when distinct prompts collapse to shared token content."""

    def hashes(
        stream: Mapping[str, object],
        *,
        label: str,
    ) -> tuple[str, ...]:
        examples = stream.get("examples")
        if isinstance(examples, list):
            values = tuple(
                _require_sha256(
                    example.get("content_sha256")
                    if isinstance(example, Mapping)
                    else None,
                    label=f"{label} content_sha256",
                )
                for example in examples
            )
        else:
            raw_values = stream.get("content_sha256")
            if (
                isinstance(raw_values, (str, bytes))
                or not isinstance(raw_values, Sequence)
            ):
                raise ValueError(
                    f"{label} tokenized content hashes are unavailable"
                )
            values = tuple(
                _require_sha256(
                    value,
                    label=f"{label} content_sha256",
                )
                for value in raw_values
            )
        if not values or len(values) != len(set(values)):
            raise ValueError(
                f"{label} tokenized content hashes are empty or duplicated"
            )
        return values

    fit_hashes = hashes(fit_stream, label="fit")
    eval_hashes = hashes(eval_stream, label="development evaluation")
    overlap = set(fit_hashes) & set(eval_hashes)
    if overlap:
        raise ValueError(
            "tokenized content overlaps between fit and development "
            "evaluation"
        )
    return {
        "fit_content_count": len(fit_hashes),
        "eval_content_count": len(eval_hashes),
        "overlap_count": 0,
        "passed": True,
    }


def _assert_source_safe_artifact(
    value: object,
    *,
    prompt_texts: frozenset[str],
    path: tuple[str, ...] = (),
    require_machine_strings: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_ARTIFACT_KEYS:
                raise RuntimeError(
                    f"artifact contains forbidden field {'.'.join((*path, key_text))}"
                )
            _assert_source_safe_artifact(
                item,
                prompt_texts=prompt_texts,
                path=(*path, key_text),
                require_machine_strings=require_machine_strings,
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_source_safe_artifact(
                item,
                prompt_texts=prompt_texts,
                path=(*path, str(index)),
                require_machine_strings=require_machine_strings,
            )
    elif isinstance(value, str):
        if value in prompt_texts:
            raise RuntimeError("artifact contains raw prompt text")
        if require_machine_strings and any(
            character.isspace()
            or unicodedata.category(character).startswith("C")
            for character in value
        ):
            location = ".".join(path) or "<root>"
            raise RuntimeError(
                "artifact contains non-machine string at "
                f"{location}; free-form text is not allowed"
            )


def _build_json_report(
    payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    report = {
        "schema": GEMMA3_MODAL_GENERATOR_DEV_SCHEMA,
        "format_version": GEMMA3_MODAL_GENERATOR_DEV_FORMAT_VERSION,
        "scientific_status": payload["scientific_status"],
        "model": payload["model"],
        "protocol": payload["protocol"],
        "splits": payload["splits"],
        "selected_layer_cluster": payload["selected_layer_cluster"],
        "rate_distortion": {
            "computational_modes": payload[
                "computational_mode_metadata"
            ],
            "modal_generators": payload["modal_generator_metadata"],
            "dense_fused_optimization_comparison": payload[
                "dense_generator_rate_curve"
            ],
        },
        "lowering": payload["modal_generator_lowering_metadata"],
        "graph": payload["modal_generator_graph_metadata"],
        "source_replacement_accounting": payload[
            "source_replacement_accounting_metadata"
        ],
        "compiler_pipeline": payload[
            "modal_compiler_pipeline_metadata"
        ],
        "evaluation": payload["evaluation"],
        "artifact": {
            "tensor_file": output.name,
            "scientific_payload_sha256": payload[
                "scientific_payload_sha256"
            ],
            "contains_source_model_weights": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_raw_token_rows": False,
            "contains_generator_weights": True,
        },
    }
    return {
        **report,
        "report_sha256": _json_sha256(report, domain=_REPORT_DOMAIN),
    }


def run_gemma3_modal_generator_dev_experiment(
    *,
    fit_export_path: Path | str,
    eval_export_path: Path | str,
    revision: str,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
    minimum_fragment_modes: int = DEFAULT_MINIMUM_FRAGMENT_MODES,
    mode_ranks: Sequence[int] = DEFAULT_MODE_RANKS,
    selected_mode_rank: int = DEFAULT_SELECTED_MODE_RANK,
    generator_ranks: Sequence[int] = DEFAULT_GENERATOR_RANKS,
    selected_generator_rank: int = DEFAULT_SELECTED_GENERATOR_RANK,
    ridge: float = 0.0,
) -> dict[str, object]:
    """Run the complete development pilot without opening a heldout split."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    resolved_output = Path(output)
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    report_path = resolved_output.with_suffix(".json")
    if resolved_output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a modal-generator result")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")

    _progress(
        "preflight: check two self-attested fit-only export declarations "
        "for membership overlap"
    )
    fit_export = load_development_prompt_export(fit_export_path)
    eval_export = load_development_prompt_export(eval_export_path)
    split_policy = validate_development_split_pair(fit_export, eval_export)

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned Gemma checkpoint from local cache")
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
    source_model_sha256 = adapter.model_fingerprint()
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    if model_metadata.get("resolved_commit") != revision:
        raise ValueError("loaded Gemma model does not bind the pinned revision")

    _progress("tokenize: materialize replayable fit/evaluation streams")
    fit_batches, fit_stream = _materialize_split(
        tokenizer,
        fit_export.prompts,
        split_name="modal_generator_development_fit",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    eval_batches, eval_stream = _materialize_split(
        tokenizer,
        eval_export.prompts,
        split_name="modal_generator_development_eval",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    fit_split_sha256 = _require_sha256(
        fit_stream.get("serialized_sha256"),
        label="fit tokenized stream sha256",
    )
    eval_split_sha256 = _require_sha256(
        eval_stream.get("serialized_sha256"),
        label="eval tokenized stream sha256",
    )
    if fit_split_sha256 == eval_split_sha256:
        raise ValueError("tokenized fit/evaluation streams must differ")
    tokenized_content_audit = _tokenized_content_disjointness(
        fit_stream,
        eval_stream,
    )

    layer_specs, leaf_site, _ = _whole_model_layer_specs(adapter)
    objective_sha256 = _objective_sha256()
    natural_layer_specs_list: list[NaturalMLPLayerParameterSpec] = []
    for spec in layer_specs:
        layer_input_site, layer_output_site, _, _ = _layer_runtime_sites(
            adapter,
            spec.layer_ordinal,
        )
        natural_layer_specs_list.append(
            NaturalMLPLayerParameterSpec.from_cross_block_layer_spec(
                spec,
                input_width=adapter.layers[
                    spec.layer_ordinal
                ].residual_width,
                output_width=adapter.layers[
                    spec.layer_ordinal
                ].residual_width,
                input_site=layer_input_site,
                output_site=layer_output_site,
            )
        )
    natural_layer_specs = tuple(natural_layer_specs_list)
    parameter_catalog = build_natural_mlp_parameter_group_catalog(
        model_fingerprint=source_model_sha256,
        layer_specs=natural_layer_specs,
    )

    _progress("trace: collect prompt-conditioned virtual-gate Fisher summaries")
    fit_trace = collect_gemma_prompt_mode_trace(
        adapter,
        fit_batches,
        layer_specs=layer_specs,
        leaf_activation_site=leaf_site,
        split_sha256=fit_split_sha256,
        objective_sha256=objective_sha256,
    )
    eval_trace = collect_gemma_prompt_mode_trace(
        adapter,
        eval_batches,
        layer_specs=layer_specs,
        leaf_activation_site=leaf_site,
        split_sha256=eval_split_sha256,
        objective_sha256=objective_sha256,
    )
    if eval_trace.provenance.source_model_fingerprint != source_model_sha256:
        raise RuntimeError("evaluation trace source model binding drifted")

    _progress("cluster: freeze the top fit-Fisher layer-cluster fragment")
    fisher, clusters, fragment_plan, selection = (
        build_fit_fisher_cluster_pilot(
            fit_trace,
            parameter_catalog=parameter_catalog,
            cluster_count=cluster_count,
            minimum_fragment_modes=minimum_fragment_modes,
        )
    )
    input_site, output_site, _, down_weight = _layer_runtime_sites(
        adapter,
        selection.layer_ordinal,
    )

    _progress("rows: replay only the selected layer fragment")
    fit_fragment = _collect_live_fragment_rows(
        adapter,
        fit_batches,
        leaf_activation_site=leaf_site,
        input_site=input_site,
        down_input_site=selection.activation_site,
        mode_indices=selection.removed_mode_indices,
        down_projection_weight=down_weight,
    )
    eval_fragment = _collect_live_fragment_rows(
        adapter,
        eval_batches,
        leaf_activation_site=leaf_site,
        input_site=input_site,
        down_input_site=selection.activation_site,
        mode_indices=selection.removed_mode_indices,
        down_projection_weight=down_weight,
    )

    _progress("fit: computational-mode and modal-generator rank ladders")
    fitted = fit_layer_cluster_modal_generator(
        fit_fragment,
        eval_fragment,
        selection=selection,
        source_model_sha256=source_model_sha256,
        parameter_catalog_sha256=parameter_catalog.artifact_sha256,
        fisher_coupling_sha256=fisher.artifact_sha256,
        fragment_plan=fragment_plan,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
        input_site=input_site,
        output_site=output_site,
        mode_ranks=mode_ranks,
        selected_mode_rank=selected_mode_rank,
        generator_ranks=generator_ranks,
        selected_generator_rank=selected_generator_rank,
        ridge=ridge,
    )
    source_accounting, compiler_pipeline = (
        build_single_node_modal_compiler_pipeline(
            fit_prompt_trace=fit_trace,
            parameter_catalog=parameter_catalog,
            fisher_coupling=fisher,
            parameter_clusters=clusters,
            fragment_plan=fragment_plan,
            selection=selection,
            fitted=fitted,
        )
    )
    graph_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        compiler_pipeline.graph_plan,
        tuple(node.lowering for node in compiler_pipeline.nodes),
    )
    dense_fused_executor = Gemma3ModalGeneratorExecutor(
        adapter,
        (
            Gemma3ModalGeneratorReplacement.from_lowering(
                fitted.lowering,
            ),
        ),
    )
    _progress("evaluate: incremental graph traversal vs graph deletion")
    graph_evaluation = evaluate_modal_generator_graph_conditions(
        adapter,
        graph_executor,
        eval_batches,
    )
    _progress("evaluate: separately fused dense optimization comparison")
    dense_fused_evaluation = evaluate_modal_generator_conditions(
        adapter,
        dense_fused_executor,
        eval_batches,
    )
    graph_resources = graph_evaluation["resource_accounting"]
    if not isinstance(graph_resources, Mapping):
        raise RuntimeError("graph evaluation resource accounting is invalid")
    if (
        graph_resources["native_removed_learned_parameters"]
        != source_accounting.source_parameter_count
        or graph_resources["modal_graph_learned_parameters"]
        != compiler_pipeline.graph_parameter_count
        or graph_resources["net_stored_parameter_savings"]
        != compiler_pipeline.net_parameter_savings
    ):
        raise RuntimeError(
            "live graph execution and compiler accounting disagree"
        )
    evaluation = {
        "primary_graph_traversal": graph_evaluation,
        "dense_fused_optimization_comparison": dense_fused_evaluation,
        "resource_accounting_paths_are_separate": True,
    }
    if adapter.model_fingerprint() != source_model_sha256:
        raise RuntimeError("modal-generator experiment mutated the source model")

    payload_without_digest: dict[str, object] = {
        "schema": GEMMA3_MODAL_GENERATOR_DEV_SCHEMA,
        "format_version": GEMMA3_MODAL_GENERATOR_DEV_FORMAT_VERSION,
        "scientific_status": {
            "outcome": "development_only_modal_generator_measurement",
            "compression_claim": False,
            "heldout_confirmation": False,
            "clusters_frozen_from_fit_only": True,
            "evaluation_used_for_selection": False,
            "ready_for_calibration_b": False,
            "development_export_provenance": "declared_self_attested",
            "external_export_authentication_claim": False,
            "numerical_extraction_provenance": (
                "caller_declared_self_attested"
            ),
            "split_membership_provenance": "caller_declared_self_attested",
            "numerical_extraction_externally_authenticated": False,
            "split_membership_externally_authenticated": False,
        },
        "model": {
            **model_metadata,
            "adapter_model_fingerprint": source_model_sha256,
        },
        "protocol": {
            "recipe": (
                "weights",
                "fisher_coupling",
                "parameter_clusters",
                "computational_modes",
                "modal_generators",
                "graph_of_generator_interactions",
                "inference_by_graph_traversal",
            ),
            "scope": "top_fit_fisher_single_layer_cluster_pilot",
            "primary_execution_path": (
                "incremental_modal_generator_graph_traversal"
            ),
            "graph_node_count": len(compiler_pipeline.graph_plan.nodes),
            "graph_interaction_count": len(
                compiler_pipeline.graph_plan.interactions
            ),
            "graph_interaction_status": (
                "edgeless_single_node_pilot_no_pairwise_interaction_exists"
            ),
            "graph_traversal_order": (
                compiler_pipeline.graph_plan.traversal_order
            ),
            "dense_fused_path_status": (
                "separate_optimization_comparison_not_primary_graph_result"
            ),
            "cluster_count": cluster_count,
            "minimum_fragment_modes": minimum_fragment_modes,
            "mode_ranks": tuple(mode_ranks),
            "selected_mode_rank": selected_mode_rank,
            "generator_ranks": tuple(generator_ranks),
            "selected_generator_rank": selected_generator_rank,
            "ridge": ridge,
            "max_length": max_length,
            "tokenization_batch_size": tokenization_batch_size,
            "clusters_fit_split_only": True,
            "rank_selection_predeclared": True,
            "export_provenance_assurance": "declared_self_attested",
            "export_provenance_externally_authenticated": False,
            "numerical_extraction_provenance": (
                "caller_declared_self_attested"
            ),
            "split_membership_provenance": "caller_declared_self_attested",
            "tokenized_content_disjoint": True,
            "local_files_only": True,
        },
        "splits": {
            "policy": split_policy,
            "fit_export": fit_export.metadata(),
            "eval_export": eval_export.metadata(),
            "fit_tokenized": _safe_tokenized_stream_metadata(fit_stream),
            "eval_tokenized": _safe_tokenized_stream_metadata(eval_stream),
            "tokenized_content_disjointness": tokenized_content_audit,
        },
        "fit_prompt_trace": fit_trace.state_dict(),
        "eval_prompt_trace": eval_trace.state_dict(),
        "parameter_catalog": parameter_catalog.state_dict(),
        "fisher_coupling": fisher.state_dict(),
        "parameter_clusters": clusters.state_dict(),
        "parameter_cluster_fragments": fragment_plan.state_dict(),
        "selected_layer_cluster": selection.metadata(),
        "computational_modes": fitted.computational_modes.state_dict(),
        "modal_generators": fitted.modal_generators.state_dict(),
        "modal_generator_lowering": fitted.lowering.state_dict(),
        "modal_generator_graph": compiler_pipeline.graph_plan.state_dict(),
        "source_replacement_accounting": (
            source_accounting.state_dict()
        ),
        "modal_compiler_pipeline": compiler_pipeline.state_dict(),
        "dense_fused_executable_generator": (
            fitted.dense_fused_executable_plan.state_dict()
        ),
        "computational_mode_metadata": (
            fitted.computational_modes.metadata()
        ),
        "modal_generator_metadata": fitted.modal_generators.metadata(),
        "modal_generator_lowering_metadata": fitted.lowering.metadata(),
        "modal_generator_graph_metadata": {
            "artifact_sha256": compiler_pipeline.graph_plan.artifact_sha256,
            "node_count": len(compiler_pipeline.graph_plan.nodes),
            "interaction_count": len(
                compiler_pipeline.graph_plan.interactions
            ),
            "traversal_order": compiler_pipeline.graph_plan.traversal_order,
            "parameter_count": (
                compiler_pipeline.graph_plan.parameter_count
            ),
            "macs_per_token": (
                compiler_pipeline.graph_plan.macs_per_token
            ),
        },
        "source_replacement_accounting_metadata": {
            "artifact_sha256": source_accounting.artifact_sha256,
            "fragment_ids": source_accounting.fragment_ids,
            "group_indices": source_accounting.group_indices,
            "source_parameter_count": (
                source_accounting.source_parameter_count
            ),
            "source_macs_per_token": (
                source_accounting.source_macs_per_token
            ),
        },
        "modal_compiler_pipeline_metadata": compiler_pipeline.metadata(),
        "dense_generator_rate_curve": fitted.dense_generator_rate_curve,
        "evaluation": evaluation,
        "contains_source_model_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_token_rows": False,
        "contains_tokenizer_state": False,
        "contains_generator_weights": True,
    }
    all_prompts = frozenset((*fit_export.prompts, *eval_export.prompts))
    _assert_source_safe_artifact(
        payload_without_digest,
        prompt_texts=all_prompts,
    )
    scientific_digest = _payload_sha256(payload_without_digest)
    payload = {
        **payload_without_digest,
        "scientific_payload_sha256": scientific_digest,
    }
    report = _build_json_report(payload, output=resolved_output)
    _assert_source_safe_artifact(report, prompt_texts=all_prompts)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved_output.open("xb") as handle:
            torch.save(payload, handle)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except BaseException:
        if resolved_output.exists() and not report_path.exists():
            resolved_output.unlink()
        raise
    _progress(f"wrote {resolved_output} and {report_path}")
    return report


def load_gemma3_modal_generator_dev_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load the source-free development artifact and nested plans."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("modal-generator development artifact must be a dict")
    if set(raw) != _ARTIFACT_FIELDS:
        raise ValueError("modal-generator artifact top-level fields are invalid")
    if (
        raw.get("schema") != GEMMA3_MODAL_GENERATOR_DEV_SCHEMA
        or raw.get("format_version")
        != GEMMA3_MODAL_GENERATOR_DEV_FORMAT_VERSION
    ):
        raise ValueError("unsupported modal-generator development artifact")
    if (
        raw["contains_source_model_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_token_ids"] is not False
        or raw["contains_raw_token_rows"] is not False
        or raw["contains_tokenizer_state"] is not False
        or raw["contains_generator_weights"] is not True
    ):
        raise ValueError("modal-generator artifact safety flags are invalid")
    try:
        _assert_source_safe_artifact(
            raw,
            prompt_texts=frozenset(),
            require_machine_strings=True,
        )
    except RuntimeError as error:
        raise ValueError(
            f"modal-generator source-safety scan failed: {error}"
        ) from error
    digest = raw.get("scientific_payload_sha256")
    _require_sha256(digest, label="scientific_payload_sha256")
    without_digest = {
        key: value
        for key, value in raw.items()
        if key != "scientific_payload_sha256"
    }
    if _payload_sha256(without_digest) != digest:
        raise ValueError("modal-generator scientific payload hash mismatch")
    fit_trace = PromptModeTrace.from_state_dict(raw["fit_prompt_trace"])
    eval_trace = PromptModeTrace.from_state_dict(raw["eval_prompt_trace"])
    catalog = NaturalMLPParameterGroupCatalog.from_state_dict(
        raw["parameter_catalog"]
    )
    fisher = GroupedVirtualGateFisher.from_state_dict(raw["fisher_coupling"])
    clusters = FisherPromptClusterPlan.from_state_dict(
        raw["parameter_clusters"]
    )
    fragments = ParameterClusterLayerFragmentPlan.from_state_dict(
        raw["parameter_cluster_fragments"]
    )
    mode_curve = ComputationalModeRateCurve.from_state_dict(
        raw["computational_modes"]
    )
    generator_curve = ModalGeneratorRateCurve.from_state_dict(
        raw["modal_generators"]
    )
    lowering = ModalGeneratorLowering.from_state_dict(
        raw["modal_generator_lowering"]
    )
    graph = ModalGeneratorGraphPlan.from_state_dict(
        raw["modal_generator_graph"]
    )
    accounting = ModalSourceReplacementAccounting.from_state_dict(
        raw["source_replacement_accounting"]
    )
    pipeline = ModalCompilerPipeline.from_state_dict(
        raw["modal_compiler_pipeline"]
    )
    dense_fused = ModalGeneratorPlan.from_state_dict(
        raw["dense_fused_executable_generator"]
    )

    selected_basis = mode_curve.selected_basis
    selected_generator = generator_curve.selected_plan
    if selected_basis is None or selected_generator is None:
        raise ValueError("development artifact lacks fixed selected ranks")
    if (
        selected_basis.artifact_sha256
        != lowering.computational_mode_basis.artifact_sha256
        or selected_generator.artifact_sha256
        != lowering.coordinate_generator_plan.artifact_sha256
        or dense_fused.artifact_sha256
        != lowering.fused_residual_plan.artifact_sha256
    ):
        raise ValueError("saved lowering inputs or dense comparison drifted")
    if (
        graph.artifact_sha256 != pipeline.graph_plan.artifact_sha256
        or catalog.artifact_sha256
        != pipeline.parameter_catalog.artifact_sha256
        or fragments.artifact_sha256
        != pipeline.parameter_cluster_fragments.artifact_sha256
        or fisher.artifact_sha256
        != pipeline.grouped_fisher.referenced_artifact_sha256
        or clusters.artifact_sha256
        != pipeline.fisher_clusters.referenced_artifact_sha256
        or pipeline.source_replacement_accounting is None
        or accounting.artifact_sha256
        != pipeline.source_replacement_accounting.artifact_sha256
    ):
        raise ValueError("saved compiler pipeline lineage drifted")
    if (
        len(graph.nodes) != 1
        or graph.interactions
        or len(pipeline.nodes) != 1
        or graph.nodes[0].weights.artifact_sha256
        != lowering.graph_weights.artifact_sha256
        or pipeline.nodes[0].lowering.artifact_sha256
        != lowering.artifact_sha256
    ):
        raise ValueError("development graph is not the selected one-node lowering")
    selected_fragments = tuple(
        fragment
        for fragment in fragments.fragments
        if fragment.artifact_sha256 == lowering.selected_fragment_sha256
    )
    if len(selected_fragments) != 1:
        raise ValueError("saved selected fragment is not unique")
    expected_graph_metadata = {
        "artifact_sha256": graph.artifact_sha256,
        "node_count": len(graph.nodes),
        "interaction_count": len(graph.interactions),
        "traversal_order": graph.traversal_order,
        "parameter_count": graph.parameter_count,
        "macs_per_token": graph.macs_per_token,
    }
    expected_accounting_metadata = {
        "artifact_sha256": accounting.artifact_sha256,
        "fragment_ids": accounting.fragment_ids,
        "group_indices": accounting.group_indices,
        "source_parameter_count": accounting.source_parameter_count,
        "source_macs_per_token": accounting.source_macs_per_token,
    }
    for label, saved, expected in (
        (
            "selected layer cluster",
            raw["selected_layer_cluster"],
            selected_fragments[0].metadata(),
        ),
        (
            "computational mode",
            raw["computational_mode_metadata"],
            mode_curve.metadata(),
        ),
        (
            "modal generator",
            raw["modal_generator_metadata"],
            generator_curve.metadata(),
        ),
        (
            "modal generator lowering",
            raw["modal_generator_lowering_metadata"],
            lowering.metadata(),
        ),
        (
            "modal generator graph",
            raw["modal_generator_graph_metadata"],
            expected_graph_metadata,
        ),
        (
            "source replacement accounting",
            raw["source_replacement_accounting_metadata"],
            expected_accounting_metadata,
        ),
        (
            "modal compiler pipeline",
            raw["modal_compiler_pipeline_metadata"],
            pipeline.metadata(),
        ),
    ):
        if saved != expected:
            raise ValueError(f"saved {label} metadata is inconsistent")
    if (
        fit_trace.provenance.source_model_fingerprint
        != catalog.model_fingerprint
        or eval_trace.provenance.source_model_fingerprint
        != catalog.model_fingerprint
        or fit_trace.provenance.calibration_split_sha256
        != pipeline.fit_split_sha256
        or eval_trace.provenance.calibration_split_sha256
        != pipeline.eval_split_sha256
    ):
        raise ValueError("saved trace and compiler split/model bindings drifted")

    protocol = raw.get("protocol")
    expected_recipe = (
        "weights",
        "fisher_coupling",
        "parameter_clusters",
        "computational_modes",
        "modal_generators",
        "graph_of_generator_interactions",
        "inference_by_graph_traversal",
    )
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("recipe") != expected_recipe
        or protocol.get("primary_execution_path")
        != "incremental_modal_generator_graph_traversal"
        or protocol.get("graph_node_count") != 1
        or protocol.get("graph_interaction_count") != 0
        or protocol.get("graph_traversal_order") != graph.traversal_order
        or protocol.get("numerical_extraction_provenance")
        != "caller_declared_self_attested"
        or protocol.get("split_membership_provenance")
        != "caller_declared_self_attested"
    ):
        raise ValueError("saved modal graph protocol is inconsistent")
    splits = raw.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("saved development split metadata is invalid")
    fit_tokenized = splits.get("fit_tokenized")
    eval_tokenized = splits.get("eval_tokenized")
    if not isinstance(fit_tokenized, Mapping) or not isinstance(
        eval_tokenized,
        Mapping,
    ):
        raise ValueError("saved tokenized stream metadata is invalid")
    content_audit = _tokenized_content_disjointness(
        fit_tokenized,
        eval_tokenized,
    )
    if splits.get("tokenized_content_disjointness") != content_audit:
        raise ValueError("saved tokenized-content audit does not recompute")
    split_policy = splits.get("policy")
    if (
        not isinstance(split_policy, Mapping)
        or split_policy.get("export_provenance_assurance")
        != "declared_self_attested"
        or split_policy.get(
            "export_provenance_externally_authenticated"
        )
        is not False
        or split_policy.get("split_membership_provenance")
        != "caller_declared_self_attested"
        or split_policy.get("split_membership_externally_authenticated")
        is not False
        or split_policy.get("declared_membership_overlap_checked") is not True
    ):
        raise ValueError("saved export provenance assurance is overstated")
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, Mapping) or (
        evaluation.get("resource_accounting_paths_are_separate") is not True
    ):
        raise ValueError("graph and fused evaluations are not separated")
    graph_evaluation = evaluation.get("primary_graph_traversal")
    dense_evaluation = evaluation.get(
        "dense_fused_optimization_comparison"
    )
    if (
        not isinstance(graph_evaluation, Mapping)
        or graph_evaluation.get("execution_path")
        != "incremental_modal_generator_graph_traversal"
        or not isinstance(dense_evaluation, Mapping)
        or dense_evaluation.get("execution_path")
        != "dense_fused_generator_optimization_comparison"
    ):
        raise ValueError("saved evaluation execution paths are invalid")
    graph_resources = graph_evaluation.get("resource_accounting")
    if (
        not isinstance(graph_resources, Mapping)
        or graph_resources.get("native_removed_learned_parameters")
        != accounting.source_parameter_count
        or graph_resources.get("modal_graph_learned_parameters")
        != pipeline.graph_parameter_count
        or graph_resources.get("net_stored_parameter_savings")
        != pipeline.net_parameter_savings
    ):
        raise ValueError("saved graph execution accounting is inconsistent")
    return raw


def _rank_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ranks must be comma-separated integers"
        ) from error
    try:
        return _positive_int_tuple(result, label="ranks")
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local-only development Gemma modal-generator pilot."
        )
    )
    parser.add_argument("--fit-export", type=Path, default=DEFAULT_FIT_EXPORT)
    parser.add_argument("--eval-export", type=Path, default=DEFAULT_EVAL_EXPORT)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
    )
    parser.add_argument(
        "--minimum-fragment-modes",
        type=int,
        default=DEFAULT_MINIMUM_FRAGMENT_MODES,
    )
    parser.add_argument(
        "--mode-ranks",
        type=_rank_list,
        default=DEFAULT_MODE_RANKS,
    )
    parser.add_argument(
        "--selected-mode-rank",
        type=int,
        default=DEFAULT_SELECTED_MODE_RANK,
    )
    parser.add_argument(
        "--generator-ranks",
        type=_rank_list,
        default=DEFAULT_GENERATOR_RANKS,
    )
    parser.add_argument(
        "--selected-generator-rank",
        type=int,
        default=DEFAULT_SELECTED_GENERATOR_RANK,
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_modal_generator_dev_experiment(
        fit_export_path=arguments.fit_export,
        eval_export_path=arguments.eval_export,
        revision=arguments.revision,
        output=arguments.output,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        cluster_count=arguments.cluster_count,
        minimum_fragment_modes=arguments.minimum_fragment_modes,
        mode_ranks=arguments.mode_ranks,
        selected_mode_rank=arguments.selected_mode_rank,
        generator_ranks=arguments.generator_ranks,
        selected_generator_rank=arguments.selected_generator_rank,
        ridge=arguments.ridge,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
