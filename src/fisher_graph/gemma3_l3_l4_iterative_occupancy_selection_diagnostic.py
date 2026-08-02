"""Execute Iteration 5 with pre-open arm selection and exact fresh evidence.

The live boundary is deliberately split:

1. the reusable expanded A-fit panel receives sixteen source forwards and
   sixteen accepted-X4 + lag-B NLL-VJP forwards;
2. each VJP is reduced once to a shared cumulative/EW occupancy fit record;
3. eight family-blocked fits per arm select one arm without opening the fresh
   panel;
4. both arms are fit on all reusable development records and frozen; then
5. a durable claim is created, the fresh panel is opened once, and every
   fresh example receives source, parent, cumulative, and EW forwards.

That is exactly 96 model forwards.  The nonselected arm is measured only to
make the comparison informative; it cannot become eligible after the fresh
panel opens.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .compiler.calibration import CausalLanguageModelNLL
from .gemma3_experiment import (
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
    load_gemma3_graph_organized_svd_candidate,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .gemma3_l3_l4_h4_damping_materialization import (
    load_gemma_h4_damping_materialization,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from .gemma3_l3_l4_h4_damping_selection_panel import (
    load_gemma3_l3_l4_h4_damping_expanded_fit_lineage,
    load_gemma3_l3_l4_h4_damping_selection_panel_artifact,
)
from .gemma3_l3_l4_iterative_conformal_route_analysis import (
    validate_gemma_iterative_conformal_route_report,
)
from .gemma3_l3_l4_iterative_occupancy_development import (
    build_gemma_iterative_occupancy_development_selection,
    validate_gemma_iterative_occupancy_development_selection,
)
from .gemma3_l3_l4_iterative_occupancy_route import (
    CENTERED_CUMULATIVE_OCCUPANCY,
    CENTERED_EW_OCCUPANCY,
    build_gemma_iterative_occupancy_conformal_route_fit_record,
    fit_gemma_iterative_occupancy_conformal_route_fold,
    fit_gemma_iterative_occupancy_conformal_route_full_provider,
)
from .gemma3_l3_l4_iterative_occupancy_selection_analysis import (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
    build_gemma_iterative_occupancy_selection_report,
    validate_gemma_iterative_occupancy_selection_report,
)
from .gemma3_l3_l4_iterative_occupancy_selection_panel import (
    Gemma3L3L4IterativeOccupancySelectionPanelSource,
    claim_gemma3_l3_l4_iterative_occupancy_selection_panel,
    freeze_gemma3_l3_l4_iterative_occupancy_selection_panel,
    load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact,
    materialize_gemma3_l3_l4_iterative_occupancy_selection_panel,
    write_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact,
    write_gemma3_l3_l4_iterative_occupancy_selection_role_input,
)
from .gemma3_l3_l4_iterative_residual_campaign import (
    _gather_logits,
    _observation,
    _panel_manifest,
    _same_source,
    _scalar_payload,
    _source_authority,
    _validate_execution,
    _validate_parent,
)
from .gemma3_l3_l4_iterative_residual_diagnostic import (
    _accepted_x4_provenance,
    _file_sha256,
    _load_factorial_report,
    _mapping,
    _source_code_sha256s,
    _validate_factorial_fit_lineage,
    _validate_factorial_live_lineage,
    _validate_factorial_materialization_lineage,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    gemma3_l3_l4_progressive_a_tokenizer_contract_sha256,
    load_gemma3_l3_l4_progressive_a_fit_role,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "CLI_NAME",
    "DEFAULT_OUTPUT",
    "OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256",
    "PREPARATION_CLI_NAME",
    "build_parser",
    "build_preparation_parser",
    "main",
    "prepare_gemma_iterative_occupancy_selection_panel",
    "preparation_main",
    "run_gemma_iterative_occupancy_selection_diagnostic",
]


CLI_NAME = "fisher-graph-gemma-l3-l4-iterative-occupancy-selection-dev"
PREPARATION_CLI_NAME = (
    "fisher-graph-gemma-l3-l4-iterative-occupancy-selection-prepare"
)
_FACTORIZED_SCOPE = "factorized_refit"
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_DEFAULT_EXPANDED_CORPUS = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
_DEFAULT_EXPANDED_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
_DEFAULT_MATERIALIZATION_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-damping-materialization-v1.report.json"
)
_DEFAULT_FACTORIAL_REPORT = (
    _LOCAL_ROOT / "progressive-a-x4-h4-factorial-fit-v1.report.json"
)
_DEFAULT_PRIOR_ITERATION_REPORT = (
    _LOCAL_ROOT / "progressive-a-iterative-conformal-route-v1.report.json"
)
_DEFAULT_SELECTION_INPUT = (
    _LOCAL_ROOT / "progressive-a-iterative-occupancy-selection-v1.private.json"
)
_DEFAULT_SELECTION_PANEL = (
    _LOCAL_ROOT / "progressive-a-iterative-occupancy-selection-v1.panel.json"
)
_DEFAULT_SELECTION_CLAIM = (
    _LOCAL_ROOT / "progressive-a-iterative-occupancy-selection-v1.claim.json"
)
_DEFAULT_PRIOR_DAMPING_SELECTION_PANEL = (
    _LOCAL_ROOT / "progressive-a-h4-damping-selection-v1.panel.json"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-iterative-occupancy-selection-v1.report.json"
)
_PRIOR_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_conformal_route_analysis"
)
_DEVELOPMENT_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-development:v1\0"
)
_EXECUTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-execution:v1\0"
)
_PLAN_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-selection-plan:v1\0"
)
_PRECLAIM_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-preclaim:v1\0"
)
_PREPARATION_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-preparation:v1\0"
)
_ARM_TO_KIND = {
    CUMULATIVE_OCCUPANCY_ARM: CENTERED_CUMULATIVE_OCCUPANCY,
    EW_OCCUPANCY_ARM: CENTERED_EW_OCCUPANCY,
}
_LIVE_PARENT_LINEAGE_KEYS = frozenset(
    {
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "accepted_x4_head_sha256",
        "bridge_binding_sha256",
        "model_sha256",
        "adapter_execution_sha256",
        "fit_manifest_sha256",
        "factorial_report_sha256",
        "factorial_report_file_sha256",
    }
)
_SOURCE_CODE_FILES = (
    "gemma3_l3_l4_iterative_occupancy_selection_diagnostic.py",
    "gemma3_l3_l4_iterative_occupancy_selection_panel.py",
    "gemma3_l3_l4_iterative_occupancy_selection_analysis.py",
    "gemma3_l3_l4_iterative_occupancy_development.py",
    "gemma3_l3_l4_iterative_occupancy_route.py",
    "gemma3_l3_l4_iterative_conformal_route.py",
    "gemma3_l3_l4_iterative_conformal_route_analysis.py",
    "gemma3_l3_l4_two_head_lowerer.py",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256 = _sha256(
    _PLAN_DOMAIN,
    {
        "arm_ids": tuple(sorted(_ARM_TO_KIND)),
        "development_panel": "expanded_calibration_a_fit_16_by_8",
        "fit_record": "one_shared_parent_nll_vjp_per_example",
        "outer_validation": "leave_one_family_out",
        "selection_rule": (
            "minimum_family_macro_predicted_absolute_delta_nll"
        ),
        "fresh_selection_panel_opened": False,
        "both_full_providers_frozen_before_fresh_open": True,
    },
)


def prepare_gemma_iterative_occupancy_selection_panel(
    *,
    expanded_corpus_artifact_path: Path | str = _DEFAULT_EXPANDED_CORPUS,
    expected_expanded_corpus_artifact_sha256: str,
    expanded_fit_binding_sha256: str,
    prior_selection_panel_path: Path | str = (
        _DEFAULT_PRIOR_DAMPING_SELECTION_PANEL
    ),
    expected_prior_selection_panel_artifact_sha256: str,
    expected_prior_selection_panel_file_sha256: str,
    selection_input_path: Path | str = _DEFAULT_SELECTION_INPUT,
    selection_panel_path: Path | str = _DEFAULT_SELECTION_PANEL,
) -> dict[str, object]:
    """Write the fixed private source and its prompt-free public commitment."""

    private_destination = Path(selection_input_path)
    public_destination = Path(selection_panel_path)
    if private_destination.exists() or public_destination.exists():
        raise FileExistsError(
            "refusing to overwrite occupancy selection preparation"
        )
    if (
        _file_sha256(prior_selection_panel_path)
        != expected_prior_selection_panel_file_sha256
    ):
        raise ValueError("prior damping selection panel file hash differs")
    lineage = load_gemma3_l3_l4_h4_damping_expanded_fit_lineage(
        expanded_corpus_artifact_path,
        expected_expanded_corpus_artifact_sha256=(
            expected_expanded_corpus_artifact_sha256
        ),
        fit_binding_sha256=expanded_fit_binding_sha256,
    )
    prior = load_gemma3_l3_l4_h4_damping_selection_panel_artifact(
        prior_selection_panel_path,
        expected_artifact_sha256=(
            expected_prior_selection_panel_artifact_sha256
        ),
    )
    private_written = False
    try:
        private_file_sha256 = (
            write_gemma3_l3_l4_iterative_occupancy_selection_role_input(
                private_destination
            )
        )
        private_written = True
        artifact = freeze_gemma3_l3_l4_iterative_occupancy_selection_panel(
            expanded_fit_lineage=lineage,
            prior_selection_panel=prior,
            selection_plan_sha256=(
                OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
            ),
            role_input_path=private_destination,
        )
        public_file_sha256 = (
            write_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
                public_destination,
                artifact,
            )
        )
    except BaseException:
        if private_written:
            private_destination.unlink(missing_ok=True)
        public_destination.unlink(missing_ok=True)
        raise
    receipt: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_l3_l4_iterative_occupancy_preparation"
        ),
        "format_version": 1,
        "expanded_corpus_artifact_sha256": (
            expected_expanded_corpus_artifact_sha256
        ),
        "expanded_fit_lineage_receipt_sha256": lineage.receipt_sha256,
        "prior_selection_panel_artifact_sha256": prior.artifact_sha256,
        "prior_selection_panel_file_sha256": (
            expected_prior_selection_panel_file_sha256
        ),
        "selection_plan_sha256": (
            OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
        ),
        "selection_role_input_file_sha256": private_file_sha256,
        "selection_panel_artifact_sha256": artifact.artifact_sha256,
        "selection_panel_file_sha256": public_file_sha256,
        "selection_manifest_sha256": artifact.manifest_sha256,
        "selection_membership_receipt_sha256": (
            artifact.membership_receipt_sha256
        ),
        "prompt_text_in_receipt": False,
        "token_ids_in_receipt": False,
        "selection_claim_created": False,
        "selection_input_opened": False,
    }
    receipt["preparation_receipt_sha256"] = _sha256(
        _PREPARATION_DOMAIN,
        receipt,
    )
    return receipt


def _load_rejected_conformal_iteration(
    path: Path | str,
    *,
    expected_report_sha256: str,
    expected_report_file_sha256: str,
    expected_collection_sha256: str,
) -> dict[str, object]:
    source = Path(path)
    if _file_sha256(source) != expected_report_file_sha256:
        raise ValueError("prior conformal-route report file hash mismatch")
    with source.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise TypeError("prior conformal-route report must be a JSON object")
    validate_gemma_iterative_conformal_route_report(report)
    semantics = _mapping(
        report.get("semantics"),
        label="prior conformal-route semantics",
    )
    decision = _mapping(
        report.get("decision"),
        label="prior conformal-route decision",
    )
    if (
        report.get("schema") != _PRIOR_SCHEMA
        or semantics.get("iteration") != 4
        or report.get("report_sha256") != expected_report_sha256
        or report.get("collection_sha256") != expected_collection_sha256
        or decision.get("retained") is not False
        or decision.get("ready_for_new_selection") is not False
        or decision.get("deployment_authorized") is not False
        or report.get("retained_full_fit") is not None
    ):
        raise ValueError(
            "Iteration 5 requires the exact rejected Iteration-4 report"
        )
    return report


def _fit_record_payload(value: object) -> dict[str, object]:
    return dict(_scalar_payload(value, label="occupancy fit record"))


def _validate_development_selection(
    development: Mapping[str, object],
) -> None:
    validate_gemma_iterative_occupancy_development_selection(development)


def _fold_payload(value: object) -> dict[str, object]:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("occupancy fold must expose to_dict")
    return dict(_scalar_payload(to_dict(), label="occupancy fold"))


def _provider_resource_receipt(
    provider: Gemma3L3L4CorrectionProvider,
) -> dict[str, object]:
    provider.validate_integrity()
    resource = getattr(provider, "resource_receipt", None)
    if not isinstance(resource, Mapping):
        raise TypeError("occupancy provider omitted its resource receipt")
    fold = _fold_payload(getattr(provider, "fold_fit", None))
    if fold.get("held_family_id") != "__full_fit__":
        raise ValueError("fresh selection requires a full-data provider")
    payload = {
        **dict(_scalar_payload(resource, label="provider resources")),
        "provider_artifact_sha256": str(provider.artifact_sha256),
        "full_fit": fold,
    }
    payload["full_provider_receipt_sha256"] = _sha256(
        _DEVELOPMENT_DOMAIN,
        payload,
    )
    return payload


def _collect_development(
    *,
    panel: object,
    adapter: object,
    bridge: object,
    parent_artifact: object,
    parent_h4: object,
    build_selection: Callable[..., Mapping[str, object]],
) -> tuple[
    tuple[object, ...],
    Mapping[str, object],
    dict[str, Gemma3L3L4CorrectionProvider],
    dict[str, object],
]:
    """Run 32 development forwards and freeze both full providers."""

    manifest = _panel_manifest(panel)  # type: ignore[arg-type]
    x4_head, validated_h4 = _validate_parent(
        panel=panel,  # type: ignore[arg-type]
        adapter=adapter,  # type: ignore[arg-type]
        bridge=bridge,  # type: ignore[arg-type]
        parent=parent_artifact,  # type: ignore[arg-type]
    )
    if validated_h4 is not parent_h4:
        raise RuntimeError("validated parent H4 identity changed")
    objective = CausalLanguageModelNLL()
    records: list[object] = []
    record_payloads: list[dict[str, object]] = []
    parent_observations: list[GemmaH4DampingFiniteNLLObservation] = []
    execution_sha256s: list[str] = []
    for example in panel.examples:  # type: ignore[union-attr]
        example.validate_integrity()
        source, source_logits, indices, targets, _positions = (
            _source_authority(
                adapter=adapter,  # type: ignore[arg-type]
                example=example,
            )
        )
        execution, gradient = bridge.execute_h4_vjp(  # type: ignore[union-attr]
            adapter,
            example.batch.model_inputs,
            objective=lambda run, batch=example.batch: objective(run, batch),
            x4_head=x4_head,
            h4_head=parent_h4,
        )
        _validate_execution(
            execution,
            example_model_inputs_sha256=example.model_inputs_sha256,
            bridge_binding_sha256=(
                bridge.bridge_binding_sha256  # type: ignore[union-attr]
            ),
            x4_head=x4_head,
            h4_head=parent_h4,
            label="occupancy development parent VJP",
        )
        candidate_h4 = getattr(execution, "candidate_h4", None)
        if (
            not isinstance(gradient, Tensor)
            or not isinstance(candidate_h4, Tensor)
            or gradient.shape != candidate_h4.shape
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ValueError("occupancy development VJP geometry differs")
        parent_logits = _gather_logits(
            getattr(execution, "logits", None),
            indices,
        )
        observation = _observation(
            example=example,
            source_logits=source_logits,
            candidate_logits=parent_logits,
            targets=targets,
        )
        record = (
            build_gemma_iterative_occupancy_conformal_route_fit_record(
                example=example,
                parent_execution=execution,
                gradient=gradient,
                parent_h4=parent_h4,
                parent_observation=observation,
            )
        )
        payload = _fit_record_payload(record)
        if (
            payload.get("example_id") != example.example_id
            or payload.get("family_id") != example.family_id
            or payload.get("model_inputs_sha256")
            != example.model_inputs_sha256
        ):
            raise ValueError("occupancy fit-record identity differs")
        records.append(record)
        record_payloads.append(payload)
        parent_observations.append(observation)
        execution_sha256s.append(str(execution.artifact_sha256))
        del (
            source,
            source_logits,
            indices,
            targets,
            execution,
            gradient,
            candidate_h4,
            parent_logits,
            observation,
        )
    canonical_records = tuple(
        record
        for _example_id, record in sorted(
            (
                (str(payload["example_id"]), record)
                for record, payload in zip(
                    records,
                    record_payloads,
                    strict=True,
                )
            )
        )
    )
    canonical_payloads = tuple(
        sorted(record_payloads, key=lambda row: str(row["example_id"]))
    )
    if (
        len(canonical_records) != 16
        or len({str(row["fit_record_sha256"]) for row in canonical_payloads})
        != 16
    ):
        raise ValueError("occupancy fit-record geometry differs")
    frozen_record_bytes = _canonical_bytes(canonical_payloads)

    families = tuple(sorted(set(manifest.values())))
    folds_by_arm: dict[str, tuple[object, ...]] = {}
    fold_payloads_by_arm: dict[str, tuple[dict[str, object], ...]] = {}
    for arm_id, occupancy_kind in _ARM_TO_KIND.items():
        folds: list[object] = []
        for held_family in families:
            training = tuple(
                record
                for record, payload in zip(
                    canonical_records,
                    canonical_payloads,
                    strict=True,
                )
                if payload["family_id"] != held_family
            )
            fold = fit_gemma_iterative_occupancy_conformal_route_fold(
                training,
                held_family_id=held_family,
                occupancy_kind=occupancy_kind,
            )
            payload = _fold_payload(fold)
            if (
                payload.get("held_family_id") != held_family
                or len(tuple(payload.get("train_example_ids", ()))) != 14
                or len(tuple(payload.get("train_family_ids", ()))) != 7
            ):
                raise RuntimeError("occupancy LOFO receipt leaked a family")
            folds.append(fold)
        folds_by_arm[arm_id] = tuple(folds)
        fold_payloads_by_arm[arm_id] = tuple(
            sorted(
                (_fold_payload(fold) for fold in folds),
                key=lambda row: str(row["held_family_id"]),
            )
        )

    development = dict(
        build_selection(
            fit_records=canonical_records,
            fold_receipts_by_arm=folds_by_arm,
        )
    )
    _validate_development_selection(development)
    if (
        development.get("selection_opened") is not False
        or development.get("selection_rule_frozen") is not True
        or development.get("selected_arm_id") not in _ARM_TO_KIND
    ):
        raise ValueError("occupancy development selection is not frozen")

    providers: dict[str, Gemma3L3L4CorrectionProvider] = {}
    resources: dict[str, object] = {}
    for arm_id, occupancy_kind in _ARM_TO_KIND.items():
        provider = fit_gemma_iterative_occupancy_conformal_route_full_provider(
            records=canonical_records,
            parent_h4=parent_h4,
            occupancy_kind=occupancy_kind,
            parent_artifact_sha256=parent_artifact.artifact_sha256,
        )
        if not isinstance(provider, Gemma3L3L4CorrectionProvider):
            raise TypeError("occupancy full fitter returned the wrong type")
        if (
            getattr(provider, "occupancy_kind", None) != occupancy_kind
            or getattr(provider, "parent_artifact_sha256", None)
            != parent_artifact.artifact_sha256
        ):
            raise ValueError("occupancy full provider identity differs")
        providers[arm_id] = provider
        resources[arm_id] = _provider_resource_receipt(provider)
    if (
        _canonical_bytes(
            tuple(_fit_record_payload(record) for record in canonical_records)
        )
        != frozen_record_bytes
    ):
        raise RuntimeError(
            "occupancy fold selection or full fitting mutated fit records"
        )

    development_receipt = {
        "selection": development,
        "fit_record_sha256s": tuple(
            sorted(str(row["fit_record_sha256"]) for row in canonical_payloads)
        ),
        "fold_receipt_sha256s_by_arm": {
            arm_id: tuple(
                sorted(str(row["fold_receipt_sha256"]) for row in rows)
            )
            for arm_id, rows in sorted(fold_payloads_by_arm.items())
        },
        "parent_observation_sha256s": tuple(
            sorted(row.observation_sha256 for row in parent_observations)
        ),
        "parent_vjp_execution_sha256s": tuple(sorted(execution_sha256s)),
    }
    development_receipt["development_receipt_sha256"] = _sha256(
        _DEVELOPMENT_DOMAIN,
        development_receipt,
    )
    return canonical_records, development, providers, {
        "resources": resources,
        "receipt": development_receipt,
    }


def _collect_fresh_selection(
    *,
    panel: object,
    adapter: object,
    bridge: object,
    x4_head: object,
    parent_h4: object,
    providers: Mapping[str, Gemma3L3L4CorrectionProvider],
) -> tuple[
    tuple[GemmaH4DampingFiniteNLLObservation, ...],
    tuple[GemmaH4DampingFiniteNLLObservation, ...],
    tuple[GemmaH4DampingFiniteNLLObservation, ...],
    dict[str, object],
]:
    """Run the exact four-forward fresh phase without any VJP."""

    manifest = {
        example.example_id: example.family_id
        for example in panel.examples  # type: ignore[union-attr]
    }
    counts = Counter(manifest.values())
    if (
        panel.role != "calibration_a_selection"  # type: ignore[union-attr]
        or len(manifest) != 16
        or len(counts) != 8
        or set(counts.values()) != {2}
    ):
        raise ValueError("fresh occupancy panel is not strict 16-by-8")
    parent_rows: list[GemmaH4DampingFiniteNLLObservation] = []
    cumulative_rows: list[GemmaH4DampingFiniteNLLObservation] = []
    ew_rows: list[GemmaH4DampingFiniteNLLObservation] = []
    execution_by_arm: dict[str, dict[str, str]] = {
        "parent": {},
        CUMULATIVE_OCCUPANCY_ARM: {},
        EW_OCCUPANCY_ARM: {},
    }
    for example in panel.examples:  # type: ignore[union-attr]
        example.validate_integrity()
        _source, source_logits, indices, targets, _positions = (
            _source_authority(
                adapter=adapter,  # type: ignore[arg-type]
                example=example,
            )
        )
        with torch.no_grad():
            parent_execution = bridge.execute(  # type: ignore[union-attr]
                adapter,
                example.batch.model_inputs,
                x4_head=x4_head,
                h4_head=parent_h4,
            )
        _validate_execution(
            parent_execution,
            example_model_inputs_sha256=example.model_inputs_sha256,
            bridge_binding_sha256=(
                bridge.bridge_binding_sha256  # type: ignore[union-attr]
            ),
            x4_head=x4_head,
            h4_head=parent_h4,
            label="fresh occupancy parent",
        )
        parent_observation = _observation(
            example=example,
            source_logits=source_logits,
            candidate_logits=_gather_logits(
                parent_execution.logits,
                indices,
            ),
            targets=targets,
        )
        arm_observations: dict[
            str, GemmaH4DampingFiniteNLLObservation
        ] = {}
        for arm_id in (CUMULATIVE_OCCUPANCY_ARM, EW_OCCUPANCY_ARM):
            provider = providers[arm_id]
            with torch.no_grad():
                execution = bridge.execute(  # type: ignore[union-attr]
                    adapter,
                    example.batch.model_inputs,
                    x4_head=x4_head,
                    h4_head=provider,
                )
            _validate_execution(
                execution,
                example_model_inputs_sha256=example.model_inputs_sha256,
                bridge_binding_sha256=(
                    bridge.bridge_binding_sha256  # type: ignore[union-attr]
                ),
                x4_head=x4_head,
                h4_head=provider,
                label=f"fresh occupancy {arm_id}",
            )
            observation = _observation(
                example=example,
                source_logits=source_logits,
                candidate_logits=_gather_logits(execution.logits, indices),
                targets=targets,
            )
            if not _same_source(parent_observation, observation):
                raise RuntimeError("fresh occupancy source authority drifted")
            arm_observations[arm_id] = observation
            execution_by_arm[arm_id][example.example_id] = str(
                execution.artifact_sha256
            )
            del execution
        parent_rows.append(parent_observation)
        cumulative_rows.append(
            arm_observations[CUMULATIVE_OCCUPANCY_ARM]
        )
        ew_rows.append(arm_observations[EW_OCCUPANCY_ARM])
        execution_by_arm["parent"][example.example_id] = str(
            parent_execution.artifact_sha256
        )
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(
                example.batch.model_inputs
            )
            != example.model_inputs_sha256
        ):
            raise RuntimeError("fresh occupancy model inputs changed")
        del (
            _source,
            source_logits,
            indices,
            targets,
            parent_execution,
            parent_observation,
            arm_observations,
        )
    return (
        tuple(sorted(parent_rows, key=lambda row: row.example_id)),
        tuple(sorted(cumulative_rows, key=lambda row: row.example_id)),
        tuple(sorted(ew_rows, key=lambda row: row.example_id)),
        {
            arm_id: dict(sorted(values.items()))
            for arm_id, values in sorted(execution_by_arm.items())
        },
    )


def _execution_audit(
    *,
    development_receipt: Mapping[str, object],
    selection_execution_by_arm: Mapping[str, object],
    selection_claim_sha256: str,
) -> dict[str, object]:
    execution_payload = {
        "development_receipt_sha256": development_receipt[
            "development_receipt_sha256"
        ],
        "selection_execution_by_arm": dict(selection_execution_by_arm),
        "selection_claim_sha256": selection_claim_sha256,
    }
    return {
        "development_example_count": 16,
        "selection_example_count": 16,
        "development_source_forward_count": 16,
        "development_parent_vjp_forward_count": 16,
        "selection_source_forward_count": 16,
        "selection_parent_forward_count": 16,
        "selection_cumulative_forward_count": 16,
        "selection_ew_forward_count": 16,
        "selection_vjp_forward_count": 0,
        "total_model_forward_count": 96,
        "model_forward_count_per_development_example": 2,
        "model_forward_count_per_selection_example": 4,
        "development_fit_records_shared_across_arms": True,
        "selection_source_reused_within_prompt": True,
        "selection_input_open_count": 1,
        "candidate_changes_after_selection_open": False,
        "raw_prompts_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "raw_activations_retained": False,
        "gradient_tensors_retained": False,
        "model_weights_retained": False,
        **execution_payload,
        "execution_receipt_sha256": _sha256(
            _EXECUTION_DOMAIN,
            execution_payload,
        ),
    }


def _preclaim_boundary_receipt(
    *,
    development: Mapping[str, object],
    resources: Mapping[str, object],
    development_receipt_sha256: str,
    selection_plan_sha256: str,
    public_inputs_unchanged: bool,
    private_input_unchanged: bool,
    source_code_unchanged: bool,
    live_model_unchanged: bool,
) -> str:
    """Prove every mutable development action ended before the claim."""

    _validate_development_selection(development)
    if (
        set(resources) != set(_ARM_TO_KIND)
        or selection_plan_sha256
        != OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
        or not all(
            (
                public_inputs_unchanged,
                private_input_unchanged,
                source_code_unchanged,
                live_model_unchanged,
            )
        )
    ):
        raise RuntimeError("occupancy preclaim boundary is incomplete")
    provider_receipts: dict[str, str] = {}
    for arm_id, raw in resources.items():
        resource = _mapping(raw, label=f"{arm_id} full provider resource")
        full_fit = _mapping(
            resource.get("full_fit"),
            label=f"{arm_id} full fit",
        )
        if full_fit.get("held_family_id") != "__full_fit__":
            raise RuntimeError("both occupancy arms require full fits")
        provider_receipts[arm_id] = str(
            resource["full_provider_receipt_sha256"]
        )
    return _sha256(
        _PRECLAIM_DOMAIN,
        {
            "development_receipt_sha256": (
                development_receipt_sha256
            ),
            "selected_arm_id": development["selected_arm_id"],
            "full_provider_receipt_sha256_by_arm": dict(
                sorted(provider_receipts.items())
            ),
            "selection_plan_sha256": selection_plan_sha256,
            "public_inputs_unchanged": True,
            "private_input_file_hash_matches_panel": True,
            "source_code_unchanged": True,
            "live_model_unchanged": True,
            "fresh_panel_opened": False,
        },
    )


def _publish_report(
    path: Path,
    report: Mapping[str, object],
) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite occupancy report")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    stage = Path(stage_name)
    try:
        with stage.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        stage.unlink(missing_ok=True)


def run_gemma_iterative_occupancy_selection_diagnostic(
    *,
    corpus_artifact_path: Path | str = _DEFAULT_EXPANDED_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    materialization_report_path: Path | str = (
        _DEFAULT_MATERIALIZATION_REPORT
    ),
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    factorial_report_path: Path | str = _DEFAULT_FACTORIAL_REPORT,
    expected_factorial_report_sha256: str,
    expected_factorial_report_file_sha256: str,
    prior_iteration_report_path: Path | str = (
        _DEFAULT_PRIOR_ITERATION_REPORT
    ),
    expected_prior_iteration_report_sha256: str,
    expected_prior_iteration_report_file_sha256: str,
    expected_prior_iteration_collection_sha256: str,
    selection_panel_path: Path | str = _DEFAULT_SELECTION_PANEL,
    expected_selection_panel_artifact_sha256: str,
    expected_selection_panel_file_sha256: str,
    selection_input_path: Path | str = _DEFAULT_SELECTION_INPUT,
    selection_claim_path: Path | str = _DEFAULT_SELECTION_CLAIM,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    _build_development_selection: Callable[
        ..., Mapping[str, object]
    ] = build_gemma_iterative_occupancy_development_selection,
) -> dict[str, object]:
    """Run the exact preregistered 96-forward Iteration-5 diagnostic."""

    destination = Path(output)
    claim_path = Path(selection_claim_path)
    private_input_path = Path(selection_input_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite occupancy report")
    if claim_path.exists():
        raise FileExistsError(
            "fresh occupancy panel already has a durable claim"
        )
    if not private_input_path.is_file():
        raise ValueError("fresh occupancy role input must be a regular file")
    if not callable(_build_development_selection):
        raise TypeError("occupancy development selector must be callable")
    prior = _load_rejected_conformal_iteration(
        prior_iteration_report_path,
        expected_report_sha256=expected_prior_iteration_report_sha256,
        expected_report_file_sha256=(
            expected_prior_iteration_report_file_sha256
        ),
        expected_collection_sha256=(
            expected_prior_iteration_collection_sha256
        ),
    )
    selection_artifact = (
        load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
            selection_panel_path,
            expected_artifact_sha256=(
                expected_selection_panel_artifact_sha256
            ),
        )
    )
    if (
        _file_sha256(selection_panel_path)
        != expected_selection_panel_file_sha256
        or selection_artifact.selection_plan_sha256
        != OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
    ):
        raise ValueError("fresh occupancy panel plan or file differs")

    factorial = _load_factorial_report(
        factorial_report_path,
        expected_report_sha256=expected_factorial_report_sha256,
        expected_report_file_sha256=(
            expected_factorial_report_file_sha256
        ),
    )
    materialization, materialization_report = (
        load_gemma_h4_damping_materialization(
            materialization_report_path,
            expected_report_sha256=(
                expected_materialization_report_sha256
            ),
            expected_report_file_sha256=(
                expected_materialization_report_file_sha256
            ),
        )
    )
    recollection = _mapping(
        materialization_report.get("recollection"),
        label="materialization recollection",
    )
    accepted_x4_provenance = (
        _validate_factorial_materialization_lineage(
            factorial=factorial,
            materialization=materialization,
            materialization_report=materialization_report,
            materialization_report_file_sha256=(
                expected_materialization_report_file_sha256
            ),
        )
    )
    parent = materialization.alpha0_artifact
    parent.validate_integrity()
    x4_head = parent.head(_X4_SITE)
    h4_head = parent.head(_H4_SITE)
    if x4_head is None or h4_head is None:
        raise ValueError("matched alpha0 parent omitted X4 or lag B")

    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    protocol.validate_integrity()
    metadata = protocol.metadata()
    tokenizer_contract = dict(
        _mapping(metadata["tokenizer"], label="frozen tokenizer")
    )
    corpus, fit_input = load_gemma3_l3_l4_progressive_a_fit_role(
        corpus_artifact_path,
        fit_input_path=fit_input_path,
        expected_artifact_sha256=str(
            recollection["corpus_artifact_sha256"]
        ),
        tokenizer_contract=tokenizer_contract,
    )
    tokenizer, live_tokenizer_contract = (
        _load_and_validate_frozen_local_tokenizer(protocol=protocol)
    )
    if (
        _canonical_bytes(live_tokenizer_contract)
        != _canonical_bytes(tokenizer_contract)
        or gemma3_l3_l4_progressive_a_tokenizer_contract_sha256(
            tokenizer_contract
        )
        != corpus.tokenizer_contract_sha256
    ):
        raise ValueError("live tokenizer differs from expanded-fit contract")
    fit_panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=fit_input,
        view=corpus.role_view("calibration_a_fit"),
        max_length=int(tokenizer_contract["max_length"]),
        device=torch.device(str(tokenizer_contract["device"])),
        forbidden_manifest_sha256s=(
            corpus.forbidden_assessment_manifest_sha256s
        ),
    )
    if (
        fit_panel.manifest_sha256 != recollection["fit_manifest_sha256"]
        or fit_panel.binding_sha256 != recollection["fit_binding_sha256"]
        or len(fit_panel.examples) != 16
        or len(fit_panel.family_ids) != 8
    ):
        raise ValueError("expanded fit panel differs from materialization")
    _validate_factorial_fit_lineage(
        factorial=factorial,
        family_by_example={
            example.example_id: example.family_id
            for example in fit_panel.examples
        },
        model_input_sha256s=tuple(
            example.model_inputs_sha256 for example in fit_panel.examples
        ),
        corpus_artifact_sha256=corpus.artifact_sha256,
        fit_input_file_sha256=fit_input.source_file_sha256,
        fit_manifest_sha256=fit_panel.manifest_sha256,
        fit_binding_sha256=fit_panel.binding_sha256,
        materialization_report_sha256=str(
            materialization_report["report_sha256"]
        ),
        materialization_report_file_sha256=(
            expected_materialization_report_file_sha256
        ),
        accepted_x4_provenance=accepted_x4_provenance,
    )
    panel_lineage = selection_artifact.expanded_fit_lineage
    if (
        panel_lineage.expanded_corpus_artifact_sha256
        != corpus.artifact_sha256
        or panel_lineage.tokenizer_contract_sha256
        != corpus.tokenizer_contract_sha256
        or panel_lineage.fit_manifest_sha256 != fit_panel.manifest_sha256
        or panel_lineage.fit_role_input_file_sha256
        != fit_input.source_file_sha256
        or panel_lineage.fit_binding_sha256 != fit_panel.binding_sha256
    ):
        raise ValueError(
            "fresh occupancy panel differs from expanded-fit lineage"
        )

    model_metadata = _mapping(metadata["model"], label="frozen model")
    graph_binding = _mapping(
        metadata["graph_candidate"],
        label="frozen graph candidate",
    )
    basis_binding = _mapping(
        metadata["prompt_blind_basis"],
        label="frozen basis",
    )
    materialized_files = _mapping(
        materialization_report.get("files"),
        label="materialization files",
    )
    immutable_paths = {
        "corpus_artifact": Path(corpus_artifact_path),
        "fit_input": Path(fit_input_path),
        "materialization_report": Path(materialization_report_path),
        "matched_alpha0_candidate": Path(
            str(
                _mapping(
                    materialized_files["matched_alpha0"],
                    label="matched alpha0 file",
                )["tensor_file"]
            )
        ),
        "challenger_alpha0_5_candidate": Path(
            str(
                _mapping(
                    materialized_files["challenger_alpha0_5"],
                    label="challenger file",
                )["tensor_file"]
            )
        ),
        "factorial_report": Path(factorial_report_path),
        "prior_iteration_report": Path(prior_iteration_report_path),
        "selection_panel": Path(selection_panel_path),
        "graph_candidate": Path(graph_candidate_path),
        "basis_package": Path(basis_package_path),
        "base_artifact": Path(base_artifact_path),
        "refit_artifact": Path(refit_artifact_path),
    }
    immutable_before = {
        name: _file_sha256(path) for name, path in immutable_paths.items()
    }
    immutable_expected = {
        "corpus_artifact": _file_sha256(corpus_artifact_path),
        "fit_input": fit_input.source_file_sha256,
        "materialization_report": (
            expected_materialization_report_file_sha256
        ),
        "matched_alpha0_candidate": str(
            _mapping(
                materialized_files["matched_alpha0"],
                label="matched alpha0 file",
            )["tensor_file_sha256"]
        ),
        "challenger_alpha0_5_candidate": str(
            _mapping(
                materialized_files["challenger_alpha0_5"],
                label="challenger file",
            )["tensor_file_sha256"]
        ),
        "factorial_report": expected_factorial_report_file_sha256,
        "prior_iteration_report": (
            expected_prior_iteration_report_file_sha256
        ),
        "selection_panel": expected_selection_panel_file_sha256,
        "graph_candidate": str(graph_binding["tensor_file_sha256"]),
        "basis_package": str(basis_binding["tensor_file_sha256"]),
        "base_artifact": str(recollection["base_artifact_file_sha256"]),
        "refit_artifact": str(recollection["refit_artifact_file_sha256"]),
    }
    if immutable_before != immutable_expected:
        raise ValueError("occupancy immutable input binding differs")
    code_before = _source_code_sha256s(_SOURCE_CODE_FILES)

    device = resolve_torch_device("cpu")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != model_metadata["source_model_sha256"]:
        raise ValueError("live raw Gemma differs from frozen source")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256 = adapter.model_fingerprint()
        factorized_execution_sha256 = adapter.execution_fingerprint()
        if (
            factorized_model_sha256
            != graph_binding["factorized_live_execution_sha256"]
            or factorized_execution_sha256
            != graph_binding["factorized_refit_execution_sha256"]
            or factorized_model_sha256
            != recollection["factorized_model_sha256"]
            or factorized_execution_sha256
            != recollection["factorized_execution_sha256"]
        ):
            raise ValueError("live factorized Gemma differs")
        graph_candidate = load_gemma3_graph_organized_svd_candidate(
            graph_candidate_path,
            expected_file_sha256=str(graph_binding["tensor_file_sha256"]),
        )
        basis = load_gemma3_l3_l4_basis_package(
            basis_package_path,
            expected_file_sha256=str(basis_binding["tensor_file_sha256"]),
            expected_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
        )
        runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            graph_candidate,
            basis,
            expected_candidate_artifact_sha256=str(
                graph_binding["logical_artifact_sha256"]
            ),
            expected_basis_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
            expected_plan_artifact_sha256=str(
                graph_binding["deployment_plan_sha256"]
            ),
            expected_live_model_sha256=str(
                graph_binding["factorized_live_execution_sha256"]
            ),
            expected_adapter_execution_sha256=str(
                graph_binding["factorized_refit_execution_sha256"]
            ),
            analysis_device="cpu",
        )
        if (
            runtime.runtime_binding_sha256
            != recollection["progressive_runtime_binding_sha256"]
        ):
            raise ValueError("progressive runtime binding differs")
        bridge = runtime.export_one_pass_bridge()
        _validate_factorial_live_lineage(
            factorial=factorial,
            materialization=materialization,
            factorized_model_sha256=factorized_model_sha256,
            factorized_execution_sha256=factorized_execution_sha256,
            bridge=bridge,
        )
        base_lineage = {
            "parent_artifact_sha256": parent.artifact_sha256,
            "parent_h4_head_sha256": h4_head.artifact_sha256,
            "accepted_x4_head_sha256": x4_head.artifact_sha256,
            "bridge_binding_sha256": bridge.bridge_binding_sha256,
            "model_sha256": factorized_model_sha256,
            "adapter_execution_sha256": factorized_execution_sha256,
            "fit_manifest_sha256": fit_panel.manifest_sha256,
            "factorial_report_sha256": str(factorial["report_sha256"]),
            "factorial_report_file_sha256": immutable_before[
                "factorial_report"
            ],
        }
        prior_lineage = _mapping(
            prior.get("lineage"),
            label="prior conformal-route lineage",
        )
        if any(
            prior_lineage.get(key) != value
            for key, value in base_lineage.items()
            if key in _LIVE_PARENT_LINEAGE_KEYS
        ):
            raise ValueError(
                "Iteration-4 report differs from the live parent lineage"
            )

        _records, development, providers, development_bundle = (
            _collect_development(
                panel=fit_panel,
                adapter=adapter,
                bridge=bridge,
                parent_artifact=parent,
                parent_h4=h4_head,
                build_selection=_build_development_selection,
            )
        )
        development_receipt = _mapping(
            development_bundle["receipt"],
            label="occupancy development receipt",
        )
        resources = _mapping(
            development_bundle["resources"],
            label="occupancy resources",
        )
        # This is the irreversible boundary.  Every public input, source
        # module, live model identity, arm choice, and both full providers is
        # frozen before the private prompt file is first read.
        public_inputs_unchanged = (
            {
                name: _file_sha256(path)
                for name, path in immutable_paths.items()
            }
            == immutable_before
        )
        source_code_unchanged = (
            _source_code_sha256s(_SOURCE_CODE_FILES) == code_before
        )
        private_input_unchanged = (
            _file_sha256(private_input_path)
            == selection_artifact.role_input_file_sha256
        )
        live_model_unchanged = (
            adapter.model_fingerprint() == factorized_model_sha256
            and adapter.execution_fingerprint()
            == factorized_execution_sha256
        )
        preclaim_boundary_receipt_sha256 = _preclaim_boundary_receipt(
            development=development,
            resources=resources,
            development_receipt_sha256=str(
                development_receipt["development_receipt_sha256"]
            ),
            selection_plan_sha256=(
                selection_artifact.selection_plan_sha256
            ),
            public_inputs_unchanged=public_inputs_unchanged,
            private_input_unchanged=private_input_unchanged,
            source_code_unchanged=source_code_unchanged,
            live_model_unchanged=live_model_unchanged,
        )
        claim = claim_gemma3_l3_l4_iterative_occupancy_selection_panel(
            claim_path,
            artifact=selection_artifact,
        )
        source = Gemma3L3L4IterativeOccupancySelectionPanelSource(
            artifact=selection_artifact,
            role_input_path=private_input_path,
        )
        selection_panel = (
            materialize_gemma3_l3_l4_iterative_occupancy_selection_panel(
                source=source,
                claim=claim,
                tokenizer=tokenizer,
                max_length=int(tokenizer_contract["max_length"]),
                device=torch.device(str(tokenizer_contract["device"])),
            )
        )
        parent_rows, cumulative_rows, ew_rows, selection_executions = (
            _collect_fresh_selection(
                panel=selection_panel,
                adapter=adapter,
                bridge=bridge,
                x4_head=x4_head,
                parent_h4=h4_head,
                providers=providers,
            )
        )
        if not source.consumed or not source.opened:
            raise RuntimeError("fresh occupancy input was not opened once")
        audit = _execution_audit(
            development_receipt=development_receipt,
            selection_execution_by_arm=selection_executions,
            selection_claim_sha256=claim.claim_sha256,
        )
        lineage = {
            **base_lineage,
            "prior_iteration_report_sha256": (
                expected_prior_iteration_report_sha256
            ),
            "prior_iteration_report_file_sha256": (
                expected_prior_iteration_report_file_sha256
            ),
            "prior_iteration_collection_sha256": (
                expected_prior_iteration_collection_sha256
            ),
            "selection_panel_artifact_sha256": (
                selection_artifact.artifact_sha256
            ),
            "selection_panel_file_sha256": (
                expected_selection_panel_file_sha256
            ),
            "selection_manifest_sha256": (
                selection_artifact.manifest_sha256
            ),
            "selection_membership_receipt_sha256": (
                selection_artifact.membership_receipt_sha256
            ),
            "selection_plan_sha256": (
                selection_artifact.selection_plan_sha256
            ),
            "selection_claim_sha256": claim.claim_sha256,
            "selection_claim_file_sha256": claim.claim_file_sha256,
            "development_receipt_sha256": development_receipt[
                "development_receipt_sha256"
            ],
            "preclaim_boundary_receipt_sha256": (
                preclaim_boundary_receipt_sha256
            ),
        }
        manifest = {
            example.example_id: example.family_id
            for example in selection_panel.examples
        }
        report = build_gemma_iterative_occupancy_selection_report(
            development=development,
            parent_observations=parent_rows,
            cumulative_observations=cumulative_rows,
            ew_observations=ew_rows,
            manifest=manifest,
            lineage=lineage,
            resources=resources,
            audit=audit,
        )
        validate_gemma_iterative_occupancy_selection_report(report)
        if (
            {
                name: _file_sha256(path)
                for name, path in immutable_paths.items()
            }
            != immutable_before
            or _file_sha256(private_input_path)
            != selection_artifact.role_input_file_sha256
            or _file_sha256(claim.path) != claim.claim_file_sha256
            or _source_code_sha256s(_SOURCE_CODE_FILES) != code_before
            or adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
        ):
            raise RuntimeError(
                "occupancy evidence inputs or runtime changed after opening"
            )
        _publish_report(destination, report)
        with destination.open("r", encoding="utf-8") as handle:
            replay = json.load(handle)
        validate_gemma_iterative_occupancy_selection_report(replay)
        if _canonical_bytes(replay) != _canonical_bytes(report):
            raise RuntimeError("published occupancy report differs")
        return report
    finally:
        switcher.close()


def build_preparation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the broad natural Iteration-5 private panel and publish "
            "its prompt-free commitment without creating a consume claim."
        )
    )
    parser.add_argument(
        "--expanded-corpus-artifact",
        type=Path,
        default=_DEFAULT_EXPANDED_CORPUS,
    )
    parser.add_argument(
        "--expanded-corpus-artifact-sha256",
        required=True,
    )
    parser.add_argument("--expanded-fit-binding-sha256", required=True)
    parser.add_argument(
        "--prior-selection-panel",
        type=Path,
        default=_DEFAULT_PRIOR_DAMPING_SELECTION_PANEL,
    )
    parser.add_argument(
        "--prior-selection-panel-sha256",
        required=True,
    )
    parser.add_argument(
        "--prior-selection-panel-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--selection-input",
        type=Path,
        default=_DEFAULT_SELECTION_INPUT,
    )
    parser.add_argument(
        "--selection-panel",
        type=Path,
        default=_DEFAULT_SELECTION_PANEL,
    )
    return parser


def preparation_main(argv: Sequence[str] | None = None) -> int:
    args = build_preparation_parser().parse_args(argv)
    receipt = prepare_gemma_iterative_occupancy_selection_panel(
        expanded_corpus_artifact_path=args.expanded_corpus_artifact,
        expected_expanded_corpus_artifact_sha256=(
            args.expanded_corpus_artifact_sha256
        ),
        expanded_fit_binding_sha256=args.expanded_fit_binding_sha256,
        prior_selection_panel_path=args.prior_selection_panel,
        expected_prior_selection_panel_artifact_sha256=(
            args.prior_selection_panel_sha256
        ),
        expected_prior_selection_panel_file_sha256=(
            args.prior_selection_panel_file_sha256
        ),
        selection_input_path=args.selection_input,
        selection_panel_path=args.selection_panel,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 5 with pre-open LOFO arm selection and one fresh "
            "family-disjoint exact evaluation."
        )
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=_DEFAULT_EXPANDED_CORPUS,
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=_DEFAULT_EXPANDED_FIT_INPUT,
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=_DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument("--materialization-report-sha256", required=True)
    parser.add_argument(
        "--materialization-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--factorial-report",
        type=Path,
        default=_DEFAULT_FACTORIAL_REPORT,
    )
    parser.add_argument("--factorial-report-sha256", required=True)
    parser.add_argument(
        "--factorial-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--prior-iteration-report",
        type=Path,
        default=_DEFAULT_PRIOR_ITERATION_REPORT,
    )
    parser.add_argument("--prior-iteration-report-sha256", required=True)
    parser.add_argument(
        "--prior-iteration-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--prior-iteration-collection-sha256",
        required=True,
    )
    parser.add_argument(
        "--selection-panel",
        type=Path,
        default=_DEFAULT_SELECTION_PANEL,
    )
    parser.add_argument("--selection-panel-sha256", required=True)
    parser.add_argument("--selection-panel-file-sha256", required=True)
    parser.add_argument(
        "--selection-input",
        type=Path,
        default=_DEFAULT_SELECTION_INPUT,
    )
    parser.add_argument(
        "--selection-claim",
        type=Path,
        default=_DEFAULT_SELECTION_CLAIM,
    )
    parser.add_argument(
        "--graph-candidate",
        type=Path,
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma_iterative_occupancy_selection_diagnostic(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        materialization_report_path=args.materialization_report,
        expected_materialization_report_sha256=(
            args.materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            args.materialization_report_file_sha256
        ),
        factorial_report_path=args.factorial_report,
        expected_factorial_report_sha256=args.factorial_report_sha256,
        expected_factorial_report_file_sha256=(
            args.factorial_report_file_sha256
        ),
        prior_iteration_report_path=args.prior_iteration_report,
        expected_prior_iteration_report_sha256=(
            args.prior_iteration_report_sha256
        ),
        expected_prior_iteration_report_file_sha256=(
            args.prior_iteration_report_file_sha256
        ),
        expected_prior_iteration_collection_sha256=(
            args.prior_iteration_collection_sha256
        ),
        selection_panel_path=args.selection_panel,
        expected_selection_panel_artifact_sha256=(
            args.selection_panel_sha256
        ),
        expected_selection_panel_file_sha256=(
            args.selection_panel_file_sha256
        ),
        selection_input_path=args.selection_input,
        selection_claim_path=args.selection_claim,
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
