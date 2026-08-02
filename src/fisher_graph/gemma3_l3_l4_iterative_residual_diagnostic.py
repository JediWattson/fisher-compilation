"""Executable first residual-boost iteration on reusable Gemma development data."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch

from .adapters.gemma3 import Gemma3CausalLMAdapter
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
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
)
from .gemma3_l3_l4_h4_damping_materialization import (
    GemmaH4DampingMaterialization,
    load_gemma_h4_damping_materialization,
)
from .gemma3_l3_l4_h4_incremental_signal_diagnostic import (
    _canonical_json_bytes,
)
from .gemma3_l3_l4_iterative_residual_analysis import (
    build_gemma_iterative_residual_report,
    validate_gemma_iterative_residual_report,
)
from .gemma3_l3_l4_iterative_residual_boost import (
    build_gemma_iterative_residual_fit_record,
    fit_gemma_iterative_residual_fold_provider,
    fit_gemma_iterative_residual_full_provider,
)
from .gemma3_l3_l4_iterative_residual_campaign import (
    DEFAULT_GEMMA_ITERATIVE_RESIDUAL_CAMPAIGN_RECIPE,
    GemmaIterativeResidualCampaignRecipe,
    collect_gemma_iterative_residual_campaign_live,
    publish_gemma_iterative_residual_campaign_report,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    _file_sha256,
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    gemma3_l3_l4_progressive_a_tokenizer_contract_sha256,
    load_gemma3_l3_l4_progressive_a_fit_role,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .gemma3_l3_l4_x4_h4_factorial_analysis import (
    validate_gemma_x4_h4_factorial_report,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma_iterative_residual_diagnostic",
]


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
DEFAULT_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-iterative-residual-position-v1.report.json"
)
_ACCEPTED_X4_PROVENANCE_KEYS = frozenset(
    {
        "campaign_spec_sha256",
        "candidate_artifact_sha256",
        "candidate_execution_sha256",
        "candidate_file_sha256",
        "candidate_runtime_binding_sha256",
        "protocol_sha256",
        "report_file_sha256",
        "report_sha256",
        "transcript_sha256",
    }
)
_FACTORIAL_LINEAGE_KEYS = frozenset(
    {
        "accepted_x4_campaign_spec_sha256",
        "accepted_x4_candidate_artifact_sha256",
        "accepted_x4_candidate_execution_sha256",
        "accepted_x4_candidate_file_sha256",
        "accepted_x4_candidate_runtime_binding_sha256",
        "accepted_x4_protocol_sha256",
        "accepted_x4_report_file_sha256",
        "accepted_x4_report_sha256",
        "accepted_x4_transcript_sha256",
        "corpus_artifact_sha256",
        "fit_binding_sha256",
        "fit_input_file_sha256",
        "fit_manifest_sha256",
        "materialization_report_file_sha256",
        "materialization_report_sha256",
    }
)
_FACTORIAL_RESOURCE_KEYS = frozenset(
    {
        "accepted_x4_head_logical_macs_per_token_upper_bound",
        "accepted_x4_head_prepared_float_scalar_count",
        "bridge_logical_macs_per_token_upper_bound",
        "bridge_prepared_float_scalar_count",
        "independent_h4_logical_macs_per_token_upper_bound",
        "independent_h4_prepared_float_scalar_count",
        "lag_b_h4_logical_macs_per_token_upper_bound",
        "lag_b_h4_prepared_float_scalar_count",
    }
)


@dataclass(frozen=True, slots=True)
class _GemmaIterativeDiagnosticRecipe:
    """Internal wiring for one preregistered iterative candidate."""

    campaign_recipe: GemmaIterativeResidualCampaignRecipe
    make_fit_record: Callable[..., object]
    fit_fold: Callable[..., object]
    build_report: Callable[..., Mapping[str, object]]
    fit_full: Callable[..., object]
    validate_report: Callable[[Mapping[str, object]], None]
    report_label: str
    expected_parent_lineage: Mapping[str, str] = field(
        default_factory=dict
    )
    extra_lineage: Mapping[str, str] = field(default_factory=dict)
    extra_immutable_inputs: tuple[
        tuple[str, Path, str], ...
    ] = ()
    source_code_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(
            self.campaign_recipe,
            GemmaIterativeResidualCampaignRecipe,
        ):
            raise TypeError("diagnostic campaign recipe is invalid")
        for name in (
            "make_fit_record",
            "fit_fold",
            "build_report",
            "fit_full",
            "validate_report",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"diagnostic {name} must be callable")
        if not isinstance(self.report_label, str) or not self.report_label:
            raise ValueError("diagnostic report label must be nonempty")
        for mapping_name in (
            "expected_parent_lineage",
            "extra_lineage",
        ):
            values = getattr(self, mapping_name)
            if not isinstance(values, Mapping):
                raise TypeError(
                    f"diagnostic {mapping_name} must be a mapping"
                )
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"diagnostic {mapping_name} key is invalid"
                    )
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                ):
                    raise ValueError(
                        f"diagnostic {mapping_name} hash is invalid"
                    )
        labels: set[str] = set()
        for label, path, file_sha256 in self.extra_immutable_inputs:
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError(
                    "diagnostic immutable labels must be unique"
                )
            labels.add(label)
            if not isinstance(path, Path):
                raise TypeError("diagnostic immutable path must be a Path")
            if (
                not isinstance(file_sha256, str)
                or len(file_sha256) != 64
            ):
                raise ValueError(
                    "diagnostic immutable file hash is invalid"
                )
        if (
            type(self.source_code_files) is not tuple
            or len(self.source_code_files)
            != len(set(self.source_code_files))
            or any(
                not isinstance(name, str) or not name
                for name in self.source_code_files
            )
        ):
            raise ValueError(
                "diagnostic source-code file list must be canonical"
            )


@dataclass(frozen=True, slots=True)
class _GemmaDevelopmentCollectionRecipe:
    """Internal wiring for a development-only collection over reusable A-fit.

    Unlike ``_GemmaIterativeDiagnosticRecipe``, this recipe never constructs
    or evaluates a candidate provider.  It exists so diagnostics such as the
    exact token-loss Fisher collector can reuse the authenticated model/panel
    setup without importing, opening, or claiming any selection panel.
    """

    collect: Callable[..., Mapping[str, object]]
    validate_report: Callable[[Mapping[str, object]], None]
    publish_report: Callable[[Path, Mapping[str, object]], None]
    report_label: str
    expected_parent_lineage: Mapping[str, str] = field(
        default_factory=dict
    )
    extra_lineage: Mapping[str, str] = field(default_factory=dict)
    extra_immutable_inputs: tuple[
        tuple[str, Path, str], ...
    ] = ()
    source_code_files: tuple[str, ...] = ()
    collection_panel_factory: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        for name in ("collect", "validate_report", "publish_report"):
            if not callable(getattr(self, name)):
                raise TypeError(f"development {name} must be callable")
        if (
            self.collection_panel_factory is not None
            and not callable(self.collection_panel_factory)
        ):
            raise TypeError(
                "development collection_panel_factory must be callable"
            )
        if not isinstance(self.report_label, str) or not self.report_label:
            raise ValueError("development report label must be nonempty")
        for mapping_name in (
            "expected_parent_lineage",
            "extra_lineage",
        ):
            values = getattr(self, mapping_name)
            if not isinstance(values, Mapping):
                raise TypeError(
                    f"development {mapping_name} must be a mapping"
                )
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"development {mapping_name} key is invalid"
                    )
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                ):
                    raise ValueError(
                        f"development {mapping_name} hash is invalid"
                    )
        labels: set[str] = set()
        for label, path, file_sha256 in self.extra_immutable_inputs:
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError(
                    "development immutable labels must be unique"
                )
            labels.add(label)
            if not isinstance(path, Path):
                raise TypeError("development immutable path must be a Path")
            if (
                not isinstance(file_sha256, str)
                or len(file_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in file_sha256
                )
            ):
                raise ValueError(
                    "development immutable file hash is invalid"
                )
        if (
            type(self.source_code_files) is not tuple
            or len(self.source_code_files)
            != len(set(self.source_code_files))
            or any(
                not isinstance(name, str) or not name
                for name in self.source_code_files
            )
        ):
            raise ValueError(
                "development source-code file list must be canonical"
            )


def _position_make_fit_record(
    *,
    parent_h4: object,
    **kwargs: object,
) -> object:
    del parent_h4
    return build_gemma_iterative_residual_fit_record(**kwargs)


def _position_fit_fold(
    *,
    parent_artifact_sha256: str,
    **kwargs: object,
) -> object:
    return fit_gemma_iterative_residual_fold_provider(
        **kwargs,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def _position_fit_full(
    *,
    parent_artifact_sha256: str,
    **kwargs: object,
) -> object:
    return fit_gemma_iterative_residual_full_provider(
        **kwargs,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def _default_diagnostic_recipe() -> _GemmaIterativeDiagnosticRecipe:
    return _GemmaIterativeDiagnosticRecipe(
        campaign_recipe=(
            DEFAULT_GEMMA_ITERATIVE_RESIDUAL_CAMPAIGN_RECIPE
        ),
        make_fit_record=_position_make_fit_record,
        fit_fold=_position_fit_fold,
        build_report=build_gemma_iterative_residual_report,
        fit_full=_position_fit_full,
        validate_report=validate_gemma_iterative_residual_report,
        report_label="iterative residual",
    )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_factorial_report(
    path: Path | str,
    *,
    expected_report_sha256: str,
    expected_report_file_sha256: str,
) -> dict[str, object]:
    source = Path(path)
    if _file_sha256(source) != expected_report_file_sha256:
        raise ValueError("factorial report file hash mismatch")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("factorial report must be a JSON object")
    validate_gemma_x4_h4_factorial_report(value)
    if value.get("report_sha256") != expected_report_sha256:
        raise ValueError("factorial report logical hash mismatch")
    return value


def _factorial_lineage(
    report: Mapping[str, object],
) -> Mapping[str, object]:
    lineage = _mapping(report.get("lineage"), label="factorial lineage")
    values = _mapping(
        lineage.get("sha256s"),
        label="factorial lineage sha256s",
    )
    if set(values) != _FACTORIAL_LINEAGE_KEYS:
        raise ValueError("factorial prerequisite lineage fields differ")
    return values


def _accepted_x4_provenance(
    recollection: Mapping[str, object],
) -> Mapping[str, object]:
    provenance = _mapping(
        recollection.get("accepted_x4_provenance"),
        label="materialization accepted X4 provenance",
    )
    if set(provenance) != _ACCEPTED_X4_PROVENANCE_KEYS:
        raise ValueError(
            "materialization accepted X4 provenance fields differ"
        )
    return provenance


def _expected_factorial_lineage(
    *,
    corpus_artifact_sha256: str,
    fit_input_file_sha256: str,
    fit_manifest_sha256: str,
    fit_binding_sha256: str,
    materialization_report_sha256: str,
    materialization_report_file_sha256: str,
    accepted_x4_provenance: Mapping[str, object],
) -> dict[str, str]:
    if set(accepted_x4_provenance) != _ACCEPTED_X4_PROVENANCE_KEYS:
        raise ValueError("accepted X4 provenance fields differ")
    return {
        "corpus_artifact_sha256": corpus_artifact_sha256,
        "fit_input_file_sha256": fit_input_file_sha256,
        "fit_manifest_sha256": fit_manifest_sha256,
        "fit_binding_sha256": fit_binding_sha256,
        "materialization_report_sha256": (
            materialization_report_sha256
        ),
        "materialization_report_file_sha256": (
            materialization_report_file_sha256
        ),
        **{
            f"accepted_x4_{name}": str(accepted_x4_provenance[name])
            for name in sorted(_ACCEPTED_X4_PROVENANCE_KEYS)
        },
    }


def _validate_factorial_materialization_lineage(
    *,
    factorial: Mapping[str, object],
    materialization: GemmaH4DampingMaterialization,
    materialization_report: Mapping[str, object],
    materialization_report_file_sha256: str,
) -> Mapping[str, object]:
    """Bind the prerequisite report to the executable materialization."""

    materialization.validate_integrity()
    lineage = _factorial_lineage(factorial)
    recollection = _mapping(
        materialization_report.get("recollection"),
        label="materialization recollection",
    )
    provenance = _accepted_x4_provenance(recollection)
    expected_static = {
        "materialization_report_sha256": str(
            materialization_report.get("report_sha256")
        ),
        "materialization_report_file_sha256": (
            materialization_report_file_sha256
        ),
        **{
            f"accepted_x4_{name}": str(provenance[name])
            for name in sorted(_ACCEPTED_X4_PROVENANCE_KEYS)
        },
    }
    if any(lineage.get(name) != value for name, value in expected_static.items()):
        raise ValueError(
            "factorial prerequisite differs from materialization lineage"
        )

    accepted_artifact_sha256 = str(
        provenance["candidate_artifact_sha256"]
    )
    for artifact in (
        materialization.alpha0_artifact,
        materialization.alpha0_5_artifact,
    ):
        if artifact.parent_artifact_sha256 != accepted_artifact_sha256:
            raise ValueError(
                "factorial prerequisite and materialized H4 parent differ"
            )
    return provenance


def _validate_factorial_fit_lineage(
    *,
    factorial: Mapping[str, object],
    family_by_example: Mapping[str, str],
    model_input_sha256s: Sequence[str],
    corpus_artifact_sha256: str,
    fit_input_file_sha256: str,
    fit_manifest_sha256: str,
    fit_binding_sha256: str,
    materialization_report_sha256: str,
    materialization_report_file_sha256: str,
    accepted_x4_provenance: Mapping[str, object],
) -> None:
    """Replay the factorial validator against the exact live fit panel."""

    expected_lineage = _expected_factorial_lineage(
        corpus_artifact_sha256=corpus_artifact_sha256,
        fit_input_file_sha256=fit_input_file_sha256,
        fit_manifest_sha256=fit_manifest_sha256,
        fit_binding_sha256=fit_binding_sha256,
        materialization_report_sha256=materialization_report_sha256,
        materialization_report_file_sha256=(
            materialization_report_file_sha256
        ),
        accepted_x4_provenance=accepted_x4_provenance,
    )
    validate_gemma_x4_h4_factorial_report(
        factorial,
        expected_manifest=family_by_example,
        expected_lineage=expected_lineage,
    )
    execution = _mapping(
        factorial.get("execution"),
        label="factorial execution",
    )
    if tuple(execution.get("example_receipt_sha256s", ())) != tuple(
        sorted(model_input_sha256s)
    ):
        raise ValueError(
            "factorial prerequisite model-input grid differs from live fit"
        )


def _validate_factorial_live_lineage(
    *,
    factorial: Mapping[str, object],
    materialization: GemmaH4DampingMaterialization,
    factorized_model_sha256: str,
    factorized_execution_sha256: str,
    bridge: object,
) -> None:
    """Bind the old attribution execution to the current live runtime."""

    bridge_binding_sha256 = getattr(
        bridge,
        "bridge_binding_sha256",
        None,
    )
    bridge_float_count = getattr(
        bridge,
        "prepared_float_scalar_count",
        None,
    )
    bridge_macs = getattr(
        bridge,
        "logical_macs_per_token_upper_bound",
        None,
    )
    execution = _mapping(
        factorial.get("execution"),
        label="factorial execution",
    )
    if (
        execution.get("source_model_sha256")
        != factorized_model_sha256
        or execution.get("source_execution_sha256")
        != factorized_execution_sha256
        or execution.get("bridge_binding_sha256")
        != bridge_binding_sha256
    ):
        raise ValueError(
            "factorial prerequisite differs from live model or bridge"
        )

    alpha0 = materialization.alpha0_artifact
    alpha0_5 = materialization.alpha0_5_artifact
    for artifact in (alpha0, alpha0_5):
        if (
            artifact.live_model_sha256 != factorized_model_sha256
            or artifact.adapter_execution_sha256
            != factorized_execution_sha256
            or artifact.bridge_binding_sha256 != bridge_binding_sha256
        ):
            raise ValueError(
                "materialized factorial arm differs from live runtime"
            )
    x4_head = alpha0.head(_X4_SITE)
    lag_b = alpha0.head(_H4_SITE)
    independent = alpha0_5.head(_H4_SITE)
    if x4_head is None or lag_b is None or independent is None:
        raise ValueError("materialized factorial arm omitted a required head")
    resources = _mapping(
        factorial.get("resources"),
        label="factorial resources",
    )
    if set(resources) != _FACTORIAL_RESOURCE_KEYS:
        raise ValueError("factorial prerequisite resource fields differ")
    expected_resources = {
        "accepted_x4_head_logical_macs_per_token_upper_bound": (
            x4_head.logical_macs_per_token_upper_bound
        ),
        "accepted_x4_head_prepared_float_scalar_count": (
            x4_head.prepared_float_scalar_count
        ),
        "bridge_logical_macs_per_token_upper_bound": bridge_macs,
        "bridge_prepared_float_scalar_count": bridge_float_count,
        "independent_h4_logical_macs_per_token_upper_bound": (
            independent.logical_macs_per_token_upper_bound
        ),
        "independent_h4_prepared_float_scalar_count": (
            independent.prepared_float_scalar_count
        ),
        "lag_b_h4_logical_macs_per_token_upper_bound": (
            lag_b.logical_macs_per_token_upper_bound
        ),
        "lag_b_h4_prepared_float_scalar_count": (
            lag_b.prepared_float_scalar_count
        ),
    }
    if dict(resources) != expected_resources:
        raise ValueError(
            "factorial prerequisite resource accounting differs from live "
            "arms"
        )


def _source_code_sha256s(
    extra_names: Sequence[str] = (),
) -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = tuple(
        dict.fromkeys(
            (
                "gemma3_l3_l4_iterative_residual_diagnostic.py",
                "gemma3_l3_l4_iterative_residual_campaign.py",
                "gemma3_l3_l4_iterative_residual_boost.py",
                "gemma3_l3_l4_iterative_residual_analysis.py",
                "gemma3_l3_l4_graph_organized_svd_shadow_runtime.py",
                "gemma3_l3_l4_h4_damping_materialization.py",
                *extra_names,
            )
        )
    )
    return {name: _file_sha256(package / name) for name in names}


def run_gemma_iterative_residual_diagnostic(
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
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    _diagnostic_recipe: (
        _GemmaIterativeDiagnosticRecipe
        | _GemmaDevelopmentCollectionRecipe
        | None
    ) = None,
) -> dict[str, object]:
    """Execute the frozen four-bin LOFO iteration on expanded A-fit only."""

    diagnostic_recipe = (
        _default_diagnostic_recipe()
        if _diagnostic_recipe is None
        else _diagnostic_recipe
    )
    if not isinstance(
        diagnostic_recipe,
        (
            _GemmaIterativeDiagnosticRecipe,
            _GemmaDevelopmentCollectionRecipe,
        ),
    ):
        raise TypeError(
            "_diagnostic_recipe must be a strict diagnostic recipe"
        )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite {diagnostic_recipe.report_label} report"
        )
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
        _canonical_json_bytes(live_tokenizer_contract)
        != _canonical_json_bytes(tokenizer_contract)
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
            example.model_inputs_sha256
            for example in fit_panel.examples
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
    if (
        getattr(h4_head, "fit_manifest_sha256", None)
        != fit_panel.manifest_sha256
        or getattr(
            materialization.alpha0_5_artifact.head(_H4_SITE),
            "fit_manifest_sha256",
            None,
        )
        != fit_panel.manifest_sha256
    ):
        raise ValueError(
            "materialized factorial H4 arms differ from live fit manifest"
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
        "graph_candidate": Path(graph_candidate_path),
        "basis_package": Path(basis_package_path),
        "base_artifact": Path(base_artifact_path),
        "refit_artifact": Path(refit_artifact_path),
    }
    for label, path, _file_hash in (
        diagnostic_recipe.extra_immutable_inputs
    ):
        if label in immutable_paths:
            raise ValueError(
                "iterative extra immutable label collides with a base input"
            )
        immutable_paths[label] = path
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
        "graph_candidate": str(graph_binding["tensor_file_sha256"]),
        "basis_package": str(basis_binding["tensor_file_sha256"]),
        "base_artifact": str(recollection["base_artifact_file_sha256"]),
        "refit_artifact": str(
            recollection["refit_artifact_file_sha256"]
        ),
    }
    immutable_expected.update(
        {
            label: file_hash
            for label, _path, file_hash in (
                diagnostic_recipe.extra_immutable_inputs
            )
        }
    )
    if immutable_before != immutable_expected:
        raise ValueError("iterative immutable input binding differs")
    code_before = _source_code_sha256s(
        diagnostic_recipe.source_code_files
    )
    collection_panel = fit_panel
    if (
        isinstance(
            diagnostic_recipe,
            _GemmaDevelopmentCollectionRecipe,
        )
        and diagnostic_recipe.collection_panel_factory is not None
    ):
        collection_panel = (
            diagnostic_recipe.collection_panel_factory(
                tokenizer=tokenizer,
                tokenizer_contract=tokenizer_contract,
                parent_corpus=corpus,
                parent_fit_panel=fit_panel,
            )
        )
        collection_examples = getattr(
            collection_panel,
            "examples",
            None,
        )
        collection_families = getattr(
            collection_panel,
            "family_ids",
            None,
        )
        if (
            not isinstance(collection_examples, tuple)
            or len(collection_examples) != 16
            or not isinstance(collection_families, tuple)
            or len(collection_families) != 8
            or getattr(collection_panel, "role", None)
            != "calibration_a_fit"
            or getattr(collection_panel, "manifest_sha256", None)
            == fit_panel.manifest_sha256
        ):
            raise ValueError(
                "development collection override must be a distinct "
                "16-by-8 A-fit panel"
            )
        parent_example_ids = {
            example.example_id for example in fit_panel.examples
        }
        collection_example_ids = {
            example.example_id for example in collection_examples
        }
        if (
            parent_example_ids & collection_example_ids
            or set(fit_panel.family_ids) & set(collection_families)
        ):
            raise ValueError(
                "development collection override must be prompt- and "
                "family-disjoint from parent fit"
            )

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
        if any(
            base_lineage.get(key) != value
            for key, value in (
                diagnostic_recipe.expected_parent_lineage.items()
            )
        ):
            raise ValueError(
                "iterative prerequisite differs from the live parent lineage"
            )
        if set(base_lineage) & set(diagnostic_recipe.extra_lineage):
            raise ValueError(
                "iterative extra lineage collides with parent lineage"
            )
        lineage = {
            **base_lineage,
            **dict(diagnostic_recipe.extra_lineage),
        }
        if isinstance(
            diagnostic_recipe,
            _GemmaDevelopmentCollectionRecipe,
        ):
            collection_kwargs: dict[str, object] = {
                "panel": collection_panel,
                "adapter": adapter,
                "bridge": bridge,
                "parent_artifact": parent,
                "parent_h4": h4_head,
                "x4_head": x4_head,
                "lineage": lineage,
            }
            if collection_panel is not fit_panel:
                collection_kwargs["parent_fit_panel"] = fit_panel
            report = dict(
                diagnostic_recipe.collect(
                    **collection_kwargs,
                )
            )
        else:
            result = collect_gemma_iterative_residual_campaign_live(
                panel=fit_panel,
                adapter=adapter,
                bridge=bridge,
                parent_artifact=parent,
                make_fit_record=lambda **kwargs: (
                    diagnostic_recipe.make_fit_record(
                        parent_h4=h4_head,
                        **kwargs,
                    )
                ),
                fit_fold=lambda **kwargs: (
                    diagnostic_recipe.fit_fold(
                        **kwargs,
                        parent_artifact_sha256=parent.artifact_sha256,
                    )
                ),
                build_report=diagnostic_recipe.build_report,
                fit_full=lambda **kwargs: (
                    diagnostic_recipe.fit_full(
                        **kwargs,
                        parent_artifact_sha256=parent.artifact_sha256,
                    )
                ),
                lineage=lineage,
                recipe=diagnostic_recipe.campaign_recipe,
            )
            report = dict(result.report)
        diagnostic_recipe.validate_report(report)
        if (
            {
                name: _file_sha256(path)
                for name, path in immutable_paths.items()
            }
            != immutable_before
            or _source_code_sha256s(
                diagnostic_recipe.source_code_files
            )
            != code_before
            or adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
        ):
            raise RuntimeError(
                "iterative immutable input, code, or runtime changed"
            )
        if isinstance(
            diagnostic_recipe,
            _GemmaDevelopmentCollectionRecipe,
        ):
            diagnostic_recipe.publish_report(destination, report)
        else:
            publish_gemma_iterative_residual_campaign_report(
                destination,
                report,
            )
        return report
    finally:
        switcher.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed four-bin residual iteration on reusable A-fit."
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
    report = run_gemma_iterative_residual_diagnostic(
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
