"""Deterministic materialization of the frozen Gemma H4 damping recipe.

The fit-only damping report stores hashes, not coefficients.  This module
recollects the same accepted-X4 fit traces, reproduces those coefficients,
requires every frozen tensor hash to match, and emits exactly two executable
children:

* a matched alpha=0 L3-only H4 head containing the all-row baseline ``B``;
* the frozen alpha=0.5 independent-state H4 head.

There is no alpha search and no selection, guard, or calibration-B capability
in this materialization boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import Tensor

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
from .gemma3_l3_l4_h4_incremental_signal_diagnostic import (
    _DAMPING_ANALYSIS_DOMAIN,
    _DAMPING_FORMAT_VERSION,
    _DAMPING_REPORT_DOMAIN,
    _DAMPING_SCHEMA,
    _accepted_x4_artifact,
    _canonical_json_bytes,
    _mapping,
    _metrics,
    _prepare_lag_rows,
    _selected_weights,
    derive_candidate_h4_output_decoder,
    derive_gemma_h4_damping_recipe_tensors,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    _file_sha256 as _campaign_file_sha256,
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    load_gemma3_l3_l4_progressive_a_fit_role,
)
from .gemma3_l3_l4_progressive_worker import GemmaTwoHeadFitSequence
from .gemma3_l3_l4_progressive_worker import (
    LegacyRank64GemmaProgressiveExecutable,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
    GemmaL3L4TwoHeadArtifact,
    GemmaL3L4TwoHeadExecutable,
    _tensor_sha256,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "GemmaH4DampingMaterialization",
    "build_parser",
    "build_gemma_h4_damping_materialization",
    "load_gemma_h4_damping_materialization",
    "main",
    "publish_gemma_h4_damping_materialization",
    "run_gemma_h4_damping_materialization",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_h4_damping_materialization"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-materialization-report:v1\0"
)
_RESIDUAL_MAP_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-residual-map:v1\0"
)
_BASELINE_RECIPE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-alpha0-recipe:v1\0"
)
_ARTIFACT_RECIPE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-artifact-recipe:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_FROZEN_ALPHA = 0.5
_FROZEN_LAG_COUNT = 16
_FROZEN_INPUT_RANK = 32
_FROZEN_OUTPUT_RANK = 8
_FROZEN_RIDGE = 1.0e-6
_FACTORIZED_SCOPE = "factorized_refit"
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_DEFAULT_EXPANDED_CORPUS = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
_DEFAULT_EXPANDED_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
_DEFAULT_DAMPING_REPORT = (
    _LOCAL_ROOT
    / "progressive-a-h4-incremental-signal-damping-fit-v1.report.json"
)
_DEFAULT_ACCEPTED_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-projected-state-v6.campaign.json"
)
_DEFAULT_ACCEPTED_CANDIDATE = (
    _LOCAL_ROOT
    / "progressive-a-h4-projected-state-v6.campaign.candidate.pt"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-h4-damping-materialization-v1.report.json"
)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_tensor_free(value: object, *, path: str = "report") -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} contains a coefficient tensor")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_tensor_free(nested, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_tensor_free(nested, path=f"{path}[{index}]")


def _authenticate_damping_report(
    report: Mapping[str, object],
    *,
    expected_report_sha256: str | None,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if not isinstance(report, Mapping):
        raise TypeError("damping report must be a mapping")
    bound_report = dict(report)
    observed_report_sha256 = bound_report.pop("report_sha256", None)
    if (
        report.get("schema") != _DAMPING_SCHEMA
        or report.get("format_version") != _DAMPING_FORMAT_VERSION
        or observed_report_sha256
        != _sha256(_DAMPING_REPORT_DOMAIN, bound_report)
        or (
            expected_report_sha256 is not None
            and observed_report_sha256
            != _require_sha256(
                expected_report_sha256,
                label="expected damping report",
            )
        )
    ):
        raise ValueError("frozen damping report integrity differs")
    diagnostic = _mapping(
        report.get("diagnostic"),
        label="frozen damping diagnostic",
    )
    bound_diagnostic = dict(diagnostic)
    observed_analysis_sha256 = bound_diagnostic.pop(
        "analysis_sha256",
        None,
    )
    if (
        observed_analysis_sha256
        != _sha256(_DAMPING_ANALYSIS_DOMAIN, bound_diagnostic)
    ):
        raise ValueError("frozen damping analysis integrity differs")
    selection = _mapping(
        diagnostic.get("selection"),
        label="frozen damping selection",
    )
    recipe = _mapping(
        selection.get("winning_recipe"),
        label="frozen damping winning recipe",
    )
    spec = _mapping(report.get("spec"), label="frozen damping spec")
    fixed_head = _mapping(spec.get("fixed_head"), label="frozen fixed head")
    safety = _mapping(report.get("safety"), label="frozen damping safety")
    if (
        selection.get("status") != "fit_only_damping_recipe_frozen"
        or float(recipe.get("state_scale", -1.0)) != _FROZEN_ALPHA
        or int(recipe.get("lag_count", -1)) != _FROZEN_LAG_COUNT
        or int(recipe.get("input_rank", -1)) != _FROZEN_INPUT_RANK
        or int(recipe.get("output_rank", -1)) != _FROZEN_OUTPUT_RANK
        or recipe.get("encoder_kind")
        != "independent_crossfit_h4_svd"
        or dict(fixed_head)
        != {
            "encoder_kind": "independent_crossfit_h4_svd",
            "input_rank": _FROZEN_INPUT_RANK,
            "lag_count": _FROZEN_LAG_COUNT,
            "output_rank": _FROZEN_OUTPUT_RANK,
        }
        or float(spec.get("ridge", -1.0)) != _FROZEN_RIDGE
        or bool(spec.get("selection_input_accepted"))
        or bool(spec.get("guard_input_accepted"))
        or not bool(safety.get("fit_role_opened"))
        or bool(safety.get("selection_role_opened"))
        or bool(safety.get("guard_role_opened"))
        or bool(safety.get("calibration_b_opened"))
    ):
        raise ValueError("frozen damping promotion contract differs")
    return diagnostic, spec, recipe


def _head_metrics(
    *,
    rows: object,
    prediction_modal: Tensor,
    decoder: Tensor,
) -> dict[str, float]:
    weights = _selected_weights(
        rows,  # type: ignore[arg-type]
        torch.ones(len(rows.family_ids), dtype=torch.bool),  # type: ignore[attr-defined]
    )
    return _metrics(
        prediction_modal=prediction_modal,
        target_modal=rows.target_modal,  # type: ignore[attr-defined]
        full_residual=rows.full_residual,  # type: ignore[attr-defined]
        loss_gradient=rows.loss_gradient,  # type: ignore[attr-defined]
        decoder=decoder,
        weights=weights,
    )


def _artifact_recipe_sha256(
    *,
    alpha: float,
    parent: GemmaL3L4TwoHeadArtifact,
    parent_receipt_sha256: str,
    residual_map_sha256: str,
    analysis_artifact_sha256: str,
    frozen_recipe_sha256: str,
    h4_head_sha256: str,
) -> str:
    return _sha256(
        _ARTIFACT_RECIPE_DOMAIN,
        {
            "format_version": 1,
            "alpha": alpha,
            "parent_artifact_sha256": parent.artifact_sha256,
            "parent_receipt_sha256": parent_receipt_sha256,
            "residual_map_sha256": residual_map_sha256,
            "analysis_artifact_sha256": analysis_artifact_sha256,
            "frozen_recipe_sha256": frozen_recipe_sha256,
            "h4_head_sha256": h4_head_sha256,
            "joint_policy": "preserve_accepted_x4_then_append_h4",
            "selection_authorized": False,
        },
    )


@dataclass(frozen=True, slots=True)
class GemmaH4DampingMaterialization:
    """The matched alpha=0 control and fixed alpha=0.5 challenger."""

    alpha0_artifact: GemmaL3L4TwoHeadArtifact
    alpha0_5_artifact: GemmaL3L4TwoHeadArtifact
    report_payload: Mapping[str, object]
    report_payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report_payload, Mapping):
            raise TypeError("materialization report payload must be a mapping")
        payload = json.loads(
            _canonical_json_bytes(self.report_payload).decode("utf-8")
        )
        object.__setattr__(self, "report_payload", payload)
        object.__setattr__(
            self,
            "report_payload_sha256",
            _sha256(_REPORT_DOMAIN, payload),
        )
        self.validate_integrity()

    def validate_integrity(self) -> None:
        """Reauthenticate tensors and their scalar/hash-only report binding."""

        self.alpha0_artifact.validate_integrity()
        self.alpha0_5_artifact.validate_integrity()
        _assert_tensor_free(self.report_payload)
        if (
            self.report_payload.get("schema") != _SCHEMA
            or self.report_payload.get("format_version") != _FORMAT_VERSION
        ):
            raise ValueError("materialization report schema differs")
        for artifact in (self.alpha0_artifact, self.alpha0_5_artifact):
            if artifact.head(_X4_SITE) is None or artifact.head(_H4_SITE) is None:
                raise ValueError("materialized artifact must contain X4 and H4")
        if (
            self.alpha0_artifact.artifact_sha256
            == self.alpha0_5_artifact.artifact_sha256
        ):
            raise ValueError("materialized control and challenger must differ")
        if (
            _sha256(_REPORT_DOMAIN, self.report_payload)
            != self.report_payload_sha256
        ):
            raise RuntimeError("materialization report payload drifted")
        alpha0_head = self.alpha0_artifact.head(_H4_SITE)
        alpha0_5_head = self.alpha0_5_artifact.head(_H4_SITE)
        assert alpha0_head is not None
        assert alpha0_5_head is not None
        alpha0_x4 = self.alpha0_artifact.head(_X4_SITE)
        alpha0_5_x4 = self.alpha0_5_artifact.head(_X4_SITE)
        assert alpha0_x4 is not None
        assert alpha0_5_x4 is not None
        if (
            self.alpha0_artifact.parent_artifact_sha256
            != self.alpha0_5_artifact.parent_artifact_sha256
            or self.alpha0_artifact.parent_receipt_sha256
            != self.alpha0_5_artifact.parent_receipt_sha256
            or self.alpha0_artifact.bridge_binding_sha256
            != self.alpha0_5_artifact.bridge_binding_sha256
            or self.alpha0_artifact.live_model_sha256
            != self.alpha0_5_artifact.live_model_sha256
            or self.alpha0_artifact.adapter_execution_sha256
            != self.alpha0_5_artifact.adapter_execution_sha256
            or alpha0_x4.artifact_sha256 != alpha0_5_x4.artifact_sha256
            or alpha0_head.conditioning != "l3_source_modes"
            or alpha0_head.state_encoder is not None
            or alpha0_5_head.conditioning
            != "l3_source_modes_plus_independent_realized_h4_modes_v1"
            or alpha0_5_head.state_encoder is None
        ):
            raise ValueError("materialized arm semantics differ")
        artifacts = _mapping(
            self.report_payload.get("artifacts"),
            label="materialization artifact metadata",
        )
        alpha0_metadata = _mapping(
            artifacts.get("matched_alpha0"),
            label="matched alpha0 metadata",
        )
        alpha0_5_metadata = _mapping(
            artifacts.get("challenger_alpha0_5"),
            label="challenger alpha0.5 metadata",
        )
        accepted_metadata = _mapping(
            artifacts.get("accepted_x4_only"),
            label="accepted X4 metadata",
        )

        def check_artifact(
            artifact: GemmaL3L4TwoHeadArtifact,
            head: GemmaCausalResidualHead,
            metadata: Mapping[str, object],
        ) -> bool:
            return (
                metadata.get("artifact_sha256")
                == artifact.artifact_sha256
                and metadata.get("execution_sha256")
                == artifact.execution_sha256
                and metadata.get("runtime_binding_sha256")
                == artifact.runtime_binding_sha256
                and metadata.get("h4_head_sha256")
                == head.artifact_sha256
                and metadata.get("h4_prepared_float_scalar_count")
                == head.prepared_float_scalar_count
                and metadata.get(
                    "h4_logical_macs_per_token_upper_bound"
                )
                == head.logical_macs_per_token_upper_bound
            )

        if (
            not check_artifact(
                self.alpha0_artifact,
                alpha0_head,
                alpha0_metadata,
            )
            or not check_artifact(
                self.alpha0_5_artifact,
                alpha0_5_head,
                alpha0_5_metadata,
            )
            or accepted_metadata.get("artifact_sha256")
            != self.alpha0_artifact.parent_artifact_sha256
            or accepted_metadata.get("h4_prepared_float_scalar_count") != 0
            or accepted_metadata.get(
                "h4_logical_macs_per_token_upper_bound"
            )
            != 0
        ):
            raise ValueError("materialization report artifact binding differs")
        lineage = _mapping(
            self.report_payload.get("lineage"),
            label="materialization lineage",
        )
        if (
            lineage.get("accepted_x4_artifact_sha256")
            != self.alpha0_artifact.parent_artifact_sha256
            or lineage.get("accepted_x4_receipt_sha256")
            != self.alpha0_artifact.parent_receipt_sha256
            or lineage.get("damping_analysis_sha256")
            != alpha0_head.analysis_artifact_sha256
            or lineage.get("fit_manifest_sha256")
            != alpha0_head.fit_manifest_sha256
            or tuple(lineage.get("fit_sequence_sha256s", ()))
            != alpha0_head.fit_sequence_sha256s
            or tuple(lineage.get("family_ids", ()))
            != alpha0_head.family_ids
            or lineage.get("fit_row_count") != alpha0_head.fit_row_count
            or alpha0_head.fit_sequence_sha256s
            != alpha0_5_head.fit_sequence_sha256s
            or alpha0_head.family_ids != alpha0_5_head.family_ids
            or alpha0_head.fit_row_count != alpha0_5_head.fit_row_count
            or accepted_metadata.get("artifact_sha256")
            != lineage.get("accepted_x4_artifact_sha256")
            or accepted_metadata.get("execution_sha256")
            != lineage.get("accepted_x4_execution_sha256")
            or accepted_metadata.get("runtime_binding_sha256")
            != lineage.get("accepted_x4_runtime_binding_sha256")
        ):
            raise ValueError("materialization report lineage differs")
        coefficients = _mapping(
            self.report_payload.get("coefficient_reproduction"),
            label="materialization coefficient reproduction",
        )
        if (
            coefficients.get("status") != "exact_hash_match"
            or coefficients.get("decoder_sha256")
            != _tensor_sha256(alpha0_5_head.decoder)
            or alpha0_5_head.state_encoder is None
            or coefficients.get("state_encoder_sha256")
            != _tensor_sha256(alpha0_5_head.state_encoder)
            or coefficients.get("state_kernel_sha256")
            != _tensor_sha256(alpha0_5_head.state_kernel)
            or coefficients.get("stored_lag_coefficients_sha256")
            != _tensor_sha256(
                alpha0_5_head.lag_kernel.reshape(
                    -1,
                    alpha0_5_head.rank,
                )
            )
            or coefficients.get("baseline_lag_coefficients_sha256")
            != _tensor_sha256(
                alpha0_head.lag_kernel.reshape(-1, alpha0_head.rank)
            )
        ):
            raise ValueError(
                "materialization report coefficient binding differs"
            )
        safety = _mapping(
            self.report_payload.get("safety"),
            label="materialization safety metadata",
        )
        if dict(safety) != {
            "fit_only_inputs_opened": True,
            "selection_input_capability_present": False,
            "selection_role_opened": False,
            "guard_input_capability_present": False,
            "guard_role_opened": False,
            "calibration_b_loader_present": False,
            "calibration_b_opened": False,
            "alternate_alpha_fallback_present": False,
            "prompt_text_in_report": False,
            "token_ids_in_report": False,
            "logits_in_report": False,
            "activation_rows_in_report": False,
            "gradient_rows_in_report": False,
            "fit_sequences_in_report": False,
            "coefficient_tensors_in_report": False,
            "model_weights_in_artifacts": False,
            "selection_claim": False,
            "compression_claim": False,
            "latency_claim": False,
        }:
            raise ValueError("materialization safety metadata differs")


def build_gemma_h4_damping_materialization(
    *,
    sequences: Sequence[GemmaTwoHeadFitSequence],
    output_decoder: Tensor,
    accepted_x4_artifact: GemmaL3L4TwoHeadArtifact,
    accepted_x4_receipt_sha256: str,
    damping_report: Mapping[str, object],
    expected_damping_report_sha256: str | None = None,
) -> GemmaH4DampingMaterialization:
    """Reproduce and hash-check the only frozen damping candidate."""

    parent_receipt = _require_sha256(
        accepted_x4_receipt_sha256,
        label="accepted X4 receipt",
    )
    accepted_x4_artifact.validate_integrity()
    accepted_x4 = accepted_x4_artifact.head(_X4_SITE)
    if accepted_x4 is None or accepted_x4_artifact.head(_H4_SITE) is not None:
        raise ValueError("parent artifact must be the accepted X4-only winner")
    diagnostic, spec, frozen_recipe = _authenticate_damping_report(
        damping_report,
        expected_report_sha256=expected_damping_report_sha256,
    )
    accepted_provenance = _mapping(
        damping_report.get("accepted_x4_provenance"),
        label="frozen accepted X4 provenance",
    )
    if (
        accepted_provenance.get("candidate_artifact_sha256")
        != accepted_x4_artifact.artifact_sha256
        or accepted_provenance.get("candidate_execution_sha256")
        != accepted_x4_artifact.execution_sha256
        or accepted_provenance.get("candidate_runtime_binding_sha256")
        != accepted_x4_artifact.runtime_binding_sha256
    ):
        raise ValueError("frozen report and accepted X4 parent differ")
    ordered = tuple(
        sorted(
            (sequence.detached_copy() for sequence in sequences),
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    if (
        not ordered
        or len({value.artifact_sha256 for value in ordered}) != len(ordered)
        or any(
            value.runtime_binding_sha256
            != accepted_x4_artifact.runtime_binding_sha256
            for value in ordered
        )
    ):
        raise ValueError("materialization fit traces are invalid")
    diagnostic_input = _mapping(
        diagnostic.get("input"),
        label="frozen damping input",
    )
    sequence_sha256s = tuple(
        sorted(value.artifact_sha256 for value in ordered)
    )
    family_ids = tuple(sorted({value.family_id for value in ordered}))
    if (
        tuple(sorted(diagnostic_input.get("fit_sequence_sha256s", ())))
        != sequence_sha256s
        or tuple(diagnostic_input.get("family_ids", ())) != family_ids
        or int(diagnostic_input.get("affected_row_count", -1))
        != sum(value.affected_rows for value in ordered)
        or spec.get("fit_manifest_sha256")
        != diagnostic_input.get("fit_manifest_sha256")
        and "fit_manifest_sha256" in diagnostic_input
    ):
        raise ValueError("materialization fit trace binding differs")
    tensors, reproduced_recipe = derive_gemma_h4_damping_recipe_tensors(
        sequences=ordered,
        output_decoder=output_decoder,
        lag_count=_FROZEN_LAG_COUNT,
        input_rank=_FROZEN_INPUT_RANK,
        state_scale=_FROZEN_ALPHA,
        ridge=_FROZEN_RIDGE,
    )
    if _canonical_json_bytes(reproduced_recipe) != _canonical_json_bytes(
        frozen_recipe
    ):
        raise RuntimeError("frozen alpha=0.5 coefficient hashes did not reproduce")
    rows = _prepare_lag_rows(
        ordered,
        decoder=tensors.decoder,
        lag_count=tensors.lag_count,
    )
    alpha0_prediction = rows.design @ tensors.baseline_lag_coefficients
    alpha0_5_prediction = (
        rows.design @ tensors.stored_lag_coefficients
        + (rows.realized_h4 @ tensors.state_encoder.T)
        @ tensors.state_kernel
    )
    alpha0_metrics = _head_metrics(
        rows=rows,
        prediction_modal=alpha0_prediction,
        decoder=tensors.decoder,
    )
    alpha0_5_metrics = _head_metrics(
        rows=rows,
        prediction_modal=alpha0_5_prediction,
        decoder=tensors.decoder,
    )
    analysis_sha256 = _require_sha256(
        diagnostic.get("analysis_sha256"),
        label="damping analysis",
    )
    fit_manifest_sha256 = _require_sha256(
        spec.get("fit_manifest_sha256"),
        label="expanded fit manifest",
    )
    residual_map_sha256 = _sha256(
        _RESIDUAL_MAP_DOMAIN,
        {
            "format_version": 1,
            "damping_analysis_sha256": analysis_sha256,
            "output_decoder_sha256": _tensor_sha256(tensors.decoder),
            "output_rank": tensors.output_rank,
            "site": _H4_SITE,
        },
    )
    baseline_recipe = {
        "format_version": 1,
        "alpha": 0.0,
        "lag_count": tensors.lag_count,
        "output_rank": tensors.output_rank,
        "decoder_sha256": _tensor_sha256(tensors.decoder),
        "baseline_lag_coefficients_sha256": _tensor_sha256(
            tensors.baseline_lag_coefficients
        ),
        "runtime_formula": "(lagged_l3 @ baseline_lag) @ decoder",
        "matched_to_frozen_recipe_sha256": frozen_recipe["recipe_sha256"],
    }
    baseline_recipe_sha256 = _sha256(
        _BASELINE_RECIPE_DOMAIN,
        baseline_recipe,
    )
    common_head = {
        "site": _H4_SITE,
        "parent_runtime_binding_sha256": (
            accepted_x4_artifact.runtime_binding_sha256
        ),
        "residual_map_sha256": residual_map_sha256,
        "analysis_artifact_sha256": analysis_sha256,
        "fit_manifest_sha256": fit_manifest_sha256,
        "bridge_binding_sha256": accepted_x4_artifact.bridge_binding_sha256,
        "decoder": tensors.decoder,
        "ridge": _FROZEN_RIDGE,
        "fit_row_count": int(rows.design.shape[0]),
        "family_ids": family_ids,
        "fit_sequence_sha256s": sequence_sha256s,
        "fit_objective": "candidate_nll_vjp_metric_ridge_v1",
    }
    alpha0_head = GemmaCausalResidualHead(
        **common_head,
        lag_kernel=tensors.baseline_lag_kernel,
        state_encoder=torch.empty((0, 0), dtype=torch.float64),
        state_kernel=torch.empty((0, 0), dtype=torch.float64),
        conditioning="l3_source_modes",
        weighted_residual_rmse=alpha0_metrics[
            "projected_residual_rmse"
        ],
        normalized_nll_direction_rmse=alpha0_metrics[
            "normalized_nll_direction_rmse"
        ],
        linearized_nll_residual_rmse=alpha0_metrics[
            "linearized_nll_residual_rmse"
        ],
    )
    alpha0_5_head = GemmaCausalResidualHead(
        **common_head,
        lag_kernel=tensors.lag_kernel,
        state_encoder=tensors.state_encoder,
        state_kernel=tensors.state_kernel,
        conditioning=(
            "l3_source_modes_plus_independent_realized_h4_modes_v1"
        ),
        weighted_residual_rmse=alpha0_5_metrics[
            "projected_residual_rmse"
        ],
        normalized_nll_direction_rmse=alpha0_5_metrics[
            "normalized_nll_direction_rmse"
        ],
        linearized_nll_residual_rmse=alpha0_5_metrics[
            "linearized_nll_residual_rmse"
        ],
    )

    def child(
        *,
        alpha: float,
        head: GemmaCausalResidualHead,
        frozen_recipe_sha256: str,
    ) -> GemmaL3L4TwoHeadArtifact:
        recipe_sha256 = _artifact_recipe_sha256(
            alpha=alpha,
            parent=accepted_x4_artifact,
            parent_receipt_sha256=parent_receipt,
            residual_map_sha256=residual_map_sha256,
            analysis_artifact_sha256=analysis_sha256,
            frozen_recipe_sha256=frozen_recipe_sha256,
            h4_head_sha256=head.artifact_sha256,
        )
        return GemmaL3L4TwoHeadArtifact(
            parent_artifact_sha256=accepted_x4_artifact.artifact_sha256,
            parent_receipt_sha256=parent_receipt,
            residual_map_sha256=residual_map_sha256,
            analysis_artifact_sha256=analysis_sha256,
            bridge_binding_sha256=accepted_x4_artifact.bridge_binding_sha256,
            live_model_sha256=accepted_x4_artifact.live_model_sha256,
            adapter_execution_sha256=(
                accepted_x4_artifact.adapter_execution_sha256
            ),
            heads=(accepted_x4, head),
            recipe_sha256=recipe_sha256,
        )

    alpha0_artifact = child(
        alpha=0.0,
        head=alpha0_head,
        frozen_recipe_sha256=baseline_recipe_sha256,
    )
    alpha0_5_artifact = child(
        alpha=_FROZEN_ALPHA,
        head=alpha0_5_head,
        frozen_recipe_sha256=_require_sha256(
            frozen_recipe.get("recipe_sha256"),
            label="frozen damping recipe",
        ),
    )
    report_payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "contract": {
            "operation": "deterministic_frozen_recipe_materialization",
            "candidate_alphas": (0.0, _FROZEN_ALPHA),
            "selected_alpha": _FROZEN_ALPHA,
            "alpha_search_performed": False,
            "matched_control": "accepted_x4_plus_all_row_B_lag_head",
            "challenger": "accepted_x4_plus_frozen_alpha0_5_head",
        },
        "lineage": {
            "damping_report_sha256": damping_report["report_sha256"],
            "damping_analysis_sha256": analysis_sha256,
            "damping_recipe_sha256": frozen_recipe["recipe_sha256"],
            "accepted_x4_artifact_sha256": (
                accepted_x4_artifact.artifact_sha256
            ),
            "accepted_x4_execution_sha256": (
                accepted_x4_artifact.execution_sha256
            ),
            "accepted_x4_runtime_binding_sha256": (
                accepted_x4_artifact.runtime_binding_sha256
            ),
            "accepted_x4_receipt_sha256": parent_receipt,
            "fit_manifest_sha256": fit_manifest_sha256,
            "fit_sequence_sha256s": sequence_sha256s,
            "family_ids": family_ids,
            "fit_row_count": int(rows.design.shape[0]),
        },
        "coefficient_reproduction": {
            "status": "exact_hash_match",
            "decoder_sha256": _tensor_sha256(tensors.decoder),
            "state_encoder_sha256": _tensor_sha256(
                tensors.state_encoder
            ),
            "state_kernel_sha256": _tensor_sha256(tensors.state_kernel),
            "stored_lag_coefficients_sha256": _tensor_sha256(
                tensors.stored_lag_coefficients
            ),
            "baseline_lag_coefficients_sha256": _tensor_sha256(
                tensors.baseline_lag_coefficients
            ),
            "baseline_recipe_sha256": baseline_recipe_sha256,
        },
        "artifacts": {
            "accepted_x4_only": {
                "artifact_sha256": accepted_x4_artifact.artifact_sha256,
                "execution_sha256": accepted_x4_artifact.execution_sha256,
                "runtime_binding_sha256": (
                    accepted_x4_artifact.runtime_binding_sha256
                ),
                "h4_prepared_float_scalar_count": 0,
                "h4_logical_macs_per_token_upper_bound": 0,
            },
            "matched_alpha0": {
                "artifact_sha256": alpha0_artifact.artifact_sha256,
                "execution_sha256": alpha0_artifact.execution_sha256,
                "runtime_binding_sha256": (
                    alpha0_artifact.runtime_binding_sha256
                ),
                "h4_head_sha256": alpha0_head.artifact_sha256,
                "h4_prepared_float_scalar_count": (
                    alpha0_head.prepared_float_scalar_count
                ),
                "h4_logical_macs_per_token_upper_bound": (
                    alpha0_head.logical_macs_per_token_upper_bound
                ),
                "fit_metrics": alpha0_metrics,
            },
            "challenger_alpha0_5": {
                "artifact_sha256": alpha0_5_artifact.artifact_sha256,
                "execution_sha256": alpha0_5_artifact.execution_sha256,
                "runtime_binding_sha256": (
                    alpha0_5_artifact.runtime_binding_sha256
                ),
                "h4_head_sha256": alpha0_5_head.artifact_sha256,
                "h4_prepared_float_scalar_count": (
                    alpha0_5_head.prepared_float_scalar_count
                ),
                "h4_logical_macs_per_token_upper_bound": (
                    alpha0_5_head.logical_macs_per_token_upper_bound
                ),
                "fit_metrics": alpha0_5_metrics,
            },
        },
        "safety": {
            "fit_only_inputs_opened": True,
            "selection_input_capability_present": False,
            "selection_role_opened": False,
            "guard_input_capability_present": False,
            "guard_role_opened": False,
            "calibration_b_loader_present": False,
            "calibration_b_opened": False,
            "alternate_alpha_fallback_present": False,
            "prompt_text_in_report": False,
            "token_ids_in_report": False,
            "logits_in_report": False,
            "activation_rows_in_report": False,
            "gradient_rows_in_report": False,
            "fit_sequences_in_report": False,
            "coefficient_tensors_in_report": False,
            "model_weights_in_artifacts": False,
            "selection_claim": False,
            "compression_claim": False,
            "latency_claim": False,
        },
    }
    _assert_tensor_free(report_payload)
    return GemmaH4DampingMaterialization(
        alpha0_artifact=alpha0_artifact,
        alpha0_5_artifact=alpha0_5_artifact,
        report_payload=report_payload,
    )


def _stage_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def publish_gemma_h4_damping_materialization(
    materialization: GemmaH4DampingMaterialization,
    *,
    output: Path | str,
) -> dict[str, object]:
    """Atomically publish two tensor-only artifacts and one scalar report."""

    if not isinstance(materialization, GemmaH4DampingMaterialization):
        raise TypeError("materialization has the wrong type")
    materialization.validate_integrity()
    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("materialization report output must end in .json")
    alpha0_path = destination.with_suffix(".alpha0.candidate.pt")
    alpha0_5_path = destination.with_suffix(".alpha0_5.candidate.pt")
    paths = (alpha0_path, alpha0_5_path, destination)
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite materialization output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    alpha0_stage = _stage_path(alpha0_path)
    alpha0_5_stage = _stage_path(alpha0_5_path)
    report_stage = _stage_path(destination)
    published: list[Path] = []
    try:
        torch.save(
            materialization.alpha0_artifact.state_dict(),
            alpha0_stage,
        )
        torch.save(
            materialization.alpha0_5_artifact.state_dict(),
            alpha0_5_stage,
        )
        for stage, expected in (
            (alpha0_stage, materialization.alpha0_artifact),
            (alpha0_5_stage, materialization.alpha0_5_artifact),
        ):
            raw = torch.load(stage, map_location="cpu", weights_only=True)
            if not isinstance(raw, Mapping):
                raise RuntimeError("staged materialization state is not a mapping")
            restored = GemmaL3L4TwoHeadArtifact.from_state_dict(raw)
            if restored.artifact_sha256 != expected.artifact_sha256:
                raise RuntimeError("staged materialization failed reauthentication")
        materialization.validate_integrity()
        report: dict[str, object] = {
            **dict(materialization.report_payload),
            "files": {
                "matched_alpha0": {
                    "tensor_file": str(alpha0_path),
                    "tensor_file_sha256": _file_sha256(alpha0_stage),
                    "tensor_file_bytes": alpha0_stage.stat().st_size,
                },
                "challenger_alpha0_5": {
                    "tensor_file": str(alpha0_5_path),
                    "tensor_file_sha256": _file_sha256(alpha0_5_stage),
                    "tensor_file_bytes": alpha0_5_stage.stat().st_size,
                },
                "contains_model_weights": False,
                "committable": False,
            },
        }
        report["report_sha256"] = _sha256(_REPORT_DOMAIN, report)
        _assert_tensor_free(report)
        with report_stage.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for stage, path in (
            (alpha0_stage, alpha0_path),
            (alpha0_5_stage, alpha0_5_path),
            (report_stage, destination),
        ):
            os.link(stage, path)
            published.append(path)
        return report
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        alpha0_stage.unlink(missing_ok=True)
        alpha0_5_stage.unlink(missing_ok=True)
        report_stage.unlink(missing_ok=True)


def load_gemma_h4_damping_materialization(
    report_path: Path | str,
    *,
    expected_report_sha256: str | None = None,
    expected_report_file_sha256: str | None = None,
) -> tuple[GemmaH4DampingMaterialization, Mapping[str, object]]:
    """Strict-load and reauthenticate both published executable arms."""

    source = Path(report_path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("materialization report must be a regular file")
    observed_file_sha256 = _file_sha256(source)
    if (
        expected_report_file_sha256 is not None
        and observed_file_sha256
        != _require_sha256(
            expected_report_file_sha256,
            label="expected materialization report file",
        )
    ):
        raise ValueError("materialization report file hash differs")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("materialization report must contain an object")
    report = dict(raw)
    observed_report_sha256 = report.pop("report_sha256", None)
    if (
        raw.get("schema") != _SCHEMA
        or raw.get("format_version") != _FORMAT_VERSION
        or observed_report_sha256 != _sha256(_REPORT_DOMAIN, report)
        or (
            expected_report_sha256 is not None
            and observed_report_sha256
            != _require_sha256(
                expected_report_sha256,
                label="expected materialization report",
            )
        )
    ):
        raise ValueError("materialization report integrity differs")
    files = _mapping(report.pop("files", None), label="materialization files")
    if (
        bool(files.get("contains_model_weights"))
        or bool(files.get("committable"))
    ):
        raise ValueError("materialization file safety metadata differs")

    def load_arm(
        name: str,
        *,
        report_name: str,
    ) -> GemmaL3L4TwoHeadArtifact:
        record = _mapping(files.get(name), label=f"{name} tensor file")
        tensor_path = Path(str(record.get("tensor_file")))
        if tensor_path.is_symlink() or not tensor_path.is_file():
            raise ValueError(f"{name} tensor artifact must be a regular file")
        if (
            _file_sha256(tensor_path)
            != _require_sha256(
                record.get("tensor_file_sha256"),
                label=f"{name} tensor file",
            )
            or tensor_path.stat().st_size
            != int(record.get("tensor_file_bytes", -1))
        ):
            raise ValueError(f"{name} tensor file binding differs")
        state = torch.load(
            tensor_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, Mapping):
            raise ValueError(f"{name} tensor state must be a mapping")
        artifact = GemmaL3L4TwoHeadArtifact.from_state_dict(state)
        metadata = _mapping(
            _mapping(
                report.get("artifacts"),
                label="materialization artifacts",
            ).get(report_name),
            label=f"{name} artifact metadata",
        )
        if (
            artifact.artifact_sha256
            != metadata.get("artifact_sha256")
            or artifact.execution_sha256
            != metadata.get("execution_sha256")
            or artifact.runtime_binding_sha256
            != metadata.get("runtime_binding_sha256")
            or artifact.head(_X4_SITE) is None
            or artifact.head(_H4_SITE) is None
        ):
            raise ValueError(f"{name} executable binding differs")
        return artifact

    alpha0 = load_arm("matched_alpha0", report_name="matched_alpha0")
    alpha0_5 = load_arm(
        "challenger_alpha0_5",
        report_name="challenger_alpha0_5",
    )
    if (
        alpha0.parent_artifact_sha256
        != alpha0_5.parent_artifact_sha256
        or alpha0.parent_receipt_sha256
        != alpha0_5.parent_receipt_sha256
        or alpha0.bridge_binding_sha256
        != alpha0_5.bridge_binding_sha256
        or alpha0.live_model_sha256 != alpha0_5.live_model_sha256
        or alpha0.adapter_execution_sha256
        != alpha0_5.adapter_execution_sha256
        or alpha0.head(_H4_SITE).conditioning != "l3_source_modes"
        or alpha0_5.head(_H4_SITE).conditioning
        != "l3_source_modes_plus_independent_realized_h4_modes_v1"
    ):
        raise ValueError("materialized arm semantics differ")
    materialization = GemmaH4DampingMaterialization(
        alpha0_artifact=alpha0,
        alpha0_5_artifact=alpha0_5,
        report_payload=report,
    )
    return materialization, raw


def _accepted_candidate_receipt(
    report: Mapping[str, object],
) -> str:
    result = _mapping(report.get("result"), label="accepted campaign result")
    iterations = result.get("iterations")
    if (
        isinstance(iterations, (str, bytes))
        or not isinstance(iterations, Sequence)
    ):
        raise ValueError("accepted campaign iterations are invalid")
    receipts = tuple(
        _require_sha256(
            receipt,
            label="accepted candidate receipt",
        )
        for raw in iterations
        for receipt in (
            _mapping(raw, label="accepted campaign iteration").get(
                "accepted_candidate_receipt_sha256"
            ),
        )
        if receipt is not None
    )
    if len(receipts) != 1:
        raise ValueError("accepted X4 campaign lacks one accepted receipt")
    return receipts[0]


def _source_code_sha256s() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "gemma3_l3_l4_h4_damping_materialization.py",
        "gemma3_l3_l4_h4_incremental_signal_diagnostic.py",
        "gemma3_l3_l4_progressive_a_corpus.py",
        "gemma3_l3_l4_progressive_worker.py",
        "gemma3_l3_l4_two_head_lowerer.py",
        "gemma3_l3_l4_graph_organized_svd_shadow_runtime.py",
    )
    return {name: _file_sha256(package / name) for name in names}


def run_gemma_h4_damping_materialization(
    *,
    corpus_artifact_path: Path | str = _DEFAULT_EXPANDED_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    damping_report_path: Path | str = _DEFAULT_DAMPING_REPORT,
    expected_damping_report_sha256: str,
    expected_damping_report_file_sha256: str,
    accepted_x4_report_path: Path | str = _DEFAULT_ACCEPTED_REPORT,
    accepted_x4_candidate_path: Path | str = _DEFAULT_ACCEPTED_CANDIDATE,
    expected_accepted_x4_candidate_file_sha256: str,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Recollect the fixed A-fit traces and publish both executable arms."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite materialization output")
    damping_path = Path(damping_report_path)
    if (
        _campaign_file_sha256(damping_path)
        != _require_sha256(
            expected_damping_report_file_sha256,
            label="expected damping report file",
        )
    ):
        raise ValueError("frozen damping report file hash differs")
    raw_damping = json.loads(damping_path.read_text(encoding="utf-8"))
    if not isinstance(raw_damping, Mapping):
        raise ValueError("frozen damping report must contain an object")
    _authenticate_damping_report(
        raw_damping,
        expected_report_sha256=expected_damping_report_sha256,
    )
    accepted_report, accepted_artifact, accepted_provenance = (
        _accepted_x4_artifact(
            report_path=accepted_x4_report_path,
            candidate_path=accepted_x4_candidate_path,
            expected_candidate_file_sha256=(
                expected_accepted_x4_candidate_file_sha256
            ),
        )
    )
    frozen_accepted_provenance = _mapping(
        raw_damping.get("accepted_x4_provenance"),
        label="frozen accepted X4 provenance",
    )
    if _canonical_json_bytes(
        frozen_accepted_provenance
    ) != _canonical_json_bytes(accepted_provenance):
        raise ValueError("accepted X4 provenance differs from frozen damping")
    accepted_receipt = _accepted_candidate_receipt(accepted_report)
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    protocol.validate_integrity()
    protocol_metadata = protocol.metadata()
    tokenizer_contract = dict(
        _mapping(protocol_metadata["tokenizer"], label="frozen tokenizer")
    )
    max_length = int(tokenizer_contract["max_length"])
    corpus, fit_input = load_gemma3_l3_l4_progressive_a_fit_role(
        corpus_artifact_path,
        fit_input_path=fit_input_path,
        tokenizer_contract=tokenizer_contract,
    )
    damping_spec = _mapping(
        raw_damping.get("spec"),
        label="frozen damping spec",
    )
    if corpus.artifact_sha256 != damping_spec.get(
        "corpus_artifact_sha256"
    ):
        raise ValueError("expanded fit corpus differs from frozen damping")
    tokenizer, live_tokenizer_contract = (
        _load_and_validate_frozen_local_tokenizer(protocol=protocol)
    )
    if _canonical_json_bytes(
        live_tokenizer_contract
    ) != _canonical_json_bytes(tokenizer_contract):
        raise ValueError("live tokenizer differs from materialization contract")
    fit_panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=fit_input,
        view=corpus.role_view("calibration_a_fit"),
        max_length=max_length,
        device=torch.device(str(tokenizer_contract["device"])),
        forbidden_manifest_sha256s=(
            corpus.forbidden_assessment_manifest_sha256s
        ),
    )
    if (
        fit_panel.manifest_sha256
        != damping_spec.get("fit_manifest_sha256")
        or fit_panel.binding_sha256
        != damping_spec.get("fit_binding_sha256")
        or len(fit_panel.examples) != 16
        or len({value.family_id for value in fit_panel.examples}) != 8
    ):
        raise ValueError("expanded fit panel differs from frozen damping")
    model_metadata = _mapping(
        protocol_metadata["model"],
        label="frozen Gemma model",
    )
    graph_binding = _mapping(
        protocol_metadata["graph_candidate"],
        label="frozen graph candidate",
    )
    basis_binding = _mapping(
        protocol_metadata["prompt_blind_basis"],
        label="frozen basis",
    )
    code_before = _source_code_sha256s()
    graph_path = Path(graph_candidate_path)
    basis_path = Path(basis_package_path)
    base_path = Path(base_artifact_path)
    refit_path = Path(refit_artifact_path)
    frozen_input_file_sha256s = {
        "graph_candidate": _campaign_file_sha256(graph_path),
        "basis": _campaign_file_sha256(basis_path),
        "base_artifact": _campaign_file_sha256(base_path),
        "refit_artifact": _campaign_file_sha256(refit_path),
    }
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
    if adapter.model_fingerprint() != str(
        model_metadata["source_model_sha256"]
    ):
        raise ValueError("live raw Gemma differs from the frozen source")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_path,
        refit_path,
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
            != str(graph_binding["factorized_live_execution_sha256"])
            or factorized_execution_sha256
            != str(graph_binding["factorized_refit_execution_sha256"])
        ):
            raise ValueError("live factorized Gemma differs from frozen execution")
        graph_candidate = load_gemma3_graph_organized_svd_candidate(
            graph_path,
            expected_file_sha256=str(graph_binding["tensor_file_sha256"]),
        )
        basis = load_gemma3_l3_l4_basis_package(
            basis_path,
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
        source_probe = LegacyRank64GemmaProgressiveExecutable(
            adapter=adapter,
            runtime=runtime,
            candidate_execution_sha256=factorized_execution_sha256,
        )
        executable = GemmaL3L4TwoHeadExecutable(
            adapter=adapter,
            shadow_runtime=runtime,
            bridge=runtime.export_one_pass_bridge(),
            source_probe=source_probe,
            artifact=accepted_artifact,
        )
        sequences: list[GemmaTwoHeadFitSequence] = []
        for example in fit_panel.examples:
            observation = executable.observe(
                example,
                collect_carrier_fisher=True,
            )
            sequence = observation.two_head_fit_sequence
            if sequence is None:
                raise RuntimeError("accepted X4 omitted a materialization trace")
            sequences.append(sequence.detached_copy())
        decoder, decoder_audit = derive_candidate_h4_output_decoder(
            sequences,
            output_rank=_FROZEN_OUTPUT_RANK,
        )
        frozen_recipe = _mapping(
            _mapping(
                _mapping(
                    raw_damping.get("diagnostic"),
                    label="frozen damping diagnostic",
                ).get("selection"),
                label="frozen damping selection",
            ).get("winning_recipe"),
            label="frozen damping recipe",
        )
        if decoder_audit.get("decoder_sha256") != frozen_recipe.get(
            "decoder_sha256"
        ):
            raise RuntimeError("materialized output decoder hash differs")
        materialization = build_gemma_h4_damping_materialization(
            sequences=sequences,
            output_decoder=decoder,
            accepted_x4_artifact=accepted_artifact,
            accepted_x4_receipt_sha256=accepted_receipt,
            damping_report=raw_damping,
            expected_damping_report_sha256=(
                expected_damping_report_sha256
            ),
        )
        code_after = _source_code_sha256s()
        if code_after != code_before:
            raise RuntimeError("materialization source changed during run")
        if (
            adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
            or _campaign_file_sha256(graph_path)
            != str(graph_binding["tensor_file_sha256"])
            or _campaign_file_sha256(basis_path)
            != str(basis_binding["tensor_file_sha256"])
            or _campaign_file_sha256(base_path)
            != frozen_input_file_sha256s["base_artifact"]
            or _campaign_file_sha256(refit_path)
            != frozen_input_file_sha256s["refit_artifact"]
        ):
            raise RuntimeError(
                "materialization model or frozen artifacts changed during run"
            )
        enriched_payload = {
            **dict(materialization.report_payload),
            "recollection": {
                "kind": "live_model_fit_only",
                "corpus_artifact_sha256": corpus.artifact_sha256,
                "fit_manifest_sha256": fit_panel.manifest_sha256,
                "fit_binding_sha256": fit_panel.binding_sha256,
                "accepted_x4_provenance": accepted_provenance,
                "output_decoder": decoder_audit,
                "raw_model_sha256": str(
                    model_metadata["source_model_sha256"]
                ),
                "factorized_model_sha256": factorized_model_sha256,
                "factorized_execution_sha256": (
                    factorized_execution_sha256
                ),
                "progressive_runtime_binding_sha256": (
                    runtime.runtime_binding_sha256
                ),
                "graph_candidate_file_sha256": (
                    frozen_input_file_sha256s["graph_candidate"]
                ),
                "basis_file_sha256": frozen_input_file_sha256s["basis"],
                "base_artifact_file_sha256": (
                    frozen_input_file_sha256s["base_artifact"]
                ),
                "refit_artifact_file_sha256": (
                    frozen_input_file_sha256s["refit_artifact"]
                ),
                "source_code_sha256s": code_before,
            },
        }
        return publish_gemma_h4_damping_materialization(
            replace(
                materialization,
                report_payload=enriched_payload,
            ),
            output=destination,
        )
    finally:
        switcher.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the hash-locked alpha=0 baseline and alpha=0.5 H4 "
            "challenger; selection, guard, and calibration-B inputs are absent."
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
        "--damping-report",
        type=Path,
        default=_DEFAULT_DAMPING_REPORT,
    )
    parser.add_argument("--damping-report-sha256", required=True)
    parser.add_argument("--damping-report-file-sha256", required=True)
    parser.add_argument(
        "--accepted-x4-report",
        type=Path,
        default=_DEFAULT_ACCEPTED_REPORT,
    )
    parser.add_argument(
        "--accepted-x4-candidate",
        type=Path,
        default=_DEFAULT_ACCEPTED_CANDIDATE,
    )
    parser.add_argument(
        "--accepted-x4-candidate-sha256",
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
    report = run_gemma_h4_damping_materialization(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        damping_report_path=args.damping_report,
        expected_damping_report_sha256=args.damping_report_sha256,
        expected_damping_report_file_sha256=(
            args.damping_report_file_sha256
        ),
        accepted_x4_report_path=args.accepted_x4_report,
        accepted_x4_candidate_path=args.accepted_x4_candidate,
        expected_accepted_x4_candidate_file_sha256=(
            args.accepted_x4_candidate_sha256
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
