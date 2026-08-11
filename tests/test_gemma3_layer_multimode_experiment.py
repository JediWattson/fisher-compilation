from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fisher_graph import gemma3_layer_multimode_experiment as experiment


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _production_preflight(tmp_path: Path):
    info = tmp_path.stat()
    resource = experiment.LayerMultimodeResourcePreflightReceipt(
        workspace_parent_path=str(tmp_path),
        workspace_path=str(
            tmp_path / "layer-multimode-experiment-v2.work"
        ),
        report_path=str(tmp_path / "report.json"),
        bundle_path=str(tmp_path / "candidate.bundle"),
        summary_path=str(tmp_path / "summary.json"),
        workspace_parent_device=info.st_dev,
        workspace_parent_inode=info.st_ino,
        output_parent_identities=(
            (info.st_dev, info.st_ino),
            (info.st_dev, info.st_ino),
            (info.st_dev, info.st_ino),
        ),
        observed_free_bytes_by_device=(
            (info.st_dev, experiment._MINIMUM_FREE_DISK_BYTES),
        ),
        physical_memory_bytes=experiment._MINIMUM_PHYSICAL_MEMORY_BYTES,
        A_recapture_payload_estimate_bytes=(
            experiment._A_RECAPTURE_PAYLOAD_ESTIMATE_BYTES
        ),
        A_materialization_payload_estimate_bytes=(
            experiment._A_MATERIALIZATION_PAYLOAD_ESTIMATE_BYTES
        ),
        full_grid_fit_peak_estimate_bytes=(
            experiment._FULL_GRID_FIT_PEAK_ESTIMATE_BYTES
        ),
        model_and_atomic_staging_estimate_bytes=(
            experiment._MODEL_AND_ATOMIC_STAGING_ESTIMATE_BYTES
        ),
        required_free_disk_bytes=experiment._MINIMUM_FREE_DISK_BYTES,
        required_physical_memory_bytes=(
            experiment._MINIMUM_PHYSICAL_MEMORY_BYTES
        ),
        workspace_path_safe=True,
        output_paths_safe=True,
        test_only_nonproduction=False,
    )
    seed = _sha("production-preflight")
    receipt = experiment.LayerMultimodeExecutionPreflightReceipt(
        setup_sha256=seed,
        setup_file_sha256=seed,
        panel_manifest_sha256=seed,
        panel_manifest_file_sha256=seed,
        panel_receipt_sha256=seed,
        panel_receipt_file_sha256=seed,
        panel_bundle_sha256=seed,
        implementation_qualification_sha256=seed,
        implementation_qualification_file_sha256=seed,
        predecessor_v2_setup_sha256=seed,
        predecessor_v2_report_sha256=seed,
        predecessor_A_capture_receipt_sha256=seed,
        predecessor_runtime_execution_receipt_sha256=seed,
        source_model_sha256=seed,
        evaluation_protocol_sha256=seed,
        resource_preflight_receipt=resource,
        test_only_nonproduction=False,
        production_runtime_ready=True,
    )
    object.__setattr__(
        receipt,
        "_construction_authority",
        experiment._PRODUCTION_AUTHORITY,
    )
    experiment._LIVE_PREFLIGHTS[id(receipt)] = receipt
    return receipt


@dataclass(frozen=True)
class _Artifact:
    label: str
    passed: bool = True

    @property
    def artifact_sha256(self) -> str:
        return _sha(self.label)

    def state_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "passed": self.passed,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class _RuntimeQualification:
    receipt_sha256: str

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "test.runtime.qualification.v1",
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class _AComparisonReceipt:
    state: dict[str, object]

    @classmethod
    def build(
        cls,
        preflight,
        A_runtime: experiment.LayerMultimodePredecessorARuntimeReceipt,
    ):
        payload = {
            "schema": experiment.PREDECESSOR_A_COMPARISON_SCHEMA,
            "format_version": 1,
            "predecessor_capture_receipt_sha256": (
                preflight.predecessor_A_capture_receipt_sha256
            ),
            "recaptured_capture_receipt_sha256": (
                A_runtime.source_capture_receipt_sha256
            ),
            "predecessor_runtime_execution_receipt_sha256": (
                preflight.predecessor_runtime_execution_receipt_sha256
            ),
            "predecessor_runtime_session_binding_sha256": _sha(
                "predecessor-A-session"
            ),
            "recaptured_runtime_session_binding_sha256": (
                A_runtime.runtime_session_binding_sha256
            ),
            "role": "A_chart_transport_fit",
            "batch_count": 45,
            "row_count": 24_732,
            "role_panel_binding_sha256": _sha("A-role-panel"),
            "ordered_objective_row_catalog_sha256": _sha("A-objective"),
            "score_objective_receipt_sha256": _sha("A-score"),
            "dense_lexical_parity_receipt_sha256": _sha("A-parity"),
            "mode_raw_artifact_catalog_sha256": _sha("A-mode-raw"),
            "mode_semantic_projection_catalog_sha256": _sha(
                "A-mode-semantic"
            ),
            "lexical_nll_receipt_catalog_sha256": _sha("A-lexical"),
            "supplemental_route_projection_catalog_sha256": _sha("A-routes"),
            "compared_field_inventory": [
                "batch_count",
                "dense_lexical_parity_receipt_sha256",
                "lexical_nll_receipt_sha256s",
                "mode_raw_artifacts",
                "mode_semantic_projection",
                "ordered_objective_row_catalog_sha256",
                "role",
                "role_panel_binding_sha256",
                "row_count",
                "score_objective_receipt_sha256",
                "supplemental_route_projection",
            ],
            "intentionally_session_dependent_field_inventory": [
                "batch_axis_catalog_sha256",
                "example_capture_receipt_sha256s",
                "execution_receipt_sha256s",
                "receipt_sha256",
                "runtime_session_binding_sha256",
                "supplemental_route_receipt_sha256s",
            ],
            "runtime_sessions_expected_different": True,
            "all_comparable_fields_matched": True,
        }
        return cls(
            {
                **payload,
                "artifact_sha256": experiment._digest(
                    experiment._PREDECESSOR_A_COMPARISON_DOMAIN,
                    payload,
                ),
            }
        )

    @property
    def artifact_sha256(self) -> str:
        return str(self.state["artifact_sha256"])

    def state_dict(self) -> dict[str, object]:
        return dict(self.state)


@dataclass(frozen=True)
class _CandidateEvaluation:
    layer_ordinal: int
    chart_count: int
    dynamic_mode_count: int
    tangent_rank: int
    seed: str

    @property
    def artifact_sha256(self) -> str:
        return _sha(
            f"candidate:{self.layer_ordinal}:{self.chart_count}:"
            f"{self.dynamic_mode_count}:{self.tangent_rank}"
        )

    @property
    def source_model_sha256(self) -> str:
        return self.seed

    @property
    def fit_policy_sha256(self) -> str:
        return self.seed

    @property
    def fit_a_sha256(self) -> str:
        return _sha("fit-A")

    @property
    def validation_a_sha256(self) -> str:
        return _sha("validation-A")

    @property
    def evaluation_protocol_sha256(self) -> str:
        return self.seed

    def state_dict(self) -> dict[str, object]:
        return {
            "layer_ordinal": self.layer_ordinal,
            "chart_count": self.chart_count,
            "dynamic_mode_count": self.dynamic_mode_count,
            "tangent_rank": self.tangent_rank,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class _CandidateSupportRejection:
    layer_ordinal: int
    chart_count: int
    dynamic_mode_count: int
    tangent_rank: int
    seed: str

    @property
    def artifact_sha256(self) -> str:
        return _sha(
            f"support-rejection:{self.layer_ordinal}:{self.chart_count}:"
            f"{self.dynamic_mode_count}:{self.tangent_rank}"
        )

    @property
    def source_model_sha256(self) -> str:
        return self.seed

    @property
    def fit_policy_sha256(self) -> str:
        return self.seed

    @property
    def fit_a_sha256(self) -> str:
        return _sha("fit-A")

    @property
    def validation_a_sha256(self) -> str:
        return _sha("validation-A")

    @property
    def evaluation_protocol_sha256(self) -> str:
        return self.seed

    def state_dict(self) -> dict[str, object]:
        return {
            "layer_ordinal": self.layer_ordinal,
            "chart_count": self.chart_count,
            "dynamic_mode_count": self.dynamic_mode_count,
            "tangent_rank": self.tangent_rank,
            "artifact_sha256": self.artifact_sha256,
            "support_rejected": True,
        }


class _ExpectedSupportError(RuntimeError):
    pass


@dataclass(frozen=True)
class _SelectedLayer:
    selected_candidate_sha256: str


@dataclass(frozen=True)
class _Selection:
    layers: tuple[_SelectedLayer, ...]

    @property
    def artifact_sha256(self) -> str:
        return _sha("selection")

    def state_dict(self) -> dict[str, object]:
        return {
            "layers": [
                {"selected_candidate_sha256": value.selected_candidate_sha256}
                for value in self.layers
            ],
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class _BundleReceipt:
    bundle_sha256: str
    byte_count: int
    manifest_sha256: str
    compilation_receipt_sha256: str


class _Authority:
    def __init__(self, receipt, events, *, failure: BaseException | None = None):
        self.receipt = receipt
        self.events = events
        self.failure = failure

    def strict_preflight(self):
        self.events.append("strict_preflight")
        if self.failure is not None:
            raise self.failure
        return self.receipt

    def revalidate(self, receipt):
        assert receipt is self.receipt
        self.events.append("revalidate")


class _Panels:
    def __init__(self, events):
        self.events = events
        self._opened_roles = []
        self._next = 0
        self._terminal = False
        self._current = None
        self._freeze_receipts = {}
        self._terminal_state = "ready_B3"

    @property
    def current_role(self):
        return self._current

    @property
    def opened_roles(self):
        return tuple(self._opened_roles)

    @property
    def required_freeze_stage_kind(self):
        if self._current is None:
            return None
        return experiment.FRESH_STAGE_KIND_BY_ROLE[self._current]

    @property
    def reserve_open_count(self):
        return 0

    @property
    def freeze_receipts(self):
        return dict(self._freeze_receipts)

    @property
    def terminal_state(self):
        return self._terminal_state

    @property
    def panel_manifest_sha256(self):
        return None

    def open_role(self, role):
        if self._terminal:
            raise RuntimeError("panel source is terminal")
        expected = (experiment.B3_ROLE, experiment.C3_ROLE, experiment.D3_ROLE)
        if role != expected[self._next]:
            raise RuntimeError("role order drifted")
        self._opened_roles.append(role)
        self._current = role
        self.events.append(f"open:{role}")
        return _Artifact(f"panel:{role}")

    def freeze_current(self, stage_kind, receipt_sha256, *, terminal_passed):
        assert stage_kind == experiment.FRESH_STAGE_KIND_BY_ROLE[self._current]
        assert receipt_sha256 == _sha(f"evaluation:{self._current}")
        self.events.append(f"freeze:{stage_kind}:{terminal_passed}")
        role = self._current
        self._freeze_receipts[role] = {
            "stage_kind": stage_kind,
            "stage_receipt_sha256": receipt_sha256,
            "terminal_passed": terminal_passed,
        }
        self._next += 1
        self._current = None
        if not terminal_passed:
            self._terminal = True
            self._terminal_state = f"terminal_{role.split('_', 1)[0]}_failed"
        elif role == experiment.D3_ROLE:
            self._terminal = True
            self._terminal_state = "terminal_D3_passed"
        else:
            next_role = (experiment.B3_ROLE, experiment.C3_ROLE, experiment.D3_ROLE)[
                self._next
            ]
            self._terminal_state = f"ready_{next_role.split('_', 1)[0]}"


class _Runtime(experiment.LayerMultimodeV3Runtime):
    test_only_nonproduction = True

    def __init__(
        self,
        preflight,
        events,
        *,
        B3=True,
        C3=True,
        D3=True,
        no_selection=False,
        atlas_support_failures=(),
        split_support_failures=(),
    ):
        self.preflight = preflight
        self.events = events
        self.B3 = B3
        self.C3 = C3
        self.D3 = D3
        self.no_selection = no_selection
        self.atlas_support_failures = frozenset(atlas_support_failures)
        self.split_support_failures = frozenset(split_support_failures)
        self.atlas_calls = []
        self.fit_calls = []
        self.split_calls = []
        self.lower_calls = []
        self.support_lower_calls = []
        self.release_compiled = "unset"
        self.closed = False
        self.fresh_role_captures = []
        source_execution = _sha("source-execution")
        self.A_runtime_receipt = (
            experiment.LayerMultimodePredecessorARuntimeReceipt(
                execution_preflight_receipt_sha256=preflight.receipt_sha256,
                predecessor_preflight_receipt_sha256=_sha(
                    "predecessor-preflight"
                ),
                predecessor_runtime_qualification_receipt_sha256=_sha(
                    "predecessor-runtime-qualification"
                ),
                source_capture_receipt_sha256=(
                    preflight.predecessor_A_capture_receipt_sha256
                ),
                runtime_session_binding_sha256=_sha("A-current-session"),
                source_model_fingerprint_before_sha256=(
                    preflight.source_model_sha256
                ),
                source_model_fingerprint_after_sha256=(
                    preflight.source_model_sha256
                ),
                source_execution_fingerprint_before_sha256=source_execution,
                source_execution_fingerprint_after_sha256=source_execution,
                role_batch_counts=(
                    ("A_chart_transport_fit", 45),
                    ("B_neighbor_discovery", 0),
                    ("C_neighbor_confirmation", 0),
                ),
                exact_control_forward_count=45,
                exact_control_vjp_count=45,
                dense_parity_smoke_count=1,
                dense_parity_forward_count=1,
                dense_parity_vjp_count=1,
                total_backbone_forward_count=46,
                total_terminal_cotangent_vjp_count=46,
                captured_example_count=48,
                supplemental_route_receipt_count=864,
                mode_raw_artifact_count=38,
                lexical_nll_receipt_count=45,
                execution_receipt_count=45,
                example_capture_receipt_count=48,
                canonical_gradient_targets_per_vjp=73,
                auxiliary_gradient_targets_per_vjp=19,
                gradient_targets_per_vjp=92,
            )
        )
        self.A_comparison_receipt = _AComparisonReceipt.build(
            preflight,
            self.A_runtime_receipt,
        )

    def recapture_exposed_predecessor_A(self, *, preflight, workspace):
        assert preflight is self.preflight
        self.events.append("recapture:A")
        return experiment.ADevelopmentRecapture(
            bridge=object(),
            partition_receipt=_Artifact("partition"),
            source_model=object(),
            source_model_sha256=preflight.source_model_sha256,
            source_capture_receipt_sha256=(
                preflight.predecessor_A_capture_receipt_sha256
            ),
            predecessor_capture_receipt_sha256=(
                preflight.predecessor_A_capture_receipt_sha256
            ),
            predecessor_lineage_comparison_receipt=(
                self.A_comparison_receipt
            ),
            predecessor_A_runtime_receipt=self.A_runtime_receipt,
        )

    def materialize_A_layer(self, recapture, layer_ordinal):
        self.events.append(f"materialize:{layer_ordinal}")
        return experiment.LayerMaterialization(
            layer_ordinal=layer_ordinal,
            rows=_Artifact(f"rows:{layer_ordinal}"),
            receipt=_Artifact(f"materialization:{layer_ordinal}"),
        )

    def build_materialization_inventory(self, recapture, layers):
        assert tuple(value.layer_ordinal for value in layers) == tuple(range(18))
        return _Artifact("materialization-inventory")

    def fit_chart_atlas(self, materialization, *, chart_count, tangent_rank):
        key = (materialization.layer_ordinal, chart_count, tangent_rank)
        self.atlas_calls.append(key)
        if key in self.atlas_support_failures:
            return experiment.LayerMultimodePreoptimizerAttempt(
                result=None,
                support_error=_ExpectedSupportError(f"atlas:{key}"),
            )
        return experiment.LayerMultimodePreoptimizerAttempt(
            result=_Artifact(f"atlas:{key}"),
            support_error=None,
        )

    def fit_candidate(
        self,
        materialization,
        *,
        chart_count,
        mode_count,
        tangent_rank,
        atlas,
    ):
        key = (
            materialization.layer_ordinal,
            chart_count,
            mode_count,
            tangent_rank,
        )
        assert atlas.label == (
            f"atlas:{(materialization.layer_ordinal, chart_count, tangent_rank)}"
        )
        self.fit_calls.append(key)
        return _Artifact(f"fit:{key}")

    def evaluate_split_refit(
        self,
        materialization,
        *,
        chart_count,
        mode_count,
        tangent_rank,
    ):
        assert mode_count > 1
        key = (
            materialization.layer_ordinal,
            chart_count,
            mode_count,
            tangent_rank,
        )
        self.split_calls.append(key)
        if key in self.split_support_failures:
            return experiment.LayerMultimodePreoptimizerAttempt(
                result=None,
                support_error=_ExpectedSupportError(f"split:{key}"),
            )
        return experiment.LayerMultimodePreoptimizerAttempt(
            result=_Artifact(f"split:{key}"),
            support_error=None,
        )

    def lower_candidate_support_rejection(
        self,
        *,
        materialization,
        chart_count,
        mode_count,
        tangent_rank,
        candidate_fit,
        support_error,
        recapture,
    ):
        assert type(support_error) is _ExpectedSupportError
        key = (
            materialization.layer_ordinal,
            chart_count,
            mode_count,
            tangent_rank,
        )
        if key in self.split_support_failures:
            assert candidate_fit is not None
        else:
            assert candidate_fit is None
        self.support_lower_calls.append(key)
        return _CandidateSupportRejection(
            *key,
            seed=self.preflight.source_model_sha256,
        )

    def lower_candidate_evaluation(self, *, candidate_fit, split_refit, recapture):
        key = self.fit_calls[-1]
        assert candidate_fit.label == f"fit:{key}"
        assert (split_refit is not None) is (key[2] > 1)
        self.lower_calls.append(key)
        return _CandidateEvaluation(*key, seed=self.preflight.source_model_sha256)

    def select_full_grid(self, evaluations):
        self.events.append("select:324")
        if self.no_selection:
            from fisher_graph.gemma3_layer_multimode_selection import (
                NoAdmissibleLayerMultimodeCandidateError,
            )

            raise NoAdmissibleLayerMultimodeCandidateError(0, ())
        selected = []
        for layer in range(18):
            candidate = next(
                value
                for value in evaluations
                if value.layer_ordinal == layer
                and type(value) is _CandidateEvaluation
            )
            selected.append(_SelectedLayer(candidate.artifact_sha256))
        return _Selection(tuple(selected))

    def build_selected_inventory(
        self,
        *,
        recapture,
        materialization_inventory,
        selection,
        selected_fits,
        selected_splits,
    ):
        assert len(selected_fits) == len(selected_splits) == 18
        self.events.append("inventory:selected")
        return _Artifact("selected-inventory")

    def compile_source_free(self, *, source_model, selected_inventory):
        self.events.append("compile:source-free")
        return _Artifact("compiled")

    def freeze_compiled_candidate(self, *, compiled, selection, workspace):
        self.events.append("freeze:bundle")
        payload = b"source-free-layer-multimode-bundle"
        path = workspace / "candidate.bundle"
        path.write_bytes(payload)
        bundle_sha = hashlib.sha256(payload).hexdigest()
        compilation = _Artifact("compilation")
        return experiment.LayerMultimodeFrozenCandidate(
            compiled_model=_Artifact("reloaded-source-free-model"),
            candidate_identity=_Artifact("candidate-identity"),
            bundle_receipt=_BundleReceipt(
                bundle_sha256=bundle_sha,
                byte_count=len(payload),
                manifest_sha256=_sha("manifest"),
                compilation_receipt_sha256=compilation.artifact_sha256,
            ),
            compilation_receipt=compilation,
            bundle_path=str(path),
            bundle_file_sha256=bundle_sha,
        )

    def release_development_state(
        self,
        *,
        recapture,
        materializations,
        atlases,
        candidate_fits,
        split_refits,
        compiled,
    ):
        self.release_compiled = None if compiled is None else "present"
        prepared = {
            "compiled": compiled is not None,
            "compilation": None if compiled is None else compiled.artifact_sha256,
        }
        self._pending_release = prepared
        self.events.append("prepare-teardown:A-and-source")
        return prepared

    def confirm_development_teardown(self, prepared_release):
        assert prepared_release is self._pending_release
        self._pending_release = None
        self.events.append("teardown:A-and-source")
        return experiment.LayerMultimodeDevelopmentTeardownReceipt(
            source_model_sha256=self.preflight.source_model_sha256,
            source_execution_fingerprint_sha256=_sha("source-execution"),
            compiled_candidate_present=prepared_release["compiled"],
            compilation_receipt_sha256=(
                prepared_release["compilation"]
            ),
            compiled_model_fingerprint_sha256=(
                _sha("compiled-model") if prepared_release["compiled"] else None
            ),
            source_compilation_aliases_released=True,
            native_baseline_retained_for_heldout_evaluation=True,
            A_activation_rows_released=True,
            A_fit_models_released=True,
            A_optimizer_state_released=True,
            source_compilation_native_mlp_alias_count=0,
        )

    def _evaluation(self, role, passed):
        self.events.append(f"evaluate:{role}")
        return _Artifact(f"evaluation:{role}", passed=passed)

    def authenticate_fresh_panel_tokenization(
        self,
        *,
        panel_source,
        panel,
        role,
        selection_receipt_sha256,
        frozen_candidate_identity_sha256,
    ):
        assert panel_source.current_role == role
        self.events.append(f"authenticate-tokenization:{role}")
        receipt = experiment.LayerMultimodeFreshRoleCaptureReceipt(
            role=role,
            stage_kind=experiment.FRESH_STAGE_KIND_BY_ROLE[role],
            panel_manifest_sha256=self.preflight.panel_manifest_sha256,
            panel_manifest_file_sha256=(
                self.preflight.panel_manifest_file_sha256
            ),
            panel_receipt_sha256=self.preflight.panel_receipt_sha256,
            panel_receipt_file_sha256=self.preflight.panel_receipt_file_sha256,
            panel_bundle_sha256=self.preflight.panel_bundle_sha256,
            role_file_sha256=_sha(f"role-file:{role}"),
            selection_receipt_sha256=selection_receipt_sha256,
            frozen_candidate_identity_sha256=(
                frozen_candidate_identity_sha256
            ),
            runtime_session_binding_sha256=self._runtime_session(),
            ordered_tokenization_receipt_sha256=_sha(f"tokenization:{role}"),
            ordered_example_identity_catalog_sha256=_sha(f"examples:{role}"),
            ordered_row_identity_catalog_sha256=_sha(f"rows:{role}"),
            family_axis_sha256=_sha("family-axis"),
            batch_axis_sha256=_sha("batch-axis"),
            example_count=48,
            batch_count=8,
            predicted_next_token_position_count=24_732,
        )
        self.fresh_role_captures.append(receipt)
        return receipt

    def evaluate_b3_teacher_forced(
        self,
        *,
        panel,
        candidate,
        fresh_role_capture_receipt,
    ):
        assert fresh_role_capture_receipt is self.fresh_role_captures[-1]
        return self._evaluation(experiment.B3_ROLE, self.B3)

    def evaluate_c3_full_stack(
        self,
        *,
        panel,
        candidate,
        b3_evaluation,
        fresh_role_capture_receipt,
    ):
        assert b3_evaluation.passed
        assert fresh_role_capture_receipt is self.fresh_role_captures[-1]
        return self._evaluation(experiment.C3_ROLE, self.C3)

    def evaluate_d3_final(
        self,
        *,
        panel,
        candidate,
        c3_evaluation,
        fresh_role_capture_receipt,
    ):
        assert c3_evaluation.passed
        assert fresh_role_capture_receipt is self.fresh_role_captures[-1]
        return self._evaluation(experiment.D3_ROLE, self.D3)

    def _runtime_session(self):
        seed = self.preflight.source_model_sha256
        compiled = _sha("compiled-model")
        source_execution = _sha("source-execution")
        compiled_execution = _sha("compiled-execution")
        return experiment.layer_multimode_runtime_session_binding_sha256(
            runtime_implementation_sha256=_sha("runtime-implementation"),
            implementation_qualification_sha256=(
                self.preflight.implementation_qualification_sha256
            ),
            execution_environment_sha256=_sha("environment"),
            tokenizer_fingerprint_sha256=_sha("tokenizer"),
            runtime_qualification_receipt_sha256=_sha(
                "runtime-qualification"
            ),
            source_model_fingerprint_before_sha256=seed,
            source_model_fingerprint_after_sha256=seed,
            source_execution_fingerprint_before_sha256=source_execution,
            source_execution_fingerprint_after_sha256=source_execution,
            compiled_model_fingerprint_before_sha256=compiled,
            compiled_model_fingerprint_after_sha256=compiled,
            compiled_execution_fingerprint_before_sha256=compiled_execution,
            compiled_execution_fingerprint_after_sha256=compiled_execution,
            compiled_candidate_mutated=False,
        )

    def build_runtime_execution_receipt(self, *, terminal_status):
        seed = self.preflight.source_model_sha256
        compiled = _sha("compiled-model")
        source_execution = _sha("source-execution")
        compiled_execution = _sha("compiled-execution")
        environment = _sha("environment")
        tokenizer = _sha("tokenizer")
        implementation = _sha("runtime-implementation")
        runtime_qualification = _RuntimeQualification(
            _sha("runtime-qualification")
        )
        session = self._runtime_session()
        opened = {
            experiment.TERMINAL_B3_FAILED: 1,
            experiment.TERMINAL_C3_FAILED: 2,
            experiment.TERMINAL_D3_COMPLETED: 3,
        }[terminal_status]
        return experiment.LayerMultimodeRuntimeExecutionReceipt(
            terminal_status=terminal_status,
            runtime_implementation_sha256=implementation,
            implementation_qualification_sha256=(
                self.preflight.implementation_qualification_sha256
            ),
            runtime_qualification_receipt=runtime_qualification,
            runtime_qualification_receipt_sha256=(
                runtime_qualification.receipt_sha256
            ),
            execution_environment_sha256=environment,
            tokenizer_fingerprint_sha256=tokenizer,
            predecessor_A_capture_receipt_sha256=(
                self.preflight.predecessor_A_capture_receipt_sha256
            ),
            predecessor_runtime_execution_receipt_sha256=(
                self.preflight.predecessor_runtime_execution_receipt_sha256
            ),
            predecessor_A_runtime_receipt=self.A_runtime_receipt,
            predecessor_A_runtime_receipt_sha256=(
                self.A_runtime_receipt.receipt_sha256
            ),
            predecessor_A_lineage_comparison_receipt=(
                self.A_comparison_receipt
            ),
            predecessor_A_lineage_comparison_receipt_sha256=(
                self.A_comparison_receipt.artifact_sha256
            ),
            predecessor_A_recaptured_capture_receipt_sha256=(
                self.A_runtime_receipt.source_capture_receipt_sha256
            ),
            predecessor_A_current_runtime_session_binding_sha256=(
                self.A_runtime_receipt.runtime_session_binding_sha256
            ),
            source_model_fingerprint_before_sha256=seed,
            source_model_fingerprint_after_sha256=seed,
            source_model_mutated=False,
            source_execution_fingerprint_before_sha256=source_execution,
            source_execution_fingerprint_after_sha256=source_execution,
            source_execution_mutated=False,
            compiled_model_fingerprint_before_sha256=compiled,
            compiled_model_fingerprint_after_sha256=compiled,
            compiled_execution_fingerprint_before_sha256=compiled_execution,
            compiled_execution_fingerprint_after_sha256=compiled_execution,
            compiled_candidate_mutated=False,
            runtime_session_binding_sha256=session,
            fresh_role_capture_receipts=tuple(self.fresh_role_captures),
            predecessor_A_batch_count=45,
            predecessor_A_exact_control_forward_count=45,
            predecessor_A_exact_control_vjp_count=45,
            predecessor_A_dense_parity_forward_count=1,
            predecessor_A_dense_parity_vjp_count=1,
            B3_batch_count=8,
            B3_native_forward_count=8,
            B3_candidate_layer_velocity_forward_count=144,
            C3_batch_count=8 if opened >= 2 else 0,
            C3_native_forward_count=8 if opened >= 2 else 0,
            C3_compiled_forward_count=8 if opened >= 2 else 0,
            D3_batch_count=8 if opened >= 3 else 0,
            D3_native_forward_count=8 if opened >= 3 else 0,
            D3_compiled_forward_count=8 if opened >= 3 else 0,
            fresh_tokenization_authentication_count=opened,
            fresh_example_count=48 * opened,
            fresh_next_token_position_count=24_732 * opened,
            native_baseline_evaluator_count=1,
            native_baseline_eval_mode=True,
            native_baseline_parameter_mutation_count=0,
            native_baseline_optimizer_reference_count=0,
            compiled_candidate_native_mlp_module_count=0,
            reserve_open_count=0,
        )

    def close(
        self,
        *,
        terminal_status,
        runtime_execution_receipt_sha256,
        runtime_session_binding_sha256,
        fresh_panel_execution_receipt_sha256,
    ):
        self.events.append("runtime:close")
        self.closed = True
        return experiment.LayerMultimodePostTerminalCleanupReceipt(
            terminal_status=terminal_status,
            runtime_execution_receipt_sha256=(
                runtime_execution_receipt_sha256
            ),
            runtime_session_binding_sha256=runtime_session_binding_sha256,
            fresh_panel_execution_receipt_sha256=(
                fresh_panel_execution_receipt_sha256
            ),
            runtime_closed=True,
            native_baseline_released=True,
            native_baseline_instance_count_after_close=0,
            compiled_candidate_released=True,
            model_reference_count_after_close=0,
            tokenizer_reference_count_after_close=0,
            temporary_A_spools_removed=True,
            temporary_materializations_removed=True,
            temporary_fit_state_removed=True,
            frozen_bundle_retained_for_publication=(
                terminal_status != experiment.TERMINAL_A_NO_CANDIDATE
            ),
            reserve_open_count=0,
        )

    def abort(self):
        self.events.append("runtime:abort")
        self.closed = True


@dataclass(frozen=True)
class _Publication:
    report_path: Path
    bundle_path: Path
    summary_path: Path


class _Publisher:
    def __init__(self, events, *, fail_finalize=False):
        self.events = events
        self.fail_finalize = fail_finalize
        self.publish_calls = 0

    def seal_layer_multimode_experiment_report(self, payload):
        from fisher_graph import gemma3_layer_multimode_report as report_module

        assert set(payload) == report_module._REPORT_PAYLOAD_FIELDS
        assert payload["reserve_disclosure"] == {
            "role": experiment.RESERVE_ROLE,
            "semantically_opened": False,
            "capture_receipt_sha256": None,
            "evaluation_receipt_sha256": None,
        }
        self.events.append("report:seal")
        return dict(payload)

    def publish_layer_multimode_experiment_outputs(
        self,
        *,
        report,
        bundle_source_path,
        report_path,
        bundle_path,
        summary_path,
    ):
        self.publish_calls += 1
        self.events.append("report:publish")
        bundle_path.write_bytes(bundle_source_path.read_bytes())
        report_path.write_text(str(report["terminal_status"]), encoding="utf-8")
        summary_path.write_text("summary", encoding="utf-8")
        return _Publication(report_path, bundle_path, summary_path)

    def rollback_published_layer_multimode_experiment(self, receipt):
        self.events.append("report:rollback")
        for path in (
            receipt.report_path,
            receipt.bundle_path,
            receipt.summary_path,
        ):
            path.unlink(missing_ok=True)

    def finalize_published_layer_multimode_experiment(self, receipt):
        self.events.append("report:finalize")
        if self.fail_finalize:
            raise RuntimeError("finalize failed")


def _run(
    tmp_path,
    *,
    B3=True,
    C3=True,
    D3=True,
    no_selection=False,
    atlas_support_failures=(),
    split_support_failures=(),
    publisher=None,
):
    events = []
    preflight = experiment.make_layer_multimode_test_preflight()
    authority = _Authority(preflight, events)
    panels = _Panels(events)
    runtimes = []

    def runtime_factory(receipt):
        events.append("runtime:factory")
        runtime = _Runtime(
            receipt,
            events,
            B3=B3,
            C3=C3,
            D3=D3,
            no_selection=no_selection,
            atlas_support_failures=atlas_support_failures,
            split_support_failures=split_support_failures,
        )
        runtimes.append(runtime)
        return runtime

    actual_publisher = publisher or _Publisher(events)
    result = experiment.run_layer_multimode_experiment(
        authority=authority,
        runtime_factory=runtime_factory,
        panel_source_factory=lambda receipt, selection_sha: panels,
        publisher=actual_publisher,
        workspace_parent=tmp_path,
        report_path=tmp_path / "report.json",
        bundle_path=tmp_path / "candidate.bundle",
        summary_path=tmp_path / "summary.json",
    )
    return result, runtimes[0], panels, actual_publisher, events


def test_complete_grid_caches_atlas_and_runs_b3_c3_d3(tmp_path):
    result, runtime, panels, publisher, events = _run(tmp_path)

    assert isinstance(result.outcome, experiment.D3CompletedOutcome)
    counters = result.outcome.counters
    assert counters.materialization_count == 18
    assert counters.atlas_fit_attempt_count == 108
    assert counters.atlas_fit_success_count == 108
    assert counters.atlas_support_rejection_count == 0
    assert counters.candidate_fit_count == 324
    assert counters.split_refit_attempt_count == 216
    assert counters.split_refit_count == 216
    assert counters.split_refit_support_rejection_count == 0
    assert counters.candidate_evaluation_receipt_count == 324
    assert counters.candidate_support_rejection_count == 0
    assert counters.candidate_inventory_count == 324
    assert len(runtime.atlas_calls) == len(set(runtime.atlas_calls)) == 108
    assert len(runtime.fit_calls) == len(set(runtime.fit_calls)) == 324
    assert panels.opened_roles == (
        experiment.B3_ROLE,
        experiment.C3_ROLE,
        experiment.D3_ROLE,
    )
    assert experiment.RESERVE_ROLE not in panels.opened_roles
    assert events.index("strict_preflight") < events.index("runtime:factory")
    assert events.index("teardown:A-and-source") < events.index(
        f"open:{experiment.B3_ROLE}"
    )
    for role in (experiment.B3_ROLE, experiment.C3_ROLE, experiment.D3_ROLE):
        assert events[events.index(f"open:{role}") - 1] == "revalidate"
    assert events[events.index("report:seal") - 1] == "revalidate"
    assert tuple(value.role for value in runtime.fresh_role_captures) == (
        experiment.B3_ROLE,
        experiment.C3_ROLE,
        experiment.D3_ROLE,
    )
    assert runtime.closed
    assert publisher.publish_calls == 1
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "candidate.bundle").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert not (tmp_path / "layer-multimode-experiment-v2.work").exists()


def test_shared_atlas_support_rejection_fans_out_without_fake_fits(tmp_path):
    failed_atlas = (0, 2, 64)
    result, runtime, panels, publisher, _ = _run(
        tmp_path,
        atlas_support_failures=(failed_atlas,),
    )

    assert isinstance(result.outcome, experiment.D3CompletedOutcome)
    counters = result.outcome.counters
    assert counters.atlas_fit_attempt_count == 108
    assert counters.atlas_fit_success_count == 107
    assert counters.atlas_support_rejection_count == 1
    assert counters.candidate_fit_count == 321
    assert counters.split_refit_attempt_count == 214
    assert counters.split_refit_count == 214
    assert counters.split_refit_support_rejection_count == 0
    assert counters.candidate_evaluation_receipt_count == 321
    assert counters.candidate_support_rejection_count == 3
    assert counters.candidate_inventory_count == 324
    assert runtime.atlas_calls.count(failed_atlas) == 1
    rejected_keys = {
        (0, 2, mode_count, 64) for mode_count in (4, 2, 1)
    }
    assert set(runtime.support_lower_calls) == rejected_keys
    assert rejected_keys.isdisjoint(runtime.fit_calls)
    assert panels.opened_roles == (
        experiment.B3_ROLE,
        experiment.C3_ROLE,
        experiment.D3_ROLE,
    )
    assert publisher.publish_calls == 1


def test_split_support_rejection_rejects_only_its_candidate(tmp_path):
    failed_split = (0, 2, 4, 32)
    result, runtime, panels, publisher, _ = _run(
        tmp_path,
        split_support_failures=(failed_split,),
    )

    assert isinstance(result.outcome, experiment.D3CompletedOutcome)
    counters = result.outcome.counters
    assert counters.atlas_fit_attempt_count == 108
    assert counters.atlas_fit_success_count == 108
    assert counters.atlas_support_rejection_count == 0
    assert counters.candidate_fit_count == 324
    assert counters.split_refit_attempt_count == 216
    assert counters.split_refit_count == 215
    assert counters.split_refit_support_rejection_count == 1
    assert counters.candidate_evaluation_receipt_count == 323
    assert counters.candidate_support_rejection_count == 1
    assert counters.candidate_inventory_count == 324
    assert runtime.support_lower_calls == [failed_split]
    assert failed_split in runtime.fit_calls
    assert failed_split in runtime.split_calls
    assert panels.opened_roles == (
        experiment.B3_ROLE,
        experiment.C3_ROLE,
        experiment.D3_ROLE,
    )
    assert publisher.publish_calls == 1


def test_scalar_publication_handoff_retains_no_compiled_model(tmp_path):
    preflight = experiment.make_layer_multimode_test_preflight()
    runtime = _Runtime(preflight, [])
    frozen = runtime.freeze_compiled_candidate(
        compiled=_Artifact("compiled"),
        selection=_Selection(tuple()),
        workspace=tmp_path,
    )
    scalar = experiment._frozen_publication_evidence(frozen)
    assert not hasattr(scalar, "compiled_model")
    assert scalar.candidate_identity is frozen.candidate_identity
    assert scalar.bundle_receipt is frozen.bundle_receipt
    assert scalar.compilation_receipt is frozen.compilation_receipt


def test_b3_failure_is_terminal_and_durable(tmp_path):
    result, _, panels, publisher, _ = _run(tmp_path, B3=False)

    assert isinstance(result.outcome, experiment.B3FidelityFailureOutcome)
    assert panels.opened_roles == (experiment.B3_ROLE,)
    assert result.outcome.counters.C3_open_count == 0
    assert result.outcome.counters.D3_open_count == 0
    assert publisher.publish_calls == 1


def test_c3_failure_never_opens_d3(tmp_path):
    result, _, panels, publisher, _ = _run(tmp_path, C3=False)

    assert isinstance(result.outcome, experiment.C3FidelityFailureOutcome)
    assert panels.opened_roles == (experiment.B3_ROLE, experiment.C3_ROLE)
    assert result.outcome.counters.D3_open_count == 0
    assert publisher.publish_calls == 1


def test_A_no_candidate_returns_typed_outcome_with_zero_outputs(
    tmp_path,
    monkeypatch,
):
    candidate_hashes = tuple(_sha(f"candidate:{index}") for index in range(324))
    decision_hashes = tuple(_sha(f"decision:{index}") for index in range(324))
    terminal = experiment.LayerMultimodeNoAdmissibleSelectionReceipt(
        source_model_sha256=_sha("source"),
        selection_policy_sha256=_sha("selection-policy"),
        fit_policy_sha256=_sha("fit-policy"),
        fit_a_sha256=_sha("fit-A"),
        validation_a_sha256=_sha("validation-A"),
        evaluation_protocol_sha256=_sha("evaluation-policy"),
        candidate_sha256s=candidate_hashes,
        decision_sha256s=decision_hashes,
        failed_layer_ordinals=(3, 9),
        first_failed_layer_ordinal=3,
    )
    monkeypatch.setattr(
        experiment,
        "_build_no_selection_receipt",
        lambda evaluations, error: terminal,
    )

    result, runtime, panels, publisher, events = _run(
        tmp_path,
        no_selection=True,
    )

    assert isinstance(result.outcome, experiment.NoAdmissibleCandidateOutcome)
    assert result.outcome.no_selection_receipt is terminal
    assert result.publication is None
    assert publisher.publish_calls == 0
    assert panels.opened_roles == ()
    assert runtime.release_compiled is None
    assert runtime.closed
    assert "teardown:A-and-source" in events
    for name in ("report.json", "candidate.bundle", "summary.json"):
        assert not (tmp_path / name).exists()


def test_support_rejected_layer_returns_A_no_candidate_before_heldout(
    tmp_path,
    monkeypatch,
):
    candidate_hashes = tuple(_sha(f"support-grid:{index}") for index in range(324))
    decision_hashes = tuple(_sha(f"support-decision:{index}") for index in range(324))
    terminal = experiment.LayerMultimodeNoAdmissibleSelectionReceipt(
        source_model_sha256=_sha("source"),
        selection_policy_sha256=_sha("selection-policy"),
        fit_policy_sha256=_sha("fit-policy"),
        fit_a_sha256=_sha("fit-A"),
        validation_a_sha256=_sha("validation-A"),
        evaluation_protocol_sha256=_sha("evaluation-policy"),
        candidate_sha256s=candidate_hashes,
        decision_sha256s=decision_hashes,
        failed_layer_ordinals=(0,),
        first_failed_layer_ordinal=0,
    )
    monkeypatch.setattr(
        experiment,
        "_build_no_selection_receipt",
        lambda evaluations, error: terminal,
    )
    failed_atlases = tuple(
        (0, chart_count, tangent_rank)
        for chart_count in experiment.CHART_COUNT_CANDIDATES
        for tangent_rank in experiment.TANGENT_RANK_CANDIDATES
    )

    result, runtime, panels, publisher, _ = _run(
        tmp_path,
        no_selection=True,
        atlas_support_failures=failed_atlases,
    )

    assert isinstance(result.outcome, experiment.NoAdmissibleCandidateOutcome)
    counters = result.outcome.counters
    assert counters.atlas_fit_attempt_count == 108
    assert counters.atlas_fit_success_count == 102
    assert counters.atlas_support_rejection_count == 6
    assert counters.candidate_fit_count == 306
    assert counters.split_refit_attempt_count == 204
    assert counters.split_refit_count == 204
    assert counters.candidate_evaluation_receipt_count == 306
    assert counters.candidate_support_rejection_count == 18
    assert counters.candidate_inventory_count == 324
    assert len(runtime.support_lower_calls) == 18
    assert panels.opened_roles == ()
    assert publisher.publish_calls == 0
    assert result.publication is None
    for name in ("report.json", "candidate.bundle", "summary.json"):
        assert not (tmp_path / name).exists()


def test_preflight_failure_happens_before_runtime_or_workspace(tmp_path):
    events = []
    authority = _Authority(
        None,
        events,
        failure=RuntimeError("preflight rejected"),
    )

    with pytest.raises(RuntimeError, match="preflight rejected"):
        experiment.run_layer_multimode_experiment(
            authority=authority,
            runtime_factory=lambda receipt: pytest.fail("runtime constructed"),
            panel_source_factory=lambda receipt, selection: pytest.fail(
                "panel source constructed"
            ),
            publisher=_Publisher(events),
            workspace_parent=tmp_path,
            report_path=tmp_path / "report.json",
            bundle_path=tmp_path / "candidate.bundle",
            summary_path=tmp_path / "summary.json",
        )

    assert events == ["strict_preflight"]
    assert not (tmp_path / "layer-multimode-experiment-v2.work").exists()


def test_copied_preflight_loses_live_authority_before_runtime(tmp_path):
    receipt = experiment.make_layer_multimode_test_preflight()
    copied = experiment.LayerMultimodeExecutionPreflightReceipt(
        **{
            value.name: getattr(receipt, value.name)
            for value in fields(experiment.LayerMultimodeExecutionPreflightReceipt)
            if value.init
        }
    )
    events = []
    authority = _Authority(copied, events)

    with pytest.raises(RuntimeError, match="original live authority"):
        experiment.run_layer_multimode_experiment(
            authority=authority,
            runtime_factory=lambda value: pytest.fail("runtime constructed"),
            panel_source_factory=lambda value, selection: object(),
            publisher=_Publisher(events),
            workspace_parent=tmp_path,
            report_path=tmp_path / "report.json",
            bundle_path=tmp_path / "candidate.bundle",
            summary_path=tmp_path / "summary.json",
        )

    assert events == ["strict_preflight"]


def test_production_receipt_forbids_every_injected_execution_seam(tmp_path):
    receipt = _production_preflight(tmp_path)
    events = []
    authority = _Authority(receipt, events)

    with pytest.raises(RuntimeError, match="forbids injected"):
        experiment.run_layer_multimode_experiment(
            authority=authority,
            runtime_factory=lambda value: pytest.fail("runtime constructed"),
            panel_source_factory=lambda value, selection: pytest.fail(
                "panel source constructed"
            ),
            publisher=_Publisher(events),
            workspace_parent=tmp_path,
            report_path=tmp_path / "report.json",
            bundle_path=tmp_path / "candidate.bundle",
            summary_path=tmp_path / "summary.json",
        )

    assert events == ["strict_preflight", "revalidate"]
    assert not (tmp_path / "layer-multimode-experiment-v2.work").exists()


def test_production_authority_revalidates_rebuilt_state_not_object_identity(
    tmp_path,
    monkeypatch,
):
    original = _production_preflight(tmp_path)
    rebuilt = experiment.LayerMultimodeExecutionPreflightReceipt(
        **{
            value.name: getattr(original, value.name)
            for value in fields(
                experiment.LayerMultimodeExecutionPreflightReceipt
            )
            if value.init
        }
    )
    panel = SimpleNamespace(
        manifest={"manifest_sha256": _sha("manifest")},
        manifest_file_sha256=_sha("manifest-file"),
        receipt_file_sha256=_sha("receipt-file"),
        receipt_sha256=_sha("receipt"),
        bundle_sha256=_sha("bundle"),
        role_file_sha256_items=(("B3", _sha("B3")),),
        reserve_semantically_decoded=False,
    )
    observations = iter(
        (
            experiment._ProductionPreflightObservation(original, panel),
            experiment._ProductionPreflightObservation(rebuilt, panel),
        )
    )
    calls = []

    def observe(**kwargs):
        calls.append(kwargs)
        return next(observations)

    monkeypatch.setattr(
        experiment,
        "_strict_production_preflight_observation",
        observe,
    )
    authority = experiment.StaticLayerMultimodeExperimentAuthority(
        repository_root=tmp_path,
        setup_path=tmp_path / "setup.json",
    )

    issued = authority.strict_preflight()
    authority.revalidate(issued)

    assert issued is original
    assert calls[1]["resource_receipt"] is (
        original.resource_preflight_receipt
    )
    assert calls[1]["grant_live_authority"] is False


def test_finalize_failure_rolls_back_all_three_outputs(tmp_path):
    events = []
    publisher = _Publisher(events, fail_finalize=True)

    with pytest.raises(RuntimeError, match="finalize failed"):
        _run(tmp_path, publisher=publisher)

    assert "report:rollback" in events
    for name in ("report.json", "candidate.bundle", "summary.json"):
        assert not (tmp_path / name).exists()
    assert not (tmp_path / "layer-multimode-experiment-v2.work").exists()


def test_selection_engineering_error_propagates_without_terminal_rewrite(
    tmp_path,
    monkeypatch,
):
    def fail_selection(self, evaluations):
        raise ValueError("selector implementation fault")

    monkeypatch.setattr(_Runtime, "select_full_grid", fail_selection)
    with pytest.raises(ValueError, match="selector implementation fault"):
        _run(tmp_path)

    for name in ("report.json", "candidate.bundle", "summary.json"):
        assert not (tmp_path / name).exists()
    assert not (tmp_path / "layer-multimode-experiment-v2.work").exists()


def test_static_preflight_rejects_noncanonical_setup_before_runtime_import(
    tmp_path,
):
    sys.modules.pop("fisher_graph.gemma3_layer_multimode_runtime", None)
    with pytest.raises(ValueError, match="setup path is not canonical"):
        experiment.static_preflight_layer_multimode_experiment(
            repository_root=tmp_path,
            setup_path=tmp_path / "not-the-canonical-setup.json",
        )
    assert "fisher_graph.gemma3_layer_multimode_runtime" not in sys.modules


def test_cli_requires_an_explicit_action_and_is_runtime_inert(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        f"import runpy,sys;sys.path.insert(0,{str(source_root)!r});"
        "runpy.run_module("
        "'fisher_graph.gemma3_layer_multimode_experiment',"
        "run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--preflight-only" in completed.stderr
    assert "--run-v3" in completed.stderr
    assert not (tmp_path / "layer-multimode-experiment-v2.work").exists()


def test_fresh_process_preflight_attempt_does_not_import_model_runtime(
    tmp_path,
):
    source_root = Path(__file__).resolve().parents[1] / "src"
    repository_root = tmp_path / "missing-setup-root"
    (repository_root / ".local-runs/google--gemma-3-270m").mkdir(
        parents=True
    )
    script = (
        "import json,sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "from fisher_graph import gemma3_layer_multimode_experiment as m;"
        "error=None;message=None;"
        "\ntry:\n"
        f" m.main(['--preflight-only','--repository-root',"
        f"{str(repository_root)!r}])\n"
        "except BaseException as caught:\n"
        " error=type(caught).__name__;message=str(caught)\n"
        "print(json.dumps({'error':error,'message':message,'transformers':"
        "'transformers' in sys.modules,'model_loader':"
        "'fisher_graph.gemma3_experiment' in sys.modules,"
        "'v3_runtime':"
        "'fisher_graph.gemma3_layer_multimode_runtime' in sys.modules,"
        "'predecessor_runtime':"
        "'fisher_graph.gemma3_chart_native_neighbor_runtime' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(completed.stdout.splitlines()[-1])
    assert state["error"] == "FileNotFoundError"
    assert "gemma3-layer-multimode-setup-v2.json" in state["message"]
    assert state["transformers"] is False
    assert state["model_loader"] is False
    assert state["v3_runtime"] is False
    assert state["predecessor_runtime"] is False


def test_fresh_process_preflight_reaches_only_missing_IQ_after_setup_gate(
    tmp_path,
):
    source_root = Path(__file__).resolve().parents[1] / "src"
    repository_root = tmp_path / "missing-IQ-root"
    setup_path = (
        repository_root
        / "artifacts/research/gemma3-layer-multimode-setup-v2.json"
    )
    setup_path.parent.mkdir(parents=True)
    setup_path.write_bytes(b"setup-placeholder")
    (repository_root / ".local-runs/google--gemma-3-270m").mkdir(
        parents=True
    )
    script = f"""
import json
import sys
from types import SimpleNamespace
sys.path.insert(0, {str(source_root)!r})
from fisher_graph import gemma3_layer_multimode_experiment as m
from fisher_graph import gemma3_layer_multimode_fresh_panels as panels
from fisher_graph import gemma3_layer_multimode_setup as setup
seed = {'a' * 64!r}
setup.authorize_run = lambda *args, **kwargs: {{
    'implementation_readiness': {{'panel_receipt_file_sha256': seed}}
}}
setup.render_setup_bytes = lambda value: b'setup-placeholder'
panels.load_frozen_preflight = lambda **kwargs: SimpleNamespace()
error = None
message = None
try:
    m.main([
        '--preflight-only',
        '--repository-root',
        {str(repository_root)!r},
    ])
except BaseException as caught:
    error = type(caught).__name__
    message = str(caught)
print(json.dumps({{
    'error': error,
    'message': message,
    'transformers': 'transformers' in sys.modules,
    'model_loader': 'fisher_graph.gemma3_experiment' in sys.modules,
    'v3_runtime': (
        'fisher_graph.gemma3_layer_multimode_runtime' in sys.modules
    ),
    'predecessor_runtime': (
        'fisher_graph.gemma3_chart_native_neighbor_runtime' in sys.modules
    ),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(completed.stdout.splitlines()[-1])
    assert state["error"] == "FileNotFoundError"
    assert "gemma3-layer-multimode-implementation-qualification" in (
        state["message"]
    )
    assert state["transformers"] is False
    assert state["model_loader"] is False
    assert state["v3_runtime"] is False
    assert state["predecessor_runtime"] is False


def test_worker_freeze_contract_integrates_with_real_fresh_panel_source():
    from fisher_graph import gemma3_layer_multimode_fresh_panels as panels

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=True):
            ids = [2] if add_special_tokens else []
            for word in text.split():
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                ids.append(3 + int.from_bytes(digest[:4], "big"))
            return Encoding(ids)

    tokenizer = Tokenizer()
    prior = tuple(
        panels.make_prior_lineage_catalog(generation, [f"{index:064x}"])
        for index, generation in enumerate(panels.PRIOR_GENERATIONS, start=1)
    )
    bundle = panels.author_fresh_panel_bundle(
        tokenizer,
        prior_lineage_catalogs=prior,
    )
    preflight = panels.opaque_preflight(
        manifest_bytes=bundle.manifest_bytes,
        receipt_bytes=bundle.receipt_bytes,
        role_corpus_bytes=bundle.role_corpus_bytes,
        expected_manifest_file_sha256=bundle.manifest_file_sha256,
        expected_receipt_file_sha256=bundle.receipt_file_sha256,
        expected_role_file_sha256s=bundle.role_file_sha256s,
    )
    source = panels.SequentialPanelSource.from_frozen_a_selection(
        preflight,
        bundle.role_corpus_bytes,
        selection_receipt_sha256=_sha("selection"),
        selected_layer_count=18,
    )
    corpus = experiment._panel_open(source, experiment.B3_ROLE)
    tokenization = panels.authenticate_open_role_tokenization(
        corpus,
        role=experiment.B3_ROLE,
        manifest=source.manifest,
        tokenizer=tokenizer,
    )
    assert tokenization.predicted_token_count == 24_732
    experiment._panel_freeze(
        source,
        role=experiment.B3_ROLE,
        receipt_sha256=_sha("B3 failed"),
        terminal_passed=False,
    )
    experiment._assert_panel_closure(source, experiment.TERMINAL_B3_FAILED)
    assert source.terminal_state == "terminal_B3_failed"


def test_no_selection_receipt_rejects_duplicate_grid_hashes():
    with pytest.raises(ValueError, match="324 unique"):
        experiment.LayerMultimodeNoAdmissibleSelectionReceipt(
            source_model_sha256=_sha("source"),
            selection_policy_sha256=_sha("selection-policy"),
            fit_policy_sha256=_sha("fit-policy"),
            fit_a_sha256=_sha("fit-A"),
            validation_a_sha256=_sha("validation-A"),
            evaluation_protocol_sha256=_sha("evaluation-policy"),
            candidate_sha256s=tuple(_sha("same-candidate") for _ in range(324)),
            decision_sha256s=tuple(_sha(f"decision:{i}") for i in range(324)),
            failed_layer_ordinals=(0,),
            first_failed_layer_ordinal=0,
        )
