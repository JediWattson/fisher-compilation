from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison as runner,
)


def _behavioral(
    *,
    passed: object,
    candidate_kl: float,
    candidate_top1: float,
    affected: bool,
) -> dict[str, object]:
    supervised_tokens = 6 if affected else 8
    source_summed_nll = 12.0 if affected else 16.0
    return {
        "aggregate": {
            "example_count": 2,
            "supervised_tokens": supervised_tokens,
            "source_summed_nll": source_summed_nll,
            "candidate_summed_nll": source_summed_nll + 1.0,
            "source_nll_per_token": 2.0,
            "candidate_nll_per_token": 2.0 + 1.0 / supervised_tokens,
            "delta_nll_per_token": 1.0 / supervised_tokens,
            "source_to_candidate_summed_kl": (
                candidate_kl * supervised_tokens
            ),
            "source_to_candidate_kl_per_token": candidate_kl,
            "top1_matches": int(candidate_top1 * supervised_tokens),
            "top1_agreement_to_source": candidate_top1,
        },
        "family_summary": {
            "families": [
                {
                    "family_id": "family-a",
                    "example_count": 1,
                    "supervised_tokens": supervised_tokens // 2,
                    "source_summed_nll": source_summed_nll / 2.0,
                    "source_nll_per_token": 2.0,
                },
                {
                    "family_id": "family-b",
                    "example_count": 1,
                    "supervised_tokens": supervised_tokens // 2,
                    "source_summed_nll": source_summed_nll / 2.0,
                    "source_nll_per_token": 2.0,
                },
            ],
        },
        "per_prompt": {
            "absolute_delta_nll_per_token": {
                "p90": 1.0 / supervised_tokens,
                "worst": 1.0 / supervised_tokens,
            },
            "top1_agreement_to_source": {
                "p10": candidate_top1,
                "worst": candidate_top1,
            },
        },
        "gates": {"passed": passed},
    }


def _evaluation(
    *,
    behavioral_passed: object = False,
    affected_passed: object = False,
    behavioral_kl: float = 4.0,
    affected_kl: float = 4.0,
    top1: float = 0.4,
    affected_top1: float = 0.3,
    modal_error: float = 2.0,
    full_error: float = 2.0,
) -> dict[str, object]:
    return {
        "manifest": {
            "manifest_sha256": "1" * 64,
            "example_count": 2,
            "family_count": 2,
            "strict_example_membership": True,
            "strict_family_membership": True,
            "prompt_text_retained": False,
            "token_ids_retained": False,
        },
        "execution": {
            "total_model_forward_count": 48,
        },
        "behavioral": _behavioral(
            passed=behavioral_passed,
            candidate_kl=behavioral_kl,
            candidate_top1=top1,
            affected=False,
        ),
        "affected_behavioral": _behavioral(
            passed=affected_passed,
            candidate_kl=affected_kl,
            candidate_top1=affected_top1,
            affected=True,
        ),
        "coverage": {
            "example_count": 2,
            "supervised_tokens": 8,
            "affected_supervised_tokens": 6,
            "valid_target_rows": 12,
            "source_eligible_rows": 9,
            "affected_target_rows": 9,
        },
        "target_modal": {
            "pooled": {
                "affected_rows": 9,
                "scalar_elements": 9 * 64,
                "source_signal_l2_norm": 7.0,
                "candidate_signal_l2_norm": 6.0,
                "relative_l2_error": modal_error,
                "cosine": 0.5,
            },
        },
        "full_width_boundary": {
            "pooled": {
                "affected_rows": 9,
                "scalar_elements": 9 * 640,
                "source_signal_l2_norm": 11.0,
                "candidate_signal_l2_norm": 10.0,
                "relative_l2_error": full_error,
                "cosine": 0.4,
            },
        },
        "receipts": [
            {
                "example_id": "a.1",
                "family_id": "family-a",
                "prompt_sha256": "2" * 64,
                "tokenized_tokens": 5,
                "supervised_tokens": 4,
                "affected_supervised_tokens": 3,
                "model_inputs_sha256": "3" * 64,
                "execution_grid_sha256": "4" * 64,
                "result_artifact_sha256": "5" * 64,
                "model_forward_count": 3,
            },
            {
                "example_id": "b.1",
                "family_id": "family-b",
                "prompt_sha256": "6" * 64,
                "tokenized_tokens": 5,
                "supervised_tokens": 4,
                "affected_supervised_tokens": 3,
                "model_inputs_sha256": "7" * 64,
                "execution_grid_sha256": "8" * 64,
                "result_artifact_sha256": "9" * 64,
                "model_forward_count": 3,
            },
        ],
        "safety": {
            "scalar_only_report": True,
            "prompt_text_retained": False,
            "token_ids_retained": False,
            "logits_retained": False,
            "activations_retained": False,
            "candidate_serving_authorized": False,
        },
    }


def _evaluations(
    *,
    local_passed: object = False,
    gfa_passed: object = False,
    global_behavioral_passed: object = False,
    global_affected_passed: object = False,
    global_behavioral_kl: float = 4.0,
    global_affected_kl: float = 4.0,
) -> dict[str, object]:
    return {
        "signed_local_svd_g8": _evaluation(
            behavioral_passed=local_passed,
            affected_passed=local_passed,
        ),
        "signed_gfa_rank45": _evaluation(
            behavioral_passed=gfa_passed,
            affected_passed=gfa_passed,
            behavioral_kl=3.9,
            affected_kl=3.9,
            modal_error=1.95,
            full_error=1.95,
        ),
        "global_svd_rank45": _evaluation(
            behavioral_passed=global_behavioral_passed,
            affected_passed=global_affected_passed,
            behavioral_kl=global_behavioral_kl,
            affected_kl=global_affected_kl,
        ),
    }


def _assert_scalar_tree(value: object) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_scalar_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_scalar_tree(child)
    else:
        assert value is None or isinstance(value, (bool, int, float, str))
        if isinstance(value, float):
            assert math.isfinite(value)


def test_comparison_accepts_candidate_differences_but_freezes_source() -> None:
    comparison = runner.compare_shadow_basis_evaluations(_evaluations())

    assert comparison["source_execution_summary_matched"] is True
    assert comparison["source_hidden_tensor_equality_claim"] is False
    assert len(comparison["source_execution_summary_receipt_sha256"]) == 64
    assert comparison["source_execution_summary_receipt"]["manifest"] == {
        "manifest_sha256": "1" * 64,
        "example_count": 2,
        "family_count": 2,
        "strict_example_membership": True,
        "strict_family_membership": True,
    }
    metrics = comparison["variant_metrics"]
    assert metrics["signed_local_svd_g8"]["target_modal"][
        "relative_l2_error"
    ] == 2.0
    assert metrics["signed_gfa_rank45"]["target_modal"][
        "relative_l2_error"
    ] == 1.95
    _assert_scalar_tree(comparison)
    serialized = json.dumps(comparison, sort_keys=True, allow_nan=False)
    assert "synthetic prompt" not in serialized
    assert '"token_ids"' not in serialized
    assert '"logits"' not in serialized


_SourceMutation = Callable[[dict[str, object]], None]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["manifest"].__setitem__(
            "manifest_sha256", "f" * 64
        ),
        lambda value: value["behavioral"]["aggregate"].__setitem__(
            "source_summed_nll", 16.25
        ),
        lambda value: value["behavioral"]["family_summary"]["families"][
            0
        ].__setitem__("source_nll_per_token", 2.25),
        lambda value: value["affected_behavioral"]["aggregate"].__setitem__(
            "source_nll_per_token", 2.25
        ),
        lambda value: value["coverage"].__setitem__(
            "affected_supervised_tokens", 5
        ),
        lambda value: value["target_modal"]["pooled"].__setitem__(
            "source_signal_l2_norm", 7.25
        ),
        lambda value: value["full_width_boundary"]["pooled"].__setitem__(
            "scalar_elements", 9 * 641
        ),
        lambda value: value["receipts"][0].__setitem__(
            "model_inputs_sha256", "e" * 64
        ),
        lambda value: value["receipts"][0].__setitem__(
            "execution_grid_sha256", "c" * 64
        ),
        lambda value: value["receipts"][1].__setitem__(
            "prompt_sha256", "d" * 64
        ),
    ),
    ids=(
        "manifest",
        "aggregate-source-nll",
        "family-source-nll",
        "affected-source-nll",
        "coverage",
        "modal-source-norm",
        "full-width-source-elements",
        "model-inputs",
        "execution-grid",
        "prompt-identity",
    ),
)
def test_comparison_rejects_any_source_execution_drift(
    mutate: _SourceMutation,
) -> None:
    evaluations = _evaluations()
    changed = evaluations["signed_gfa_rank45"]
    assert isinstance(changed, dict)
    mutate(changed)

    with pytest.raises(ValueError, match="source execution differs"):
        runner.compare_shadow_basis_evaluations(evaluations)


@pytest.mark.parametrize(
    ("local_passed", "gfa_passed", "global_passed", "expected"),
    (
        (False, False, False, "no_rank45_basis_viable_attribution_inconclusive"),
        (
            False,
            False,
            True,
            "global_svd_only_viable_basis_construction_is_blocker",
        ),
        (
            False,
            True,
            False,
            "graph_specific_reversal_attribution_inconclusive",
        ),
        (
            False,
            True,
            True,
            "signed_gfa_and_global_svd_viable_local_svd_grouping_rejected",
        ),
        (
            True,
            False,
            False,
            "graph_specific_reversal_attribution_inconclusive",
        ),
        (
            True,
            False,
            True,
            "local_svd_and_global_svd_viable_signed_gfa_rejected",
        ),
        (
            True,
            True,
            False,
            "graph_specific_reversal_attribution_inconclusive",
        ),
        (
            True,
            True,
            True,
            "rank45_linear_carrier_viable_across_all_three_bases",
        ),
    ),
)
def test_classification_covers_all_arm_pass_patterns(
    local_passed: bool,
    gfa_passed: bool,
    global_passed: bool,
    expected: str,
) -> None:
    comparison = runner.compare_shadow_basis_evaluations(
        _evaluations(
            local_passed=local_passed,
            gfa_passed=gfa_passed,
            global_behavioral_passed=global_passed,
            global_affected_passed=global_passed,
        )
    )

    assert comparison["classification"] == expected
    assert comparison["pass_pattern"] == "".join(
        "1" if value else "0"
        for value in (local_passed, gfa_passed, global_passed)
    )
    assert comparison["arm_passes"] == {
        "signed_local_svd_g8": local_passed,
        "signed_gfa_rank45": gfa_passed,
        "global_svd_rank45": global_passed,
    }
    assert comparison[
        "global_svd_axiswise_dominates_both_graph_arms"
    ] is False


def test_all_fail_classification_uses_strict_affected_axis_dominance() -> None:
    tied = runner.compare_shadow_basis_evaluations(
        _evaluations(global_affected_kl=3.9)
    )
    dominant = runner.compare_shadow_basis_evaluations(
        _evaluations(
            global_affected_kl=math.nextafter(3.9, -math.inf),
        )
    )

    assert tied["pass_pattern"] == dominant["pass_pattern"] == "000"
    assert tied["global_svd_axiswise_dominates_both_graph_arms"] is False
    assert tied["classification"] == (
        "no_rank45_basis_viable_attribution_inconclusive"
    )
    assert dominant[
        "global_svd_axiswise_dominates_both_graph_arms"
    ] is True
    assert dominant["classification"] == (
        "basis_contributes_but_rank45_fixed_reference_family_still_fails"
    )


def test_gate_flags_require_literal_true_for_both_views() -> None:
    comparison = runner.compare_shadow_basis_evaluations(
        _evaluations(
            global_behavioral_passed=1,
            global_affected_passed=True,
        )
    )

    assert comparison["arm_passes"]["global_svd_rank45"] is False
    assert comparison["pass_pattern"] == "000"
    assert comparison["classification"] == (
        "no_rank45_basis_viable_attribution_inconclusive"
    )


@pytest.mark.parametrize(
    ("behavioral_passed", "affected_passed"),
    ((True, False), (False, True)),
)
def test_arm_pass_requires_both_ordinary_and_affected_gates(
    behavioral_passed: bool,
    affected_passed: bool,
) -> None:
    comparison = runner.compare_shadow_basis_evaluations(
        _evaluations(
            global_behavioral_passed=behavioral_passed,
            global_affected_passed=affected_passed,
        )
    )

    assert comparison["arm_passes"]["global_svd_rank45"] is False
    assert comparison["pass_pattern"] == "000"
    assert comparison["classification_protocol"][
        "arm_pass_requires_behavioral_and_affected_gates"
    ] is True


def test_variant_order_and_scalar_metric_contract_fail_closed() -> None:
    evaluations = _evaluations()
    reordered = {
        "signed_gfa_rank45": evaluations["signed_gfa_rank45"],
        "signed_local_svd_g8": evaluations["signed_local_svd_g8"],
        "global_svd_rank45": evaluations["global_svd_rank45"],
    }
    with pytest.raises(ValueError, match="frozen variant order"):
        runner.compare_shadow_basis_evaluations(reordered)

    nonfinite = deepcopy(evaluations)
    nonfinite["global_svd_rank45"]["target_modal"]["pooled"][
        "relative_l2_error"
    ] = math.nan
    with pytest.raises(ValueError):
        runner.compare_shadow_basis_evaluations(nonfinite)

    nonscalar = deepcopy(evaluations)
    nonscalar["global_svd_rank45"]["target_modal"]["pooled"][
        "relative_l2_error"
    ] = object()
    with pytest.raises(TypeError):
        runner.compare_shadow_basis_evaluations(nonscalar)


def test_output_must_be_json_below_local_runs(tmp_path: Path) -> None:
    valid = tmp_path / ".local-runs" / "comparison.json"

    assert runner._validate_output(valid) == valid
    with pytest.raises(ValueError, match="JSON under .local-runs"):
        runner._validate_output(tmp_path / "comparison.json")
    with pytest.raises(ValueError, match="JSON under .local-runs"):
        runner._validate_output(tmp_path / ".local-runs" / "comparison.pt")


class _FakePlan:
    def __init__(self, ordinal: int) -> None:
        self.artifact_sha256 = f"{ordinal:x}" * 64
        self.response_binding_sha256 = "a" * 64
        self.fit_weighted_kernels_sha256 = "b" * 64
        self.fit_knot_origins = (8, 24, 40)
        self.source_modes = 64
        self.source_rank = 45
        self.target_modes = 64
        self.target_rank = 64
        self.lag_count = 32
        self.fft_length = 64
        self.input_transform = "standardized_linear"
        self.input_transform_semantics = "source_standard_deviation"
        self.square_transform_scope = "disabled"
        self.interpolation_semantics = "piecewise_linear_fit_knots"
        self.factorization_semantics = "rank45_source_full_target"
        self.core_semantics = "causal_lag_kernel"
        self.fit_origin_scope = "fit_only"
        self.heldout_origins_used_for_fit = False
        self.cross_mode_terms_measured = False
        self.stored_coefficient_count = 283_456
        self.rank_semantics = "effective_source_rank"
        self._ordinal = ordinal

    def accounting(self) -> object:
        return SimpleNamespace(prepared_storage_bytes=2_268_184)

    def metadata(self) -> dict[str, object]:
        return {
            "tensor_sha256s": {
                "source_basis": f"{self._ordinal + 3:x}" * 64,
                "target_basis": "7" * 64,
                "source_scales": "8" * 64,
                "target_singular_values": "9" * 64,
            },
        }


def _plans() -> dict[str, _FakePlan]:
    return {
        name: _FakePlan(index)
        for name, index in zip(runner._VARIANT_ORDER, (1, 2, 3), strict=True)
    }


def test_rank45_plan_panel_requires_unique_matched_plan_identities() -> None:
    plans = _plans()

    receipt = runner._validate_rank45_basis_plans(plans)

    assert receipt["plan_artifact_sha256s"] == tuple(
        plans[name].artifact_sha256 for name in runner._VARIANT_ORDER
    )
    assert len(set(receipt["source_basis_sha256s"])) == 3
    assert receipt["shared_invariants"]["source_rank"] == 45
    assert receipt["shared_invariants"]["target_rank"] == 64
    assert receipt["shared_invariants"]["prepared_storage_bytes"] == 2_268_184


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda plans: setattr(
                plans["signed_gfa_rank45"],
                "artifact_sha256",
                plans["signed_local_svd_g8"].artifact_sha256,
            ),
            "unique artifact identities",
        ),
        (
            lambda plans: setattr(
                plans["signed_gfa_rank45"],
                "_ordinal",
                plans["signed_local_svd_g8"]._ordinal,
            ),
            "unique source bases",
        ),
        (
            lambda plans: setattr(plans["global_svd_rank45"], "lag_count", 31),
            "not rank, geometry, and size matched",
        ),
        (
            lambda plans: [
                setattr(plan, "source_rank", 44) for plan in plans.values()
            ],
            "frozen rank-45 geometry",
        ),
    ),
    ids=("duplicate-plan", "duplicate-basis", "geometry-drift", "wrong-rank"),
)
def test_rank45_plan_panel_rejects_identity_or_geometry_drift(
    mutate: Callable[[dict[str, _FakePlan]], None],
    message: str,
) -> None:
    plans = _plans()
    mutate(plans)

    with pytest.raises(ValueError, match=message):
        runner._validate_rank45_basis_plans(plans)


def test_variant_receipt_binds_role_plan_and_common_lineage() -> None:
    plan = _FakePlan(1)
    common = {
        "panel_file_sha256": "e" * 64,
        "factorized_live_model_sha256": "f" * 64,
    }

    first = runner._variant_receipt(
        role="signed_local_svd_g8",
        plan=plan,
        common_binding=common,
    )
    replay = runner._variant_receipt(
        role="signed_local_svd_g8",
        plan=plan,
        common_binding=common,
    )
    other_role = runner._variant_receipt(
        role="signed_gfa_rank45",
        plan=plan,
        common_binding=common,
    )

    assert first == replay
    assert first["plan_artifact_sha256"] == plan.artifact_sha256
    assert first["source_basis_sha256"] == plan.metadata()["tensor_sha256s"][
        "source_basis"
    ]
    assert first["common_binding"] == common
    assert first["candidate_serving_authorized"] is False
    assert first["artifact_sha256"] != other_role["artifact_sha256"]


def _install_orchestration_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_variant: str | None = None,
) -> tuple[list[object], object]:
    events: list[object] = []
    examples = ("example-a", "example-b")
    fit_source = SimpleNamespace(
        file_sha256="a" * 64,
        report_file_sha256="b" * 64,
        report_payload_sha256="c" * 64,
        mapping=SimpleNamespace(artifact_sha256="d" * 64),
    )
    parent = SimpleNamespace(artifact_sha256="b" * 64)
    plans = tuple(_FakePlan(index) for index in (1, 2, 3))
    candidate = SimpleNamespace(
        artifact_sha256="c" * 64,
        plan=plans[0],
        binding={"basis": "synthetic"},
        model={
            "model_id": "synthetic/gemma",
            "resolved_commit": "d" * 40,
            "source_model_sha256": runner._EXPECTED_RAW_MODEL_SHA256,
        },
    )
    basis = object()
    tokenizer = object()

    class FakeAdapter:
        factorized = False

        def model_fingerprint(self) -> str:
            return (
                runner._EXPECTED_FACTORIZED_MODEL_SHA256
                if self.factorized
                else runner._EXPECTED_RAW_MODEL_SHA256
            )

        def execution_fingerprint(self) -> str:
            assert self.factorized
            return runner._EXPECTED_FACTORIZED_EXECUTION_SHA256

    adapter = FakeAdapter()

    class FakeSwitcher:
        def __init__(self, received_adapter: object, scopes: object) -> None:
            assert received_adapter is adapter
            assert scopes == {runner._FACTORIZED_SCOPE: ("replacement",)}
            events.append("switcher-created")

        def switch(self, scope: str) -> None:
            assert scope == runner._FACTORIZED_SCOPE
            assert adapter.factorized is False
            adapter.factorized = True
            events.append("factorized")

        def close(self) -> None:
            adapter.factorized = False
            events.append("restored")

    class FakeRuntime:
        def __init__(
            self,
            plan: object,
            received_basis: object,
            **kwargs: object,
        ) -> None:
            assert adapter.factorized
            assert received_basis is basis
            self.name = str(kwargs["candidate_method"])
            runtime_count = sum(
                isinstance(event, tuple) and event[0] == "runtime"
                for event in events
            )
            assert self.name == runner._VARIANT_ORDER[runtime_count]
            assert kwargs["expected_plan_artifact_sha256"] == (
                plan.artifact_sha256
            )
            assert kwargs["expected_live_model_sha256"] == (
                runner._EXPECTED_FACTORIZED_MODEL_SHA256
            )
            assert kwargs["expected_adapter_execution_sha256"] == (
                runner._EXPECTED_FACTORIZED_EXECUTION_SHA256
            )
            self.plan = plan
            self.candidate_artifact_sha256 = str(
                kwargs["candidate_artifact_sha256"]
            )
            events.append(("runtime", self.name))

        def metadata(self) -> dict[str, object]:
            return {
                "candidate_method": self.name,
                "candidate_artifact_sha256": self.candidate_artifact_sha256,
                "plan_artifact_sha256": self.plan.artifact_sha256,
                "runtime_binding_sha256": hashlib_sha(self.name),
                "candidate_serving_authorized": False,
            }

        def validate_integrity(self) -> None:
            events.append(("validated", self.name))

    def hashlib_sha(value: str) -> str:
        # Deterministic lowercase SHA-like fixture value without introducing
        # any real prompt or artifact dependency.
        digit = str(runner._VARIANT_ORDER.index(value) + 4)
        return digit * 64

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        assert adapter.factorized
        assert kwargs["adapter"] is adapter
        assert kwargs["tokenizer"] is tokenizer
        assert kwargs["examples"] is examples
        assert kwargs["max_length"] == 96
        assert kwargs["model_input_device"] == "cpu"
        runtime = kwargs["runtime"]
        events.append(("evaluate", runtime.name))
        if runtime.name == fail_variant:
            raise RuntimeError(f"synthetic failure in {runtime.name}")
        return _evaluation(
            behavioral_kl={
                "signed_local_svd_g8": 4.0,
                "signed_gfa_rank45": 3.9,
                "global_svd_rank45": 3.0,
            }[runtime.name],
        )

    monkeypatch.setattr(
        runner,
        "_load_panel",
        lambda _path: (
            events.append("panel-loaded")
            or (
                examples,
                {
                    "file_sha256": "e" * 64,
                    "source_fit_prompt_index_sha256": "f" * 64,
                    "example_count": 2,
                    "family_count": 2,
                    "contains_prompt_text": False,
                },
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_spectral_source",
        lambda *_args, **_kwargs: events.append("fit-source-loaded")
        or fit_source,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_graph_wavelet_candidate",
        lambda *_args, **_kwargs: events.append("parent-loaded") or parent,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate",
        lambda *_args, **_kwargs: events.append("candidate-loaded")
        or candidate,
    )
    monkeypatch.setattr(
        runner,
        "_reference_plans",
        lambda received_source, received_parent: (
            plans[1],
            plans[2],
        )
        if (received_source is fit_source and received_parent is parent)
        else (_ for _ in ()).throw(AssertionError("reference lineage drift")),
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_basis_package",
        lambda *_args, **_kwargs: events.append("basis-loaded") or basis,
    )
    monkeypatch.setattr(
        runner,
        "default_gemma3_l3_l4_graph_organized_svd_shadow_protocol",
        lambda: "synthetic-tokenizer-protocol",
    )
    monkeypatch.setattr(
        runner,
        "_load_and_validate_frozen_local_tokenizer",
        lambda **_kwargs: events.append("tokenizer-loaded")
        or (
            tokenizer,
            {
                "tokenizer_class": "SyntheticTokenizer",
                "configuration_sha256": "1" * 64,
                "backend_serialized_sha256": "2" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "_frozen_tokenizer_integrity_check",
        lambda received, _contract: (
            (lambda stage: events.append(("tokenizer-integrity", stage)))
            if received is tokenizer
            else (_ for _ in ()).throw(AssertionError("wrong tokenizer"))
        ),
    )
    monkeypatch.setattr(runner, "resolve_torch_device", lambda _value: "cpu")
    monkeypatch.setattr(
        runner,
        "resolve_gemma3_huggingface_paths",
        lambda cache_dir: {"hub_cache": cache_dir},
    )
    monkeypatch.setattr(
        runner,
        "_load_local_gemma3_model_only",
        lambda **_kwargs: events.append("model-loaded") or object(),
    )
    monkeypatch.setattr(runner, "Gemma3CausalLMAdapter", lambda _model: adapter)
    monkeypatch.setattr(
        runner,
        "restore_gemma3_full_mlp_stack_refit_runtime",
        lambda *_args: events.append("refit-loaded")
        or SimpleNamespace(replacements=("replacement",)),
    )
    monkeypatch.setattr(runner, "PreparedGemma3FullMLPStackSwitcher", FakeSwitcher)
    monkeypatch.setattr(
        runner,
        "Gemma3L3L4ConditionalSpectralShadowRuntime",
        FakeRuntime,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_gemma3_l3_l4_conditional_spectral_development_shadow",
        fake_evaluate,
    )
    return events, adapter


def test_orchestration_reuses_one_model_tokenizer_and_factorized_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, adapter = _install_orchestration_fakes(monkeypatch)
    output = tmp_path / ".local-runs" / "comparison.json"
    output.parent.mkdir(parents=True)

    report = runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison(
        output=output,
        cache_dir=tmp_path / "cache",
        max_length=96,
    )

    assert events.count("model-loaded") == 1
    assert events.count("tokenizer-loaded") == 1
    assert events.count("factorized") == 1
    assert events.count("restored") == 1
    evaluation_events = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "evaluate"
    ]
    assert evaluation_events == [
        ("evaluate", name) for name in runner._VARIANT_ORDER
    ]
    assert adapter.factorized is False
    assert report["resource_accounting"] == {
        "model_load_count": 1,
        "tokenizer_load_count": 1,
        "variant_count": 3,
        "total_model_forward_count": 144,
        "all_plans_size_matched": True,
        "whole_model_parameter_reduction_claim": False,
        "latency_or_speed_claim": False,
    }
    assert report["comparison"]["source_execution_summary_matched"] is True
    assert report["comparison"]["source_hidden_tensor_equality_claim"] is False
    assert tuple(report["variants"]) == runner._VARIANT_ORDER
    assert report["scientific_status"]["formal_qualification"] is False
    assert report["scientific_status"]["candidate_serving_authorized"] is False
    published = json.loads(output.read_text(encoding="utf-8"))
    _assert_scalar_tree(published)
    assert published["safety"]["contains_prompt_text"] is False
    assert published["safety"]["contains_token_ids"] is False
    assert published["safety"]["contains_logits"] is False
    assert published["safety"]["contains_activation_tensors"] is False


def test_orchestration_restores_native_stack_after_mid_panel_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, adapter = _install_orchestration_fakes(
        monkeypatch,
        fail_variant="signed_gfa_rank45",
    )
    output = tmp_path / ".local-runs" / "comparison.json"
    output.parent.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison(
            output=output,
            max_length=96,
        )

    assert adapter.factorized is False
    assert events.count("factorized") == 1
    assert events.count("restored") == 1
    assert ("evaluate", "signed_local_svd_g8") in events
    assert ("evaluate", "signed_gfa_rank45") in events
    assert ("evaluate", "global_svd_rank45") not in events
    assert not output.exists()
