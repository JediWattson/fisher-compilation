from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_continuous_response_smoke as smoke,
)


_BASE_SHA = "1" * 64
_PROPOSAL_SHA = "2" * 64
_LAW_EVIDENCE_SHA = "3" * 64
_FIT_SHA = "4" * 64


def _rehash_provider_receipt(receipt: dict[str, object]) -> None:
    payload = {
        key: value for key, value in receipt.items() if key != "artifact_sha256"
    }
    receipt["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._PROVIDER_RECEIPT_DOMAIN
    )


def _semantic_provider_receipts(
    *, selected_coordinate_index: int
) -> dict[str, dict[str, object]]:
    definitions = {
        "base": ("base_zero", "base_zero", 0, None),
        "signed_log": ("direct", "signed_log", 1, 9.0),
        "constant_plus_one": ("constant", "linear", 1, 9.0),
        "signed_log_sign_flip": ("direct", "signed_log", -1, 9.0),
        "linear": ("direct", "linear", 1, 9.0),
    }
    result: dict[str, dict[str, object]] = {}
    for index, arm in enumerate(smoke._ARMS):
        source, law, polarity, kappa = definitions[arm]
        continuous = arm != "base"
        receipt: dict[str, object] = {
            "arm": arm,
            "provider_artifact_sha256": (
                _BASE_SHA if arm == "base" else f"{10 + index:064x}"
            ),
            "provider_metadata_sha256": f"{20 + index:064x}",
            "base_provider_artifact_sha256": _BASE_SHA,
            "proposal_provider_artifact_sha256": (
                _PROPOSAL_SHA if continuous else None
            ),
            "response_weight_sha256": smoke._expected_response_weight_sha256(
                arm=arm, selected_coordinate_index=selected_coordinate_index
            ),
            "response_source": source,
            "response_law": law,
            "polarity": polarity,
            "signed_log_kappa": kappa,
            "transfer_protocol_sha256": (
                smoke._core.CONTINUOUS_RESPONSE_PROTOCOL_SHA256
                if continuous
                else None
            ),
            "transfer_evidence_sha256": (
                _LAW_EVIDENCE_SHA if continuous else None
            ),
            "rank": 256,
            "conditional_rank": 16,
            "prepared_float_scalar_count": 394_515 if continuous else 377_608,
            "logical_macs_per_token_upper_bound": (
                565_769 if continuous else 541_187
            ),
            "analysis_only": continuous,
        }
        _rehash_provider_receipt(receipt)
        result[arm] = receipt
    return result


def _families() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                smoke._REED,
                smoke._SUNDIAL,
                "family-alpine",
                "family-cave",
                "family-kiln",
                "family-obsidian",
                "family-shell",
                "family-varve",
            )
        )
    )


def _frozen_fragment() -> dict[str, object]:
    fit_families = tuple(
        family for family in _families() if family not in smoke._FROZEN_EXCLUDED
    )
    return {
        "fragment_sha256": smoke._FROZEN_PAIR_FRAGMENT_SHA256,
        "pair_key": smoke._FROZEN_PAIR_KEY,
        "excluded_family_ids": smoke._FROZEN_EXCLUDED,
        "shared_fit_receipt": {
            "fit_key": smoke._FROZEN_PAIR_KEY,
            "excluded_family_ids": smoke._FROZEN_EXCLUDED,
            "training_family_ids": fit_families,
        },
        "directed_panels": (
            {
                "outer_held_family_id": smoke._REED,
                "inner_held_family_id": smoke._SUNDIAL,
            },
            {
                "outer_held_family_id": smoke._SUNDIAL,
                "inner_held_family_id": smoke._REED,
            },
        ),
    }


def test_source_report_file_hash_fails_closed_before_json_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(smoke._v20b, "_secure_stat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(smoke._v14, "_file_sha256", lambda _path: "0" * 64)

    def forbidden_read(_self: Path) -> bytes:
        events.append("read")
        raise AssertionError("JSON must not be read after file-hash failure")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(RuntimeError, match="file hash drifted"):
        smoke._load_authenticated_v20b_source()
    assert events == []


def test_pair_freeze_accepts_only_exact_reciprocal_six_family_complement() -> None:
    fragment = _frozen_fragment()
    assert smoke._validate_frozen_pair_fragment(
        fragment, family_ids=_families()
    )["pair_key"] == smoke._FROZEN_PAIR_KEY

    held_leak = copy.deepcopy(fragment)
    held_leak["shared_fit_receipt"]["training_family_ids"] = (
        smoke._REED,
        *tuple(held_leak["shared_fit_receipt"]["training_family_ids"])[1:],
    )
    with pytest.raises(RuntimeError, match="pair authority differs"):
        smoke._validate_frozen_pair_fragment(
            held_leak, family_ids=_families()
        )

    wrong_role = copy.deepcopy(fragment)
    wrong_role["directed_panels"][0]["inner_held_family_id"] = "family-alpine"
    with pytest.raises(RuntimeError, match="pair authority differs"):
        smoke._validate_frozen_pair_fragment(
            wrong_role, family_ids=_families()
        )


def _record(family: str, index: int) -> object:
    mask = torch.tensor([True, True, False])
    return SimpleNamespace(
        sequence=SimpleNamespace(
            example_id=f"{family}/example-{index}",
            family_id=family,
            artifact_sha256=f"{index + 1:064x}",
            support_mask=mask,
        )
    )


def test_coordinate_statistics_cannot_read_either_excluded_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = _families()
    fit_families = tuple(
        family for family in families if family not in smoke._FROZEN_EXCLUDED
    )
    records = tuple(
        _record(family, 2 * family_index + prompt_index)
        for family_index, family in enumerate(fit_families)
        for prompt_index in range(2)
    )

    class _Provider:
        parent_provider = object()
        fit_family_ids = fit_families

        def bounded_coordinates(self, parent: torch.Tensor) -> torch.Tensor:
            return parent

    monkeypatch.setattr(
        smoke,
        "_training_parent_modal",
        lambda _provider, sequence: torch.tensor(
            [
                [0.1 + 0.001 * len(sequence.example_id), -0.2],
                [-0.3, 0.4],
                [0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    grouped, receipt = smoke._fit_coordinates_by_family(
        records,
        base_provider=_Provider(),
        family_ids=families,
        excluded_family_ids=smoke._FROZEN_EXCLUDED,
    )
    assert set(grouped) == set(fit_families)
    assert receipt["coordinate_source_family_ids"] == fit_families
    assert set(receipt["coordinate_source_family_ids"]).isdisjoint(
        smoke._FROZEN_EXCLUDED
    )
    assert receipt["raw_coordinates_serialized"] is False

    leaked = records + (_record(smoke._REED, 99),)
    with pytest.raises(PermissionError, match="held family"):
        smoke._fit_coordinates_by_family(
            leaked,
            base_provider=_Provider(),
            family_ids=families,
            excluded_family_ids=smoke._FROZEN_EXCLUDED,
        )


def test_signed_log_mirror_must_be_bitwise_exact() -> None:
    positive = {
        "_transient_gain_values": {
            "a": torch.tensor([-0.75, 0.0, 0.25], dtype=torch.float64),
            "b": torch.tensor([0.5], dtype=torch.float64),
        }
    }
    mirror = {
        "_transient_gain_values": {
            key: -value for key, value in positive["_transient_gain_values"].items()
        }
    }
    smoke._assert_exact_mirror(positive, mirror)
    forged = copy.deepcopy(mirror)
    forged["_transient_gain_values"]["a"][0] += 1.0e-12
    with pytest.raises(RuntimeError, match="mirror is not exact"):
        smoke._assert_exact_mirror(positive, forged)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("response_source", "tanh_projection"),
        ("polarity", -1),
        ("signed_log_kappa", 8.0),
        ("prepared_float_scalar_count", 394_514),
    ),
)
def test_provider_semantics_fail_closed_on_arm_tamper(
    field: str, value: object
) -> None:
    receipts = _semantic_provider_receipts(selected_coordinate_index=0)
    smoke._validate_provider_receipt_semantics(
        receipts,
        selected_coordinate_index=0,
        law_evidence_sha256=_LAW_EVIDENCE_SHA,
        base_provider_artifact_sha256=_BASE_SHA,
        proposal_provider_artifact_sha256=_PROPOSAL_SHA,
    )
    receipts["signed_log"][field] = value
    _rehash_provider_receipt(receipts["signed_log"])
    with pytest.raises(ValueError, match="signed_log provider semantics differ"):
        smoke._validate_provider_receipt_semantics(
            receipts,
            selected_coordinate_index=0,
            law_evidence_sha256=_LAW_EVIDENCE_SHA,
            base_provider_artifact_sha256=_BASE_SHA,
            proposal_provider_artifact_sha256=_PROPOSAL_SHA,
        )


def test_provider_semantics_bind_selected_axis_and_base_artifact() -> None:
    receipts = _semantic_provider_receipts(selected_coordinate_index=0)
    with pytest.raises(ValueError, match="provider semantics differ"):
        smoke._validate_provider_receipt_semantics(
            receipts,
            selected_coordinate_index=1,
            law_evidence_sha256=_LAW_EVIDENCE_SHA,
            base_provider_artifact_sha256=_BASE_SHA,
            proposal_provider_artifact_sha256=_PROPOSAL_SHA,
        )


def _provider_binding_fixture() -> tuple[
    dict[str, dict[str, object]],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    receipts = _semantic_provider_receipts(selected_coordinate_index=0)
    roles: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for role_index, (outer, held) in enumerate(
        ((smoke._REED, smoke._SUNDIAL), (smoke._SUNDIAL, smoke._REED))
    ):
        ids = (f"held-{role_index}-0", f"held-{role_index}-1")
        arm_evidence: dict[str, dict[str, object]] = {}
        scores: list[dict[str, object]] = []
        for arm_index, arm in enumerate(smoke._ARMS):
            provider_sha = str(receipts[arm]["provider_artifact_sha256"])
            seed = 1000 * role_index + 100 * arm_index
            h4 = {item: f"{seed + 10 + index:064x}" for index, item in enumerate(ids)}
            logits = {
                item: f"{seed + 20 + index:064x}" for index, item in enumerate(ids)
            }
            trace_payload = {
                "arm": arm,
                "provider_artifact_sha256": provider_sha,
                "scored_family_ids": (held,),
                "response_gain_sha256s": {
                    item: f"{seed + 30 + index:064x}"
                    for index, item in enumerate(ids)
                },
                "runtime_receipt_sha256": f"{seed + 40:064x}",
                "finite": True,
                "pointwise_trust_passed": True,
                "max_bounded_direction_to_parent_norm_ratio": 0.25,
                "max_emitted_delta_to_parent_norm_ratio": 0.2,
                "endpoint_conditional_ranks_are_16": True,
                "raw_response_or_modal_tensors_serialized": False,
            }
            trace = {
                **trace_payload,
                "artifact_sha256": smoke._v14._sha256(
                    trace_payload, domain=smoke._RESPONSE_TRACE_DOMAIN
                ),
            }
            objective = 1.0 + 0.01 * arm_index
            execution_sha = smoke._execution_receipt(
                arm=arm,
                provider_artifact_sha256=provider_sha,
                fit_receipt_sha256=_FIT_SHA,
                outer_family_id=outer,
                scored_family_id=held,
                h4_sha256s=h4,
                logits_sha256s=logits,
                response_trace_sha256=str(trace["artifact_sha256"]),
                objective=objective,
            )
            changed = arm != "base"
            arm_evidence[arm] = {
                "arm": arm,
                "objective": objective,
                "provider_artifact_sha256": provider_sha,
                "execution_receipt_sha256": execution_sha,
                "post_cast_h4_sha256s": h4,
                "supervised_full_vocab_logits_sha256s": logits,
                "response_trace": trace,
                "execution_changed_from_base": changed,
            }
            scores.append(
                {
                    "arm": arm,
                    "objective": objective,
                    "execution_receipt_sha256": execution_sha,
                    "response_trace_sha256": trace["artifact_sha256"],
                    "finite": True,
                    "pointwise_trust_passed": True,
                    "rank_is_16": True,
                    "score_source": "exact_finite_execution",
                    "predicted_only": False,
                    "execution_changed_from_base": changed,
                }
            )
        capability = {
            "artifact_sha256": f"{9000 + role_index:064x}",
            "held_family_id": outer,
            "authorized_example_count": 2,
            "authorized_family_count": 1,
            "access_count": 10,
            "per_example_access_counts": {item: 5 for item in ids},
            "held_family_capability_excluded": True,
            "teacher_rows_consumed_only_through_capability": True,
        }
        evidence_payload = {
            "outer_held_family_id": outer,
            "scored_inner_family_id": held,
            "capability_receipt": capability,
            "arm_execution_evidence": arm_evidence,
            "signed_log_mirror_exact": True,
            "law_and_all_providers_frozen_before_capability": True,
        }
        evidence_rows.append(
            {
                **evidence_payload,
                "artifact_sha256": smoke._v14._sha256(
                    evidence_payload, domain=smoke._ROLE_EVIDENCE_DOMAIN
                ),
            }
        )
        roles.append(
            {
                "outer_held_family_id": outer,
                "held_family_id": held,
                "arm_scores": tuple(scores),
            }
        )
    return receipts, tuple(roles), tuple(evidence_rows)


def test_continuous_provider_artifact_is_bound_to_every_exact_execution() -> None:
    receipts, roles, evidence = _provider_binding_fixture()
    smoke._validate_provider_execution_bindings(
        receipts,
        roles=roles,
        role_evidence=evidence,
        selected_coordinate_index=0,
        law_evidence_sha256=_LAW_EVIDENCE_SHA,
        base_provider_artifact_sha256=_BASE_SHA,
        proposal_provider_artifact_sha256=_PROPOSAL_SHA,
        fit_receipt_sha256=_FIT_SHA,
    )
    receipts["signed_log"]["provider_artifact_sha256"] = "f" * 64
    _rehash_provider_receipt(receipts["signed_log"])
    with pytest.raises(ValueError, match="provider execution binding differs"):
        smoke._validate_provider_execution_bindings(
            receipts,
            roles=roles,
            role_evidence=evidence,
            selected_coordinate_index=0,
            law_evidence_sha256=_LAW_EVIDENCE_SHA,
            base_provider_artifact_sha256=_BASE_SHA,
            proposal_provider_artifact_sha256=_PROPOSAL_SHA,
            fit_receipt_sha256=_FIT_SHA,
        )

    receipts = _semantic_provider_receipts(selected_coordinate_index=0)
    receipts["base"]["provider_artifact_sha256"] = "9" * 64
    _rehash_provider_receipt(receipts["base"])
    with pytest.raises(ValueError, match="base provider semantics differ"):
        smoke._validate_provider_receipt_semantics(
            receipts,
            selected_coordinate_index=0,
            law_evidence_sha256=_LAW_EVIDENCE_SHA,
            base_provider_artifact_sha256=_BASE_SHA,
            proposal_provider_artifact_sha256=_PROPOSAL_SHA,
        )


@pytest.mark.parametrize(
    ("arm", "axis"),
    (("constant_plus_one", 0), ("signed_log", 0), ("linear", 1)),
)
def test_expected_response_weight_uses_provider_parameter_hash_domain(
    arm: str, axis: int
) -> None:
    weight = torch.zeros(3, dtype=torch.float64)
    if arm != "constant_plus_one":
        weight[axis] = 1.0
    expected = smoke._expected_response_weight_sha256(
        arm=arm, selected_coordinate_index=axis
    )
    assert expected == smoke._provider_tensor_sha256(weight)
    assert expected != smoke._v14._tensor_sha256(weight)


def test_existing_report_fast_path_never_constructs_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing-v20c.json"
    output.touch()
    expected = {"report_sha256": "a" * 64}
    monkeypatch.setattr(smoke, "_validate_output", lambda _value: output)
    monkeypatch.setattr(smoke, "_load_existing_report", lambda path: expected)

    def forbidden_model(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("existing V20c output must authenticate without a model")

    monkeypatch.setattr(
        smoke, "prepare_complete_h4_rank320_live_context", forbidden_model
    )
    assert smoke.run_gemma3_l3_l4_complete_h4_continuous_response_smoke(
        output=output
    ) == expected


def test_work_accounting_is_exact_for_one_fit_two_roles_five_arms() -> None:
    work = smoke._work_accounting()
    assert work == {
        "full_model_forward_count": 64,
        "full_suffix_backward_traversal_count": 28,
        "local_head_autograd_contraction_count": 12,
        "total_autograd_grad_call_count": 40,
        "teacher_capability_access_count": 32,
        "post_cast_h4_hash_check_count": 32,
        "supervised_full_vocab_logits_hash_check_count": 32,
        "physical_pair_endpoint_fit_count": 1,
        "held_role_count": 2,
        "exact_arm_score_count": 10,
        "empirical_fisher_response_weight_fit_count": 0,
        "breakdown": {
            "collection_native_source_forwards": 16,
            "collection_base_vjp_forwards": 16,
            "pair_endpoint_reconstruction_forwards": 12,
            "held_exact_arm_score_forwards": 20,
            "collection_base_vjp_backwards": 16,
            "pair_endpoint_reconstruction_backwards": 12,
            "pair_endpoint_local_head_contractions": 12,
        },
    }


def test_output_publication_is_0600_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke._v20b, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "report.json"
    payload = {
        "schema": smoke._SCHEMA,
        "format_version": smoke._FORMAT_VERSION,
        "artifact": output.as_posix(),
        "runner_protocol_sha256": smoke._RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": smoke._core.CONTINUOUS_RESPONSE_PROTOCOL_SHA256,
        "fresh_family_disjoint_claim_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "candidate": None,
        "provider_sidecar": None,
    }
    smoke._v20b._publish_scalar_fragment(
        payload,
        path=output,
        domain=smoke._REPORT_DOMAIN,
        hash_key="report_sha256",
        label="test V20c report",
    )
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="overwrite"):
        smoke._v20b._publish_scalar_fragment(
            payload,
            path=output,
            domain=smoke._REPORT_DOMAIN,
            hash_key="report_sha256",
            label="test V20c report",
        )
