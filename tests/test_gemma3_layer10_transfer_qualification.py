from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Callable

import pytest

import fisher_graph.gemma3_layer10_transfer_qualification as transfer


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate() -> dict[str, object]:
    return {
        "candidate_tensor_file": "candidate.pt",
        "candidate_tensor_file_sha256": _sha("candidate-file"),
        "candidate_scientific_payload_sha256": _sha("candidate-payload"),
        "compiler_pipeline_sha256": _sha("pipeline"),
        "interaction_promotion_sha256": _sha("promotion"),
        "dynamic_graph_sha256": _sha("dynamic"),
        "edgeless_graph_sha256": _sha("edgeless"),
        "model_id": "google/gemma-3-270m",
        "requested_revision": "a" * 40,
        "model_fingerprint": _sha("model"),
        "layer_ordinal": 10,
        "chosen_gain": 0.25,
        "node_count": 4,
        "interaction_count": 3,
    }


def _corpus() -> dict[str, object]:
    role_specs = {
        "calibration_a_fit": (
            256,
            8,
            "layer10_v8_transfer_shadow",
        ),
        "calibration_a_selection": (
            128,
            4,
            "layer10_v8_transfer_qualification",
        ),
        "calibration_a_guard": (128, 4, "layer10_v8_transfer_guard"),
    }
    return {
        "corpus_artifact_sha256": _sha("corpus"),
        "corpus_receipt_sha256": _sha("corpus-receipt"),
        "corpus_receipt_file_sha256": _sha("corpus-receipt-file"),
        "tokenizer_contract_sha256": _sha("tokenizer"),
        "roles": {
            role: {
                "manifest_sha256": _sha(f"{role}-manifest"),
                "role_input_file_sha256": _sha(f"{role}-file"),
                "example_count": examples,
                "family_count": families,
                "ordered_prompt_identity_sha256": transfer._domain_sha256(
                    b"fisher-graph:layer10-v8-role-prompt-identities:v1\0",
                    tuple(
                        _sha(f"{stream_label}-prompt-{index}")
                        for index in range(examples)
                    ),
                ),
            }
            for role, (examples, families, stream_label) in role_specs.items()
        },
    }


def _runtime() -> dict[str, object]:
    return {
        "device": "cpu",
        "dtype": "float32",
        "tokenization_batch_size": 1,
        "max_length": 128,
        "vocabulary_chunk_size": 16384,
        "deletion_equivalence_atol": 0.0,
        "deletion_equivalence_rtol": 0.0,
        "qualification_evaluator_file_sha256": _sha(
            "qualification-evaluator"
        ),
        "rung_evaluator_file_sha256": _sha("rung-evaluator"),
    }


def _behavior(
    *,
    native_nll: float = 0.0,
    candidate_nll: float = 0.05,
    edgeless_nll: float = 0.0495,
    deletion_nll: float = 0.16,
    candidate_kl: float = 0.05,
    candidate_top1: float = 0.80,
) -> dict[str, object]:
    def condition(
        nll: float,
        *,
        kl: float = 0.05,
        top1: float = 0.80,
    ) -> dict[str, float]:
        return {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": kl,
            "top1_agreement_to_native": top1,
        }

    return {
        "execution_path": "unified_modal_generator_graph_rung",
        "assessment_role": "open_development_assessment",
        "heldout_confirmation": False,
        "supervised_tokens": 32,
        "logical_valid_tokens": 34,
        "native": {"nll_per_token": native_nll},
        "conditions": {
            "interacting_graph": condition(
                candidate_nll,
                kl=candidate_kl,
                top1=candidate_top1,
            ),
            "edgeless_graph": condition(edgeless_nll),
            "matched_deletion": condition(deletion_nll),
        },
        "graph_comparison": {
            "node_count": 4,
            "interacting_edge_count": 3,
            "edgeless_edge_count": 0,
            "node_artifacts_identical": True,
            "deletion_paths_agree": True,
            "deletion_max_abs_logit_difference": 0.0,
        },
        "resource_accounting": {
            "interacting_graph": {
                "net_stored_parameter_savings": 509_245,
                "net_logical_macs_saved": 200_790_720,
            },
            "edgeless_graph": {
                "net_stored_parameter_savings": 513_664,
                "net_logical_macs_saved": 201_377_280,
            },
            "matched_deletion": {
                "net_stored_parameter_savings": 509_245,
                "net_logical_macs_saved": 250_099_200,
            },
        },
        "latency_or_kernel_speed_claim": False,
    }


def _stream(label: str, count: int) -> dict[str, object]:
    return {
        "schema": "fisher_graph.tokenized_calibration_stream",
        "format_version": 2,
        "split": label,
        "batches": count,
        "sequences": count,
        "serialized_sha256": _sha(f"{label}-stream"),
        "source_prompt_sha256": tuple(
            _sha(f"{label}-prompt-{index}") for index in range(count)
        ),
        "content_sha256": tuple(
            _sha(f"{label}-content-{index}") for index in range(count)
        ),
        "valid_tokens": {"minimum": 2, "maximum": 2, "total": 2 * count},
        "supervised_positions": {
            "minimum": 1,
            "maximum": 1,
            "total": count,
        },
        "contains_prompt_text": False,
        "contains_token_ids": False,
    }


def _panel(
    *,
    role: str,
    use: str,
    examples: int,
    families: int,
    behavior: dict[str, object] | None = None,
) -> dict[str, object]:
    evaluated = _behavior() if behavior is None else behavior
    return {
        "role": role,
        "use": use,
        "role_input_file_sha256": _sha(f"{role}-file"),
        "example_count": examples,
        "family_count": families,
        "tokenized_stream": _stream(use, examples),
        "behavior": evaluated,
        "qualification": transfer._panel_metrics(evaluated),
    }


def _qualification_report(
    protocol: dict[str, object],
    candidate: dict[str, object],
    corpus: dict[str, object],
) -> dict[str, object]:
    shadow = _panel(
        role="calibration_a_fit",
        use="layer10_v8_transfer_shadow",
        examples=256,
        families=8,
    )
    selection = _panel(
        role="calibration_a_selection",
        use="layer10_v8_transfer_qualification",
        examples=128,
        families=4,
    )
    payload: dict[str, object] = {
        "schema": transfer._QUALIFICATION_SCHEMA,
        "format_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "candidate_scientific_payload_sha256": candidate[
            "candidate_scientific_payload_sha256"
        ],
        "compiler_pipeline_sha256": candidate["compiler_pipeline_sha256"],
        "corpus_artifact_sha256": corpus["corpus_artifact_sha256"],
        "thresholds": dict(transfer._THRESHOLDS),
        "panels": {
            "transfer_shadow": shadow,
            "transfer_qualification": selection,
        },
        "tokenized_content_disjointness": (
            transfer._tokenized_panel_disjointness(
                {
                    "transfer_shadow": shadow,
                    "transfer_qualification": selection,
                }
            )
        ),
        "qualification_passed": True,
        "selection_opened": True,
        "candidate_changed": False,
        "candidate_tensor_file_sha256_after": candidate[
            "candidate_tensor_file_sha256"
        ],
        "guard_opened": False,
        "guard_consumed": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "safety": dict(transfer._SAFETY),
    }
    return {
        **payload,
        "qualification_sha256": transfer._domain_sha256(
            transfer._QUALIFICATION_DOMAIN,
            payload,
        ),
    }


def _rehash_qualification(report: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in report.items()
        if key != "qualification_sha256"
    }
    report["qualification_sha256"] = transfer._domain_sha256(
        transfer._QUALIFICATION_DOMAIN,
        payload,
    )


def test_protocol_hash_is_stable_reusable_and_runtime_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "protocol.json"
    candidate = _candidate()
    corpus = _corpus()
    runtime = _runtime()

    first = transfer._freeze_protocol(path, candidate, corpus, runtime)
    second = transfer._freeze_protocol(path, candidate, corpus, runtime)

    assert transfer._canonical_json_bytes(first) == transfer._canonical_json_bytes(
        second
    )
    payload = {
        key: value for key, value in first.items() if key != "protocol_sha256"
    }
    assert first["protocol_sha256"] == transfer._domain_sha256(
        transfer._PROTOCOL_DOMAIN,
        payload,
    )
    assert path.read_bytes() == transfer._canonical_json_bytes(first)

    changed_runtime = {**runtime, "dtype": "bfloat16"}
    with pytest.raises(ValueError, match="protocol differs"):
        transfer._freeze_protocol(path, candidate, corpus, changed_runtime)


def test_protocol_tamper_is_rejected_even_when_request_is_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "protocol.json"
    candidate = _candidate()
    corpus = _corpus()
    runtime = _runtime()
    protocol = transfer._freeze_protocol(path, candidate, corpus, runtime)
    tampered = copy.deepcopy(protocol)
    tampered["thresholds"][
        "maximum_delta_nll_per_token_to_native"
    ] = 999.0
    path.write_bytes(transfer._canonical_json_bytes(tampered))

    with pytest.raises(ValueError, match="hash mismatch"):
        transfer._freeze_protocol(path, candidate, corpus, runtime)


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    (
        (
            lambda behavior: behavior["conditions"]["interacting_graph"].update(
                {
                    "nll_per_token": 0.1001,
                    "delta_nll_per_token": 0.1001,
                }
            )
            or behavior["conditions"]["edgeless_graph"].update(
                {"nll_per_token": 0.1001}
            )
            or behavior["conditions"]["matched_deletion"].update(
                {"nll_per_token": 0.21}
            ),
            "native_nll_delta_passed",
        ),
        (
            lambda behavior: behavior["conditions"]["interacting_graph"].update(
                {"native_to_candidate_kl_per_token": 0.1201}
            ),
            "kl_passed",
        ),
        (
            lambda behavior: behavior["conditions"]["interacting_graph"].update(
                {"top1_agreement_to_native": 0.749}
            ),
            "top1_passed",
        ),
        (
            lambda behavior: behavior["conditions"]["edgeless_graph"].update(
                {"nll_per_token": 0.0489}
            ),
            "edgeless_regression_passed",
        ),
        (
            lambda behavior: behavior["conditions"]["matched_deletion"].update(
                {"nll_per_token": 0.1499}
            ),
            "deletion_improvement_passed",
        ),
        (
            lambda behavior: behavior["resource_accounting"][
                "interacting_graph"
            ].update({"net_stored_parameter_savings": 0}),
            "parameter_compression_passed",
        ),
        (
            lambda behavior: behavior["resource_accounting"][
                "interacting_graph"
            ].update({"net_logical_macs_saved": 0}),
            "logical_compute_compression_passed",
        ),
    ),
)
def test_panel_metric_thresholds_fail_independently(
    mutate: Callable[[dict[str, object]], object],
    failed_check: str,
) -> None:
    behavior = _behavior()
    assert transfer._panel_metrics(behavior)["passed"] is True

    mutate(behavior)
    result = transfer._panel_metrics(behavior)

    assert result["passed"] is False
    assert result["checks"][failed_check] is False
    assert sum(not value for value in result["checks"].values()) == 1


def test_tokenized_panel_disjointness_accepts_three_disjoint_panels() -> None:
    panels = {
        "fit": {"tokenized_stream": _stream("fit", 2)},
        "selection": {"tokenized_stream": _stream("selection", 2)},
        "guard": {"tokenized_stream": _stream("guard", 2)},
    }

    audit = transfer._tokenized_panel_disjointness(panels)

    assert audit["passed"] is True
    assert audit["overlap_count"] == 0


def test_tokenized_panel_disjointness_rejects_cross_panel_overlap() -> None:
    fit = _stream("fit", 2)
    selection = _stream("selection", 2)
    selection["content_sha256"] = (
        fit["content_sha256"][0],
        selection["content_sha256"][1],
    )

    with pytest.raises(ValueError, match="overlap"):
        transfer._tokenized_panel_disjointness(
            {
                "fit": {"tokenized_stream": fit},
                "selection": {"tokenized_stream": selection},
            }
        )


def test_tokenized_panel_disjointness_rejects_within_panel_duplicates() -> None:
    fit = _stream("fit", 2)
    fit["content_sha256"] = (fit["content_sha256"][0],) * 2

    with pytest.raises(ValueError, match="duplicat"):
        transfer._tokenized_panel_disjointness(
            {
                "fit": {"tokenized_stream": fit},
                "selection": {
                    "tokenized_stream": _stream("selection", 2)
                },
            }
        )


def test_strict_qualification_validation_accepts_exact_report(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    corpus = _corpus()
    protocol = transfer._freeze_protocol(
        tmp_path / "protocol.json",
        candidate,
        corpus,
        _runtime(),
    )
    report = _qualification_report(protocol, candidate, corpus)

    transfer._validate_qualification_report(
        report,
        protocol=protocol,
        candidate=candidate,
        corpus_binding=corpus,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "recomputed_panel_decision",
        "candidate_lineage",
        "cross_panel_content",
        "thresholds",
        "extra_source_field",
    ),
)
def test_strict_qualification_validation_rejects_semantic_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    candidate = _candidate()
    corpus = _corpus()
    protocol = transfer._freeze_protocol(
        tmp_path / "protocol.json",
        candidate,
        corpus,
        _runtime(),
    )
    report = _qualification_report(protocol, candidate, corpus)
    panels = report["panels"]

    if mutation == "recomputed_panel_decision":
        panels["transfer_shadow"]["qualification"]["passed"] = False
    elif mutation == "candidate_lineage":
        report["candidate_scientific_payload_sha256"] = _sha("foreign")
    elif mutation == "cross_panel_content":
        fit_stream = panels["transfer_shadow"]["tokenized_stream"]
        selection_stream = panels["transfer_qualification"][
            "tokenized_stream"
        ]
        selection_stream["content_sha256"] = (
            fit_stream["content_sha256"][0],
            *selection_stream["content_sha256"][1:],
        )
    elif mutation == "thresholds":
        report["thresholds"][
            "maximum_delta_nll_per_token_to_native"
        ] = 999.0
    elif mutation == "extra_source_field":
        report["prompt_text"] = "this must never be accepted"
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)
    _rehash_qualification(report)

    with pytest.raises((TypeError, ValueError), match=".+"):
        transfer._validate_qualification_report(
            report,
            protocol=protocol,
            candidate=candidate,
            corpus_binding=corpus,
        )
