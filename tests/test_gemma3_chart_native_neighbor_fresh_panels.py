from __future__ import annotations

import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "src/fisher_graph/gemma3_chart_native_neighbor_fresh_panels.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_chart_native_neighbor_fresh_panels_test_module",
    MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
panels = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = panels
_SPEC.loader.exec_module(panels)


TOKENIZER_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-3-270m/snapshots"
    / "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
)


def _stage_sha(index: int) -> str:
    return f"{index:064x}"


def test_module_is_standard_library_only_and_import_inert() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots == {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "pathlib",
        "re",
        "typing",
        "unicodedata",
    }


def test_frozen_manifest_receipt_and_exact_cost_surface() -> None:
    manifest = panels.load_manifest(ROOT / panels.DEFAULT_MANIFEST)
    receipt = panels.load_receipt(ROOT / panels.DEFAULT_RECEIPT)

    assert tuple(manifest["family_ids"]) == panels.FAMILY_IDS
    assert len(manifest["family_ids"]) == 6
    assert tuple(manifest["roles"]) == panels.ORDERED_ROLES
    assert manifest["row_axis_contract"] == panels.ROW_AXIS_CONTRACT
    assert receipt["row_axis_contract"] == panels.ROW_AXIS_CONTRACT
    assert manifest["stratum_contract"] == panels.STRATUM_CONTRACT
    assert receipt["stratum_contract"] == panels.STRATUM_CONTRACT
    assert manifest["required_freeze_stage_kind_by_role"] == (
        panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE
    )
    assert receipt["required_freeze_stage_kind_by_role"] == (
        panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE
    )
    assert panels.ROW_AXIS_CONTRACT["ordered_row_key_fields"] == [
        "role",
        "family_id",
        "example_id",
        "prediction_query_position",
        "target_label_position",
    ]
    assert panels.ROW_AXIS_CONTRACT[
        "carrier_chart_signature_key_fields"
    ] == ["example_id", "prediction_query_position"]
    for role in panels.EXECUTABLE_ROLES:
        role_manifest = manifest["roles"][role]
        assert role_manifest["example_count"] == 48
        assert role_manifest["examples_per_family"] == 8
        assert role_manifest["prompt_token_count"] == 24_780
        assert role_manifest["prediction_row_count"] == 24_732
        assert role_manifest["target_token_counts_per_family"] == [
            96,
            511,
            512,
            513,
            514,
            576,
            640,
            768,
        ]
        assert role_manifest["ordered_objective_row_catalog_sha256"] == (
            role_manifest["ordered_row_catalog_sha256"]
        )
        objective = role_manifest["score_objective_receipt"]
        assert objective["role"] == role
        assert objective["objective_row_count"] == 24_732
        assert objective["objective_row_key_fields"] == (
            panels.ROW_AXIS_CONTRACT["ordered_row_key_fields"]
        )
        assert objective["carrier_chart_signature_key_fields"] == [
            "example_id",
            "prediction_query_position",
        ]
        assert objective["ordered_objective_row_catalog_sha256"] == (
            role_manifest["ordered_objective_row_catalog_sha256"]
        )
        assert len(role_manifest["score_objective_receipt_sha256"]) == 64
        worker_receipt = panels.score_objective_receipt_for_role(
            role,
            manifest,
        )
        assert worker_receipt == {
            "role": role,
            "ordered_objective_row_catalog_sha256": role_manifest[
                "ordered_objective_row_catalog_sha256"
            ],
            "score_objective_receipt": objective,
            "score_objective_receipt_sha256": role_manifest[
                "score_objective_receipt_sha256"
            ],
        }
        for member in role_manifest["ordered_members"]:
            length = member["target_token_count"]
            assert member["prediction_row_count"] == length - 1
            assert member["prediction_query_position_start"] == 0
            assert (
                member["prediction_query_position_stop_exclusive"]
                == length - 1
            )
            assert member["target_label_position_start"] == 1
            assert member["target_label_position_stop_exclusive"] == length
            assert member["objective_row_stream_sha256"] == (
                member["row_stream_sha256"]
            )
    reserve = manifest["roles"][panels.RESERVE_ROLE]
    assert reserve["declared_example_count"] == 12
    assert reserve["declared_examples_per_family"] == 2
    assert reserve["tokenization_performed"] is False
    assert "ordered_members" not in reserve
    assert "prompt_token_count" not in reserve
    with pytest.raises(panels.FreshPanelFirewallError, match="no score"):
        panels.score_objective_receipt_for_role(
            panels.RESERVE_ROLE,
            manifest,
        )

    cost = receipt["cost_surface"]
    assert cost["all_executable_role_prompt_token_count"] == 99_120
    assert cost["all_executable_role_prediction_row_count"] == 98_928
    assert cost["A_transport_source_feature_width"] == 136
    assert cost["A_fit_row_to_feature_ratio"] == 24_732 / 136
    assert cost["A_fit_only"] is True
    assert cost["reserve_prompt_token_count"] is None
    assert cost["reserve_token_cost_claimed"] is False


def test_default_preflight_never_decodes_any_raw_role_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded_labels: list[str] = []
    original_decode = panels._decode_canonical_mapping

    def audited_decode(encoded: bytes, *, label: str):
        decoded_labels.append(label)
        return original_decode(encoded, label=label)

    monkeypatch.setattr(panels, "_decode_canonical_mapping", audited_decode)
    preflight = panels.opaque_preflight(repository_root=ROOT)

    assert preflight.reserve_semantically_decoded is False
    assert preflight.role_file_sha256s == panels.PINNED_ROLE_FILE_SHA256S
    assert decoded_labels == [
        "fresh prompt-free manifest",
        "fresh blind panel receipt",
    ]
    assert all("corpus" not in label for label in decoded_labels)


def test_sequential_firewall_refuses_early_reopen_and_reserve() -> None:
    source = panels.SequentialPanelSource.from_defaults(ROOT)

    with pytest.raises(panels.FreshPanelFirewallError, match="expected"):
        source.open_role("B_neighbor_discovery")
    with pytest.raises(panels.FreshPanelFirewallError, match="never"):
        source.open_reserve()

    opened = source.open_role("A_chart_transport_fit")
    assert opened["role"] == "A_chart_transport_fit"
    with pytest.raises(panels.FreshPanelFirewallError, match="must be frozen"):
        source.open_role("B_neighbor_discovery")
    assert source.required_freeze_stage_kind == (
        "chart_atlas_transport_fit"
    )
    with pytest.raises(panels.FreshPanelFirewallError, match="stage kind"):
        source.freeze_current("neighbor_candidate_catalog", _stage_sha(1))
    source.freeze_current(
        panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[
            "A_chart_transport_fit"
        ],
        _stage_sha(1),
    )
    with pytest.raises(panels.FreshPanelFirewallError, match="expected"):
        source.open_role("A_chart_transport_fit")

    for index, role in enumerate(panels.EXECUTABLE_ROLES[1:], start=2):
        assert source.next_role == role
        opened = source.open_role(role)
        assert opened["role"] == role
        assert source.required_freeze_stage_kind == (
            panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[role]
        )
        source.freeze_current(
            panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[role],
            _stage_sha(index),
        )
    assert source.next_role is None
    assert tuple(source.freeze_receipts) == panels.EXECUTABLE_ROLES
    assert source.freeze_receipts["B_neighbor_discovery"] == {
        "stage_kind": "neighbor_candidate_catalog",
        "stage_receipt_sha256": _stage_sha(2),
    }
    with pytest.raises(panels.FreshPanelFirewallError):
        source.open_role(panels.RESERVE_ROLE)


def test_reserve_refusal_precedes_file_or_tokenizer_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = panels.SequentialPanelSource.from_defaults(ROOT)
    reads: list[object] = []

    def forbidden_read(*args, **kwargs):
        reads.append((args, kwargs))
        raise AssertionError("reserve refusal must happen before a file read")

    monkeypatch.setattr(panels, "_read_bytes", forbidden_read)
    with pytest.raises(panels.FreshPanelFirewallError, match="no semantic-open"):
        source.open_role(panels.RESERVE_ROLE)
    assert reads == []

    class NeverTokenizer:
        def encode(self, *_args, **_kwargs):
            raise AssertionError("reserve refusal must happen before tokenization")

    with pytest.raises(panels.FreshPanelFirewallError, match="never reach"):
        panels.authenticate_executable_tokenization(
            role=panels.RESERVE_ROLE,
            corpus={},
            manifest=source.manifest,
            tokenizer=NeverTokenizer(),
        )


def test_identity_firewalls_and_role_specific_hard_suffixes() -> None:
    manifest = panels.load_manifest(ROOT / panels.DEFAULT_MANIFEST)
    audit = manifest["overlap_audit"]
    assert audit["all_intersection_counts_zero"] is True
    assert set(audit["role_pair_intersection_counts"].values()) == {0}
    assert set(audit["prior_intersection_counts"].values()) == {0}

    suffix_catalog = manifest["hard_long_suffix_contract"][
        "role_family_suffix_sha256s"
    ]
    flattened = [
        suffix_catalog[role][family]
        for role in panels.EXECUTABLE_ROLES
        for family in panels.FAMILY_IDS
    ]
    assert len(flattened) == 24
    assert len(set(flattened)) == 24
    for role in panels.EXECUTABLE_ROLES:
        members = manifest["roles"][role]["ordered_members"]
        for family in panels.FAMILY_IDS:
            hard = [
                member
                for member in members
                if member["family_id"] == family
                and member["target_token_count"] >= 576
            ]
            assert len(hard) == 3
            assert len({row["neutral_suffix_sha256"] for row in hard}) == 1
            assert len(
                {row["neutral_suffix_template_sha256"] for row in hard}
            ) == 1
            for row in hard:
                target = row["target_token_count"]
                assert row["evidence_token_span"][1] <= target - 512
                assert row["neutral_suffix_token_span"] == [
                    target - 512,
                    target,
                ]


def test_stratum_eligibility_and_transition_514_are_exact() -> None:
    manifest = panels.load_manifest(ROOT / panels.DEFAULT_MANIFEST)
    receipt = panels.load_receipt(ROOT / panels.DEFAULT_RECEIPT)

    assert panels.STRATA == (
        "short_96",
        "boundary_511",
        "boundary_512",
        "boundary_513",
        "boundary_transition_514",
        "hard_long_576",
        "hard_long_640",
        "hard_long_768",
    )
    assert panels.STRATUM_CONTRACT[
        "boundary_transition_514_is_diagnostic_only"
    ] is True
    assert panels.STRATUM_CONTRACT["B_C_pooled_eligible_strata"] == [
        "short_96",
        "boundary_511",
        "boundary_512",
        "boundary_513",
        "hard_long_576",
        "hard_long_640",
        "hard_long_768",
    ]
    for role in panels.EXECUTABLE_ROLES:
        members = manifest["roles"][role]["ordered_members"]
        for member in members:
            target = member["target_token_count"]
            assert member["boundary_transition_diagnostic_only"] is (
                target == 514
            )
            assert member["diagnostic_reporting_required"] is (
                target in (96, 511, 512, 513, 514)
            )
            if role == "A_chart_transport_fit":
                assert member["eligible_for_A_fit"] is True
            elif role in (
                "B_neighbor_discovery",
                "C_neighbor_confirmation",
            ):
                pooled = target != 514
                assert member["eligible_for_pooled_point_topology"] is pooled
                assert member["eligible_for_bootstrap"] is pooled
                assert member["eligible_for_LOFO"] is pooled
                assert member["may_change_chart_neighbor_graph"] is pooled
                assert member["eligible_for_hard_long_veto"] is (
                    target >= 576
                )
            else:
                assert member["eligible_for_D_response_reporting"] is True
                assert member["may_change_chart_neighbor_graph"] is False

    for role in ("B_neighbor_discovery", "C_neighbor_confirmation"):
        role_manifest = manifest["roles"][role]
        assert role_manifest["eligibility_example_counts"][
            "eligible_for_pooled_point_topology"
        ] == 42
        assert role_manifest["eligibility_prediction_row_counts"][
            "eligible_for_pooled_point_topology"
        ] == 21_654
        assert role_manifest["eligibility_example_counts"][
            "eligible_for_hard_long_veto"
        ] == 18
    cost = receipt["cost_surface"]
    assert cost["per_B_C_bootstrap_draws_per_family"] == 7
    assert cost["per_B_C_LOFO_eligible_examples_per_family"] == 7
    assert cost["per_B_C_boundary_transition_514_example_count"] == 6
    assert cost["boundary_transition_514_eligible_for_confirmation"] is False
    assert cost["D_graph_changing_example_count"] == 0


def test_hard_suffix_neutrality_audit_and_D_family_6_remediation() -> None:
    manifest = panels.load_manifest(ROOT / panels.DEFAULT_MANIFEST)
    audit = manifest["hard_long_neutrality_audit"]
    assert audit["audit_sha256"] == panels.PINNED_NEUTRALITY_AUDIT_SHA256
    assert audit["group_count"] == 24
    assert audit["all_overlap_counts_zero"] is True
    assert len(audit["groups"]) == 24
    overlap_fields = (
        "raw_choice_text_overlap_count",
        "choice_token_subsequence_overlap_count",
        "response_choice_label_token_overlap_count",
        "raw_semantic_cue_overlap_count",
        "semantic_cue_token_subsequence_overlap_count",
    )
    assert all(
        all(group[field_name] == 0 for field_name in overlap_fields)
        for group in audit["groups"].values()
    )

    source = panels.SequentialPanelSource.from_defaults(ROOT)
    d_corpus = None
    for index, role in enumerate(panels.EXECUTABLE_ROLES, start=1):
        opened = source.open_role(role)
        if role == "D_response_validation":
            d_corpus = opened
        source.freeze_current(
            panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[role],
            _stage_sha(index),
        )
    assert d_corpus is not None
    all_hard = [
        example
        for example in d_corpus["examples"]
        if example["target_token_count"] >= 576
    ]
    suffixes_by_family: dict[str, list[str]] = {
        family: [] for family in panels.FAMILY_IDS
    }
    for example in all_hard:
        family = example["family_id"]
        start, stop = example["neutral_suffix_char_span"]
        suffix = example["prompt"][start:stop]
        suffixes_by_family[family].append(suffix)
        assert all(
            not panels._whole_phrase_present(suffix, choice_text)
            for choice_text in example["choice_texts"]
        )
        assert all(
            not panels._whole_phrase_present(suffix, cue)
            for cue in panels.SEMANTIC_CUES_BY_FAMILY[family]
        )
    assert len(all_hard) == 18
    assert all(
        len(suffixes) == 3 and len(set(suffixes)) == 1
        for suffixes in suffixes_by_family.values()
    )
    remediated = suffixes_by_family[
        "fresh-v1/counterfactual-token-exchange"
    ][0]
    assert "yellow" not in remediated.casefold()


def test_prompt_text_is_confined_to_sequential_raw_corpus() -> None:
    source = panels.SequentialPanelSource.from_defaults(ROOT)
    corpus = source.open_role("A_chart_transport_fit")
    prompt = corpus["examples"][0]["prompt"]
    manifest_bytes = (ROOT / panels.DEFAULT_MANIFEST).read_bytes()
    receipt_bytes = (ROOT / panels.DEFAULT_RECEIPT).read_bytes()

    assert prompt.encode("utf-8") not in manifest_bytes
    assert prompt.encode("utf-8") not in receipt_bytes
    assert b'"prompt":' not in manifest_bytes
    assert b'"token_ids":' not in manifest_bytes
    assert b'"prompt":' not in receipt_bytes
    assert b'"token_ids":' not in receipt_bytes
    assert source.manifest["future_report_boundary"] == {
        "raw_prompt_text_allowed": False,
        "raw_choice_text_allowed": False,
        "raw_token_ids_allowed": False,
        "hashes_counts_spans_and_receipts_only": True,
    }


def test_D_has_four_distinct_single_token_choices_and_balanced_key() -> None:
    manifest = panels.load_manifest(ROOT / panels.DEFAULT_MANIFEST)
    members = manifest["roles"]["D_response_validation"][
        "ordered_members"
    ]
    assert Counter(member["correct_choice_index"] for member in members) == {
        0: 12,
        1: 12,
        2: 12,
        3: 12,
    }
    assert all(member["choice_count"] == 4 for member in members)
    assert all(member["choice_text_count"] == 4 for member in members)
    assert all(
        member["bare_choice_token_counts"] == [1, 1, 1, 1]
        and member["append_choice_token_counts"] == [1, 1, 1, 1]
        and member["distinct_bare_choice_token_id_count"] == 4
        and member["distinct_append_choice_token_id_count"] == 4
        for member in members
    )


def test_blind_author_and_source_inventory_are_frozen() -> None:
    receipt = panels.load_receipt(ROOT / panels.DEFAULT_RECEIPT)
    access = receipt["authoring_access_receipt"]
    assert all(access["opaque_byte_hash_access"].values())
    assert access["semantic_parse_access"] == {
        role: role in panels.EXECUTABLE_ROLES
        for role in panels.ORDERED_ROLES
    }
    assert access["tokenizer_access"] == {
        role: role in panels.EXECUTABLE_ROLES
        for role in panels.ORDERED_ROLES
    }
    assert not any(access["model_access"].values())
    assert access[
        "opaque_hash_access_recorded_separately_from_semantic_parse_access"
    ] is True
    declaration = receipt["blind_author_declaration"]
    assert declaration["local_runs_opened"] is False
    assert declaration["result_report_opened"] is False
    assert declaration["chart_report_opened"] is False
    assert declaration["atlas_report_opened"] is False
    assert declaration["neighbor_report_opened"] is False
    assert declaration["candidate_report_opened"] is False
    assert declaration["distance_report_opened"] is False
    assert declaration["model_weights_opened_or_loaded"] is False
    assert declaration["model_execution_performed"] is False
    assert declaration["tokenizer_only_execution_performed"] is True
    assert declaration["repository_dependency_config_only_read"] is True
    assert declaration["semantic_nonreuse_review_completed"] is True

    inventory = receipt["source_inventory"]
    assert inventory["entry_count"] == len(inventory["entries"])
    assert all(
        ".local-runs" not in entry["path"]
        for entry in inventory["entries"]
    )
    access_modes = {entry["access_mode"] for entry in inventory["entries"]}
    assert "tokenizer_only_execution" in access_modes
    assert "semantic_model_config_only_read" in access_modes
    assert "semantic_overlap_review" in access_modes


def test_pinned_tokenizer_replays_all_executable_receipts() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer_path = TOKENIZER_ROOT / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.skip("pinned Gemma tokenizer cache is unavailable")

    observed_assets = panels.authenticate_tokenizer_assets(TOKENIZER_ROOT)
    assert observed_assets["tokenizer.json"] == (
        panels.PINNED_TOKENIZER_JSON_SHA256
    )
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    source = panels.SequentialPanelSource.from_defaults(ROOT)
    receipts = []
    for index, role in enumerate(panels.EXECUTABLE_ROLES, start=1):
        corpus = source.open_role(role)
        replay = panels.authenticate_executable_tokenization(
            role=role,
            corpus=corpus,
            manifest=source.manifest,
            tokenizer=tokenizer,
        )
        receipts.append(replay)
        source.freeze_current(
            panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[role],
            _stage_sha(index),
        )

    assert [row["role"] for row in receipts] == list(
        panels.EXECUTABLE_ROLES
    )
    assert sum(row["prompt_token_count"] for row in receipts) == 99_120
    assert sum(row["prediction_row_count"] for row in receipts) == 98_928
    assert all(row["raw_prompt_text_present"] is False for row in receipts)
    assert all(row["raw_token_ids_present"] is False for row in receipts)
    assert all(
        row["carrier_chart_signature_key_fields"]
        == ["example_id", "prediction_query_position"]
        for row in receipts
    )
    assert all(
        len(row["ordered_objective_row_catalog_sha256"]) == 64
        and len(row["score_objective_receipt_sha256"]) == 64
        for row in receipts
    )
    assert all(
        row["hard_long_neutrality_groups_authenticated"] == 6
        and row["hard_long_neutrality_audit_sha256"]
        == panels.PINNED_NEUTRALITY_AUDIT_SHA256
        for row in receipts
    )


def test_tampered_open_role_fails_hash_binding() -> None:
    source = panels.SequentialPanelSource.from_defaults(ROOT)
    corpus = source.open_role("A_chart_transport_fit")
    tampered = json.loads(panels.canonical_json_bytes(corpus))
    tampered["examples"][0]["prompt"] += " changed"

    with pytest.raises(panels.FreshPanelIntegrityError, match="differs"):
        panels.validate_open_role_corpus(
            tampered,
            role="A_chart_transport_fit",
            manifest=source.manifest,
        )
