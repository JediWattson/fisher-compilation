from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_stable_mutual_knn_confirmation_experiment as subject
from fisher_graph.gemma3_stable_mutual_knn_confirmation_panel import PromptRecord
from fisher_graph.gemma3_stable_mutual_knn_setup import (
    FRESH_ORDERED_FAMILY_IDS,
    build_gemma3_stable_mutual_knn_setup,
)
from fisher_graph.gemma3_typed_mode_response_setup import (
    LEXICAL_MODE_IDS,
    RESPONSE_MODE_IDS,
)


STRATA = (
    ("short_32", 32),
    ("short_64", 64),
    ("short_96", 96),
    ("boundary_511", 511),
    ("boundary_512", 512),
    ("transition_513", 513),
    ("transition_514", 514),
    ("outside_window_576", 576),
    ("outside_window_640", 640),
    ("outside_window_768", 768),
)


def _records() -> tuple[PromptRecord, ...]:
    rows: list[PromptRecord] = []
    for stratum_index, (stratum_id, target) in enumerate(STRATA):
        for family_index, family_id in enumerate(FRESH_ORDERED_FAMILY_IDS):
            semantic_family = family_id.removeprefix("C_confirmation.")
            prompt_id = f"c.{stratum_id}.{semantic_family}"
            token_ids = tuple(range(target))
            rows.append(
                PromptRecord(
                    role="C_confirmation",
                    corpus_index=len(rows),
                    semantic_family=semantic_family,
                    prompt_id=prompt_id,
                    template_id=f"template.{stratum_index}.{family_index}",
                    instance_id=f"instance.{stratum_index}.{family_index}",
                    domain_template=f"domain.{stratum_index}.{family_index}",
                    kind=(
                        "beyond_window_probe"
                        if stratum_id.startswith("outside_window")
                        else "short_control"
                    ),
                    stratum_id=stratum_id,
                    text=f"prompt text {prompt_id}",
                    text_sha256=hashlib.sha256(prompt_id.encode()).hexdigest(),
                    canonical_sha256=hashlib.sha256(
                        f"canonical:{prompt_id}".encode()
                    ).hexdigest(),
                    semantic_lineage_id=f"semantic.{stratum_index}.{family_index}",
                    paraphrase_lineage_id=(
                        f"paraphrase.{stratum_index}.{family_index}"
                    ),
                    template_signature_sha256=hashlib.sha256(
                        f"template:{prompt_id}".encode()
                    ).hexdigest(),
                    target_valid_tokens=target,
                    input_ids_sha256=subject._token_stream_sha256(token_ids),
                    choices=(" alpha", " beta", " gamma", " delta"),
                    correct_choice=family_index % 4,
                    supports_correct_choice_index=family_index % 4,
                    choices_sha256=hashlib.sha256(
                        f"choices:{prompt_id}".encode()
                    ).hexdigest(),
                    choice_token_ids=(101, 102, 103, 104),
                    answer_rationale=f"rationale for {prompt_id}",
                    answer_rationale_sha256=hashlib.sha256(
                        f"rationale for {prompt_id}".encode()
                    ).hexdigest(),
                    answer_evidence=(
                        {"authenticated_by_panel_helper": True}
                        if stratum_id.startswith("outside_window")
                        else None
                    ),
                )
            )
    return tuple(rows)


def _measurement(records: tuple[PromptRecord, ...]) -> subject._ResponseMeasurement:
    example_ids = tuple(record.prompt_id for record in records)
    family_ids = tuple(
        f"C_confirmation.{record.semantic_family}" for record in records
    )
    strata = tuple(str(record.stratum_id) for record in records)
    responses: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
    lexical_rows = tuple(
        (
            float(index % 7) / 13.0,
            -float(index % 5) / 11.0,
            float(index % 3) / 17.0,
            float(index % 2) / 19.0,
        )
        for index in range(len(records))
    )
    for mode_index, mode_id in enumerate(RESPONSE_MODE_IDS):
        if mode_id in LEXICAL_MODE_IDS:
            responses[mode_id] = lexical_rows
        else:
            responses[mode_id] = tuple(
                (
                    row[0] + mode_index / 101.0,
                    row[1] - mode_index / 113.0,
                    row[2] + mode_index / 127.0,
                    row[3] - mode_index / 131.0,
                )
                for row in lexical_rows
            )
    return subject._ResponseMeasurement(
        example_ids=example_ids,
        family_ids_by_example=family_ids,
        stratum_ids_by_example=strata,
        responses_by_mode=responses,
        weights_by_example=tuple((0.2, 0.3, 0.4, 0.5) for _ in records),
        state_fingerprints_by_mode={
            mode_id: hashlib.sha256(mode_id.encode()).hexdigest()
            for mode_id in RESPONSE_MODE_IDS
        },
        terminal_norm_state_fingerprint_sha256="a" * 64,
        receipt={"measurement_input_sha256": "b" * 64},
    )


def test_run_fails_before_live_dependencies_when_static_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_live_dependencies = False

    def fail_preflight(**_kwargs: object) -> object:
        raise ValueError("static failure")

    def forbidden_live_dependencies() -> object:
        nonlocal opened_live_dependencies
        opened_live_dependencies = True
        raise AssertionError("live dependencies must remain unopened")

    monkeypatch.setattr(subject, "_preflight_static", fail_preflight)
    monkeypatch.setattr(
        subject, "_load_live_dependencies", forbidden_live_dependencies
    )
    with pytest.raises(ValueError, match="static failure"):
        subject.run_gemma3_stable_mutual_knn_confirmation_experiment()
    assert opened_live_dependencies is False


def test_public_runner_preserves_preflight_to_atomic_write_phase_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases: list[str] = []
    preflight = object()
    dependencies = object()
    tokenization = object()
    runtime = object()
    raw_measurement = object()
    bound_measurement = object()
    graph = object()
    report = {"report": True}
    loaded = {"loaded": True}

    def phase(name: str, result: object):
        def invoke(*_args: object, **_kwargs: object) -> object:
            phases.append(name)
            return result

        return invoke

    monkeypatch.setattr(subject, "_preflight_static", phase("preflight", preflight))
    monkeypatch.setattr(
        subject, "_load_live_dependencies", phase("dependencies", dependencies)
    )
    monkeypatch.setattr(
        subject,
        "_load_and_authenticate_tokenization",
        phase("tokenization", tokenization),
    )
    monkeypatch.setattr(
        subject,
        "_prepare_runtime_after_tokenization",
        phase("runtime", runtime),
    )
    monkeypatch.setattr(
        subject, "_measure_responses", phase("measurement", raw_measurement)
    )
    monkeypatch.setattr(
        subject,
        "_bind_fresh_pair_statistics",
        phase("statistics_binding", bound_measurement),
    )
    monkeypatch.setattr(
        subject, "_build_graph_receipt", phase("graph", graph)
    )
    monkeypatch.setattr(subject, "_assemble_report", phase("report", report))
    monkeypatch.setattr(
        subject,
        "write_gemma3_stable_mutual_knn_confirmation_report",
        phase("atomic_write_and_strict_load", loaded),
    )
    assert subject.run_gemma3_stable_mutual_knn_confirmation_experiment() == loaded
    assert phases == [
        "preflight",
        "dependencies",
        "tokenization",
        "runtime",
        "measurement",
        "statistics_binding",
        "graph",
        "report",
        "atomic_write_and_strict_load",
    ]


def test_default_relative_inputs_complete_real_static_preflight(
    tmp_path: Path,
) -> None:
    output = tmp_path / "default-static-preflight.json"
    result = subject._preflight_static(
        revision=subject.PINNED_MODEL_REVISION,
        model_id=subject.PINNED_MODEL_ID,
        output=output,
        setup_path=subject.DEFAULT_SETUP,
        predecessor_path=subject.PINNED_TYPED_RESPONSE_REPORT_PATH,
        corpus_path=subject.DEFAULT_CORPUS,
        panel_path=subject.DEFAULT_PANEL,
        panel_audit_path=subject.DEFAULT_PANEL_AUDIT,
        repository_root=subject.DEFAULT_REPOSITORY_ROOT,
        device_name="cpu",
        dtype="float32",
        batch_size=subject.PINNED_BATCH_SIZE,
    )
    assert len(result.confirmation_records) == 60
    assert result.corpus_path.is_absolute()
    assert result.panel_path.is_absolute()
    assert result.panel_audit_path.is_absolute()
    assert result.receipts["confirmation_worker_source_path"] == str(
        Path(subject.__file__).resolve()
    )
    assert result.receipts["confirmation_worker_source_sha256"] == (
        subject.PINNED_CONFIRMATION_WORKER_SOURCE_SHA256
    )
    assert result.receipts["discovery_stats_binding_sha256"] == (
        subject._PINNED_DISCOVERY_STATS_BINDING_SHA256
    )
    assert not output.exists()
    assert not result.svg_path.exists()


def test_confirmation_records_enforce_frozen_schedule_and_one_space() -> None:
    records = _records()
    assert subject._validate_confirmation_records(records) == records
    broken = list(records)
    row = broken[0]
    broken[0] = PromptRecord(
        **{
            field: getattr(row, field)
            for field in row.__dataclass_fields__
            if field != "choices"
        },
        choices=("  alpha", " beta", " gamma", " delta"),
    )
    with pytest.raises(ValueError, match="exactly one leading space"):
        subject._validate_confirmation_records(broken)


def test_fresh_pair_statistics_are_complete_and_lexical_exact() -> None:
    measurement = _measurement(_records())
    family_rows, example_rows = subject._build_fresh_pair_statistics(measurement)
    assert len(family_rows) == subject.PINNED_FAMILY_PAIR_STAT_COUNT
    assert len(example_rows) == subject.PINNED_EXAMPLE_PAIR_STAT_COUNT
    lexical = [
        row
        for row in example_rows
        if {row.mode_a_id, row.mode_b_id} == set(LEXICAL_MODE_IDS)
    ]
    assert len(lexical) == 60
    assert all(row.weighted_sse == 0.0 for row in lexical)
    assert all(
        row.weighted_mode_a_energy
        == row.weighted_dot
        == row.weighted_mode_b_energy
        for row in lexical
    )


def test_stats_catalog_digest_is_invariant_to_unordered_pair_orientation() -> None:
    row = subject.DownstreamResponsePairStats(
        family_id="family",
        mode_a_id=RESPONSE_MODE_IDS[0],
        mode_b_id=RESPONSE_MODE_IDS[1],
        response_abi_sha256=subject.PINNED_RESPONSE_ABI_SHA256,
        weighted_sse=5.0,
        weighted_mode_a_energy=2.0,
        weighted_dot=0.0,
        weighted_mode_b_energy=3.0,
        row_count=4,
    )
    reversed_row = subject.DownstreamResponsePairStats(
        family_id=row.family_id,
        mode_a_id=row.mode_b_id,
        mode_b_id=row.mode_a_id,
        response_abi_sha256=row.response_abi_sha256,
        weighted_sse=row.weighted_sse,
        weighted_mode_a_energy=row.weighted_mode_b_energy,
        weighted_dot=row.weighted_dot,
        weighted_mode_b_energy=row.weighted_mode_a_energy,
        row_count=row.row_count,
    )
    assert subject._family_stats_catalog_sha256((row,)) == (
        subject._family_stats_catalog_sha256((reversed_row,))
    )


def test_setup_adapts_to_exact_core_node_and_candidate_catalog() -> None:
    setup = build_gemma3_stable_mutual_knn_setup()
    nodes, candidates, digest = subject._node_specs_and_candidates(setup)
    assert len(nodes) == 37
    assert sum(len(node.member_mode_ids) for node in nodes) == 38
    assert len(candidates) == 14
    assert subject.stable_mutual_knn_candidate_catalog_sha256(
        nodes, candidates
    ) == digest


def test_graph_config_freezes_pooled_diagnostics_and_long_veto() -> None:
    config = subject._graph_config()
    assert config.k == 2
    assert config.primary_stratum_id == "pooled"
    assert config.diagnostic_stratum_ids == ("short", "boundary")
    assert config.hard_veto_stratum_ids == ("long",)
    assert config.bootstrap_resample_count == 10_000
    assert config.bootstrap_min_mutual_frequency == 0.80
    assert config.lofo_min_mutual_count == 5
    assert config.max_directional_relative_sse == 1.0
    assert config.max_absolute_weighted_rmse == 0.125


def test_setup_explicitly_pins_fresh_bootstrap_draw_protocol() -> None:
    setup = build_gemma3_stable_mutual_knn_setup()
    subject._assert_setup_bootstrap_protocol(setup)
    broken = dict(setup)
    gates = dict(broken["fresh_confirmation_gates"])
    bootstrap = dict(gates["within_family_bootstrap"])
    bootstrap["draw_protocol_domain_hex"] = "00"
    gates["within_family_bootstrap"] = bootstrap
    broken["fresh_confirmation_gates"] = gates
    with pytest.raises(RuntimeError, match="draw protocol drifted"):
        subject._assert_setup_bootstrap_protocol(broken)

    preflight = SimpleNamespace(confirmation_records=_records())
    config = subject._fresh_bootstrap_config(preflight)
    assert config.draw_protocol_domain == subject.BOOTSTRAP_DRAW_PROTOCOL_DOMAIN


def test_setup_exactly_pins_confirmation_worker_source() -> None:
    setup = build_gemma3_stable_mutual_knn_setup()
    contract = subject._confirmation_worker_source_contract(setup)
    assert contract == {
        "path": str(subject.PINNED_CONFIRMATION_WORKER_SOURCE_PATH),
        "sha256": subject.PINNED_CONFIRMATION_WORKER_SOURCE_SHA256,
    }
    broken = copy.deepcopy(setup)
    broken["confirmation_worker_source"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source setup contract drifted"):
        subject._confirmation_worker_source_contract(broken)


def test_tokenization_uses_twenty_exact_length_batches_without_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = _records()
    target_by_text = {record.text: record.target_valid_tokens for record in records}
    tokenized_texts: list[str] = []

    class FakeTokenizer:
        name_or_path = subject.PINNED_MODEL_ID
        bos_token_id = 2
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, value: object, **kwargs: object) -> object:
            if isinstance(value, str):
                tokenized_texts.append(value)
                choice_ids = {
                    " alpha": 101,
                    " beta": 102,
                    " gamma": 103,
                    " delta": 104,
                }
                return {
                    "input_ids": [choice_ids[value]]
                }
            assert isinstance(value, list)
            tokenized_texts.extend(value)
            lengths = [int(target_by_text[text]) for text in value]
            assert len(set(lengths)) == 1
            length = lengths[0]
            ids = torch.tensor([list(range(length)) for _ in value])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class FakeTokenizerType:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            return FakeTokenizer()

    suffix_by_family = {
        family.removeprefix("C_confirmation."): hashlib.sha256(
            f"suffix:{family}".encode()
        ).hexdigest()
        for family in FRESH_ORDERED_FAMILY_IDS
    }
    panel_rows = []
    for record in records:
        is_long = str(record.stratum_id) in subject._LONG_STRATA
        panel_rows.append(
            {
                "prompt_id": record.prompt_id,
                "stratum_id": record.stratum_id,
                "valid_tokens": record.target_valid_tokens,
                "input_ids_sha256": record.input_ids_sha256,
                "choice_token_ids_sha256": (
                    subject._panel_module._choice_token_ids_sha256(
                        record.choice_token_ids
                    )
                ),
                "choices_sha256": record.choices_sha256,
                "answer_evidence_verified": is_long,
                "answer_evidence_token_count": 2 if is_long else 0,
                "visible_suffix_token_ids_sha256": (
                    suffix_by_family[record.semantic_family]
                    if is_long
                    else None
                ),
                "visible_suffix_leak_free": True,
            }
        )
    monkeypatch.setattr(subject, "load_panel", lambda *_args: object())
    fake_panel_receipt = {
        "tokenizer_id": subject.PINNED_MODEL_ID,
        "tokenizer_revision": subject.PINNED_MODEL_REVISION,
        "row_count": 60,
        "reserve_tokenized": False,
        "visible_suffix_digest_domain": (
            subject._panel_module._VISIBLE_SUFFIX_TOKEN_IDS_DOMAIN.rstrip(
                b"\0"
            ).decode("ascii")
        ),
        "hard_long_visible_suffixes": [
            {
                "semantic_family": family,
                "row_count": 3,
                "visible_suffix_token_ids_sha256": suffix_by_family[family],
            }
            for family in subject._panel_module.SEMANTIC_FAMILIES
        ],
        "rows": panel_rows,
    }
    monkeypatch.setattr(
        subject,
        "authenticate_confirmation_tokenization",
        lambda _panel, _tokenizer: fake_panel_receipt,
    )
    monkeypatch.setattr(
        subject,
        "_PINNED_PANEL_TOKENIZATION_CANONICAL_SHA256",
        hashlib.sha256(
            subject._canonical_json_bytes(fake_panel_receipt)
        ).hexdigest(),
    )
    monkeypatch.setattr(
        subject,
        "_PINNED_PANEL_TOKENIZATION_DOMAIN_SHA256",
        subject._manifest_sha256(
            subject._TOKENIZATION_BINDING_DOMAIN, fake_panel_receipt
        ),
    )
    setup = build_gemma3_stable_mutual_knn_setup()
    preflight = subject._StaticPreflight(
        setup=setup,
        predecessor={},
        confirmation_records=records,
        setup_path=tmp_path / "setup.json",
        predecessor_path=tmp_path / "predecessor.json",
        corpus_path=tmp_path / "corpus.json",
        panel_path=tmp_path / "panel.json",
        panel_audit_path=tmp_path / "audit.json",
        output_path=tmp_path / "output.json",
        svg_path=tmp_path / "output-graph.svg",
        receipts={"panel_audit_payload_sha256": "c" * 64},
    )
    dependencies = SimpleNamespace(
        torch=torch,
        tokenizer_type=FakeTokenizerType,
        resolve_paths=lambda _cache: {"hub_cache": tmp_path},
    )
    result = subject._load_and_authenticate_tokenization(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=None,
        batch_size=4,
    )
    assert len(result.batches) == 20
    assert result.receipt["batch_row_count_catalog"] == [4, 2] * 10
    assert result.receipt["no_cross_length_padding"] is True
    assert result.receipt["extra_leading_space_prepended"] is False
    assert result.receipt["reserve_rows_tokenized"] == 0
    assert len(
        [
            row
            for row in result.receipt["rows"]
            if row["answer_evidence_token_count"] > 1
            and row["visible_suffix_token_ids_sha256"] is not None
        ]
    ) == 18
    assert len(
        result.receipt["panel_tokenization_receipt"][
            "hard_long_visible_suffixes"
        ]
    ) == 6
    assert all("reserve" not in text for text in tokenized_texts)
    # Sixty prompts plus four choices per row; the panel-owned helper was
    # replaced above and therefore contributes no hidden tokenizer calls.
    assert len(tokenized_texts) == 60 + 4 * 60

    coordinated_non_long_claim = copy.deepcopy(result.receipt)
    coordinated_non_long_claim["rows"][0][
        "answer_evidence_outside_window_authenticated"
    ] = True
    coordinated_non_long_claim["rows"][0]["answer_evidence_token_count"] = 2
    coordinated_non_long_claim["rows"][0][
        "visible_suffix_token_ids_sha256"
    ] = "f" * 64
    coordinated_non_long_claim["panel_tokenization_receipt"]["rows"][0][
        "answer_evidence_verified"
    ] = True
    coordinated_non_long_claim["panel_tokenization_receipt"]["rows"][0][
        "answer_evidence_token_count"
    ] = 2
    coordinated_non_long_claim["panel_tokenization_receipt"]["rows"][0][
        "visible_suffix_token_ids_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="non-long row"):
        subject._validate_panel_tokenization_suffix_receipt(
            coordinated_non_long_claim["panel_tokenization_receipt"],
            worker_rows=coordinated_non_long_claim["rows"],
        )

    mismatched_family_catalog = copy.deepcopy(result.receipt)
    mismatched_family_catalog["panel_tokenization_receipt"][
        "hard_long_visible_suffixes"
    ][0]["visible_suffix_token_ids_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="family proof"):
        subject._validate_panel_tokenization_suffix_receipt(
            mismatched_family_catalog["panel_tokenization_receipt"],
            worker_rows=mismatched_family_catalog["rows"],
        )


def test_resource_envelope_is_grounded_in_measured_core_shape() -> None:
    assert subject._MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES == 33_183_756
    assert subject._MAX_REPORT_BYTES == 128 * 1024 * 1024
    assert subject._MIN_FREE_BYTES_BEFORE_LIVE == 256 * 1024 * 1024
    assert subject._MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES < (
        subject._MAX_REPORT_BYTES
    )


def test_measurement_binding_rejects_coordinated_outer_catalog_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = {
        "setup_file_sha256": "1" * 64,
        "setup_sha256": "2" * 64,
        "typed_response_report_file_sha256": "3" * 64,
        "typed_response_report_sha256": "4" * 64,
        "predecessor_measurement_input_sha256": "5" * 64,
        "corpus_file_sha256": "6" * 64,
        "panel_file_sha256": "7" * 64,
        "panel_audit_file_sha256": "8" * 64,
        "panel_audit_payload_sha256": "9" * 64,
        "confirmation_panel_module_sha256": "d" * 64,
        "confirmation_worker_source_sha256": "e" * 64,
    }
    fixture_records = _records()
    worker_token_rows = [
        {
            "prompt_id": record.prompt_id,
            "stratum_id": record.stratum_id,
            "valid_token_count": record.target_valid_tokens,
            "input_ids_sha256": hashlib.sha256(
                f"input:{index}".encode()
            ).hexdigest(),
            "choices_sha256": hashlib.sha256(
                f"choices:{index}".encode()
            ).hexdigest(),
            "answer_evidence_outside_window_authenticated": (
                record.answer_evidence is not None
            ),
            "answer_evidence_token_count": (
                2 if record.answer_evidence is not None else 0
            ),
            "visible_suffix_token_ids_sha256": (
                hashlib.sha256(
                    f"suffix:{record.semantic_family}".encode()
                ).hexdigest()
                if record.answer_evidence is not None
                else None
            ),
        }
        for index, record in enumerate(fixture_records)
    ]
    panel_token_rows = [
        {
            "prompt_id": row["prompt_id"],
            "stratum_id": row["stratum_id"],
            "valid_tokens": row["valid_token_count"],
            "input_ids_sha256": row["input_ids_sha256"],
            "choices_sha256": row["choices_sha256"],
            "answer_evidence_verified": row[
                "answer_evidence_outside_window_authenticated"
            ],
            "answer_evidence_token_count": row[
                "answer_evidence_token_count"
            ],
            "visible_suffix_token_ids_sha256": row[
                "visible_suffix_token_ids_sha256"
            ],
        }
        for row in worker_token_rows
    ]
    batch_groups = tuple(
        subject.iter_confirmation_length_batches(
            fixture_records, batch_size=4
        )
    )
    token_batches = []
    for batch_index, batch_group in enumerate(batch_groups):
        prompt_ids = [record.prompt_id for record in batch_group]
        token_batches.append(
            {
                "batch_index": batch_index,
                "stratum_id": batch_group[0].stratum_id,
                "target_valid_tokens": batch_group[0].target_valid_tokens,
                "row_count": len(batch_group),
                "prompt_id_catalog_sha256": subject._manifest_sha256(
                    subject._TOKENIZATION_BINDING_DOMAIN, prompt_ids
                ),
                "no_cross_length_padding": True,
            }
        )
    tokenization = {
        "tokenization_binding_sha256": "a" * 64,
        "rows": worker_token_rows,
        "batches": token_batches,
        "panel_tokenization_receipt": {"rows": panel_token_rows},
    }
    response_hashes = {
        mode_id: hashlib.sha256(f"response:{mode_id}".encode()).hexdigest()
        for mode_id in RESPONSE_MODE_IDS
    }
    state_hashes = {
        mode_id: hashlib.sha256(f"state:{mode_id}".encode()).hexdigest()
        for mode_id in RESPONSE_MODE_IDS
    }
    response_hashes[LEXICAL_MODE_IDS[1]] = response_hashes[
        LEXICAL_MODE_IDS[0]
    ]
    state_hashes[LEXICAL_MODE_IDS[1]] = state_hashes[LEXICAL_MODE_IDS[0]]
    ordered_states = [
        {
            "mode_id": mode_id,
            "state_fingerprint_sha256": state_hashes[mode_id],
        }
        for mode_id in RESPONSE_MODE_IDS
    ]
    state_catalog_sha256 = subject._manifest_sha256(
        subject._RESPONSE_CATALOG_DOMAIN, ordered_states
    )
    monkeypatch.setattr(
        subject,
        "_PINNED_PREDECESSOR_STATE_CATALOG_SHA256",
        state_catalog_sha256,
    )
    parity = [
        {
            "batch_index": index,
            "stratum_id": batch_group[0].stratum_id,
            "target_valid_tokens": batch_group[0].target_valid_tokens,
            "example_ids": [record.prompt_id for record in batch_group],
            "token_id_logits4_sha256": "b" * 64,
            "inputs_embeds_logits4_sha256": "b" * 64,
            "byte_identical": True,
        }
        for index, batch_group in enumerate(batch_groups)
    ]
    measurement = {
        "baseline_parity_receipts": parity,
        "all_baseline_parity_receipts_byte_identical": True,
        "runtime_output_Fisher_weight_catalog_sha256": "c" * 64,
        "runtime_output_Fisher_weight_catalog_row_count": 240,
        "response_sha256_by_mode": response_hashes,
        "mode_state_fingerprint_sha256_by_mode": state_hashes,
        "terminal_norm_state_fingerprint_sha256": (
            subject._PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256
        ),
        "tied_lexical_shared_state_fingerprint_sha256": state_hashes[
            LEXICAL_MODE_IDS[0]
        ],
    }
    measurement_input = {
        "stable_setup_file_sha256": lineage["setup_file_sha256"],
        "stable_setup_sha256": lineage["setup_sha256"],
        "typed_response_report_file_sha256": lineage[
            "typed_response_report_file_sha256"
        ],
        "typed_response_report_sha256": lineage[
            "typed_response_report_sha256"
        ],
        "predecessor_measurement_input_sha256": lineage[
            "predecessor_measurement_input_sha256"
        ],
        "corpus_file_sha256": lineage["corpus_file_sha256"],
        "panel_file_sha256": lineage["panel_file_sha256"],
        "panel_audit_file_sha256": lineage["panel_audit_file_sha256"],
        "panel_audit_payload_sha256": lineage[
            "panel_audit_payload_sha256"
        ],
        "confirmation_panel_module_sha256": lineage[
            "confirmation_panel_module_sha256"
        ],
        "confirmation_worker_source_sha256": lineage[
            "confirmation_worker_source_sha256"
        ],
        "tokenization_binding_sha256": tokenization[
            "tokenization_binding_sha256"
        ],
        "response_abi_sha256": subject.PINNED_RESPONSE_ABI_SHA256,
        "response_formula": "C4_times_z_scale1_minus_z_scale0",
        "predecessor_state_catalog_sha256": state_catalog_sha256,
        "baseline_parity_catalog_sha256": subject._manifest_sha256(
            subject._RESPONSE_CATALOG_DOMAIN, parity
        ),
        "runtime_output_Fisher_weight_catalog_sha256": measurement[
            "runtime_output_Fisher_weight_catalog_sha256"
        ],
        "ordered_response_sha256_catalog": [
            {
                "mode_id": mode_id,
                "response_sha256": response_hashes[mode_id],
            }
            for mode_id in RESPONSE_MODE_IDS
        ],
        "ordered_state_fingerprint_catalog": ordered_states,
        "terminal_norm_state_fingerprint_sha256": measurement[
            "terminal_norm_state_fingerprint_sha256"
        ],
        "confirmation_prompt_id_catalog_sha256": subject._manifest_sha256(
            subject._RESPONSE_CATALOG_DOMAIN,
            [record.prompt_id for record in fixture_records],
        ),
    }
    subject._validate_measurement_cross_bindings(
        lineage=lineage,
        tokenization=tokenization,
        measurement=measurement,
        measurement_input=measurement_input,
    )

    duplicate_batch = [dict(row) for row in parity]
    duplicate_batch[1]["batch_index"] = 0
    duplicate_measurement = dict(measurement)
    duplicate_measurement["baseline_parity_receipts"] = duplicate_batch
    duplicate_input = dict(measurement_input)
    duplicate_input["baseline_parity_catalog_sha256"] = (
        subject._manifest_sha256(
            subject._RESPONSE_CATALOG_DOMAIN, duplicate_batch
        )
    )
    with pytest.raises(ValueError, match="token batches are not aligned"):
        subject._validate_measurement_cross_bindings(
            lineage=lineage,
            tokenization=tokenization,
            measurement=duplicate_measurement,
            measurement_input=duplicate_input,
        )

    coordinated_batch_metadata = copy.deepcopy(tokenization)
    coordinated_batch_metadata["batches"][0]["stratum_id"] = "short_64"
    coordinated_parity = copy.deepcopy(measurement)
    coordinated_parity["baseline_parity_receipts"][0][
        "stratum_id"
    ] = "short_64"
    coordinated_input = dict(measurement_input)
    coordinated_input["baseline_parity_catalog_sha256"] = (
        subject._manifest_sha256(
            subject._RESPONSE_CATALOG_DOMAIN,
            coordinated_parity["baseline_parity_receipts"],
        )
    )
    with pytest.raises(ValueError, match="derived from token rows"):
        subject._validate_measurement_cross_bindings(
            lineage=lineage,
            tokenization=coordinated_batch_metadata,
            measurement=coordinated_parity,
            measurement_input=coordinated_input,
        )

    tampered_lineage_binding = dict(measurement_input)
    tampered_lineage_binding["corpus_file_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="outer lineage"):
        subject._validate_measurement_cross_bindings(
            lineage=lineage,
            tokenization=tokenization,
            measurement=measurement,
            measurement_input=tampered_lineage_binding,
        )

    tampered_measurement = dict(measurement)
    tampered_hashes = dict(response_hashes)
    tampered_hashes[RESPONSE_MODE_IDS[0]] = "f" * 64
    tampered_measurement["response_sha256_by_mode"] = tampered_hashes
    with pytest.raises(ValueError, match="response/state catalogs"):
        subject._validate_measurement_cross_bindings(
            lineage=lineage,
            tokenization=tokenization,
            measurement=tampered_measurement,
            measurement_input=measurement_input,
        )


def test_model_free_torch_probe_executes_exactly_eight_hundred_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()

    class FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_attention_layernorm = torch.nn.Identity()
            self.post_feedforward_layernorm = torch.nn.Identity()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = self.post_attention_layernorm(value)
            return self.post_feedforward_layernorm(value)

    class FakeInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(FakeLayer() for _ in range(18))
            self.norm = torch.nn.LayerNorm(4)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                value = layer(value)
            return self.norm(value)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.model = FakeInner()
            self.output = torch.nn.Linear(4, 8, bias=False)
            self.output.weight = self.embedding.weight
            self.forward_count = 0

        def get_input_embeddings(self) -> torch.nn.Module:
            return self.embedding

        def get_output_embeddings(self) -> torch.nn.Module:
            return self.output

        def forward(
            self,
            *,
            input_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            attention_mask: torch.Tensor,
            use_cache: bool,
            return_dict: bool,
        ) -> object:
            del attention_mask, use_cache, return_dict
            self.forward_count += 1
            value = (
                self.embedding(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            return SimpleNamespace(logits=self.output(self.model(value)))

    model = FakeModel()
    lexical_fingerprint = subject._physical_state_sha256(
        state_id="state.tied_lexical.embedding_lm_head",
        tensors=(model.embedding.weight,),
        torch=torch,
    )
    terminal_fingerprint = subject._physical_state_sha256(
        state_id="state.terminal.rmsnorm",
        tensors=tuple(model.model.norm.parameters()),
        torch=torch,
    )
    state_fingerprints = {
        mode_id: hashlib.sha256(f"state:{mode_id}".encode()).hexdigest()
        for mode_id in RESPONSE_MODE_IDS
    }
    state_fingerprints[LEXICAL_MODE_IDS[0]] = lexical_fingerprint
    state_fingerprints[LEXICAL_MODE_IDS[1]] = lexical_fingerprint
    ordered_states = [
        {
            "mode_id": mode_id,
            "state_fingerprint_sha256": state_fingerprints[mode_id],
        }
        for mode_id in RESPONSE_MODE_IDS
    ]
    monkeypatch.setattr(
        subject,
        "_PINNED_PREDECESSOR_STATE_CATALOG_SHA256",
        subject._manifest_sha256(
            subject._RESPONSE_CATALOG_DOMAIN, ordered_states
        ),
    )
    monkeypatch.setattr(
        subject,
        "_PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256",
        terminal_fingerprint,
    )

    class FakeAdapter:
        layers = tuple(SimpleNamespace(id=str(index)) for index in range(18))

        def source_module(self, layer_id: str) -> torch.nn.Module:
            return model.model.layers[int(layer_id)]

    runtime = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        adapter=FakeAdapter(),
        source_guard=SimpleNamespace(assert_unchanged=lambda: None),
    )
    batches: list[subject._TokenBatch] = []
    for stratum_index, (_stratum_id, target) in enumerate(STRATA):
        bucket = records[stratum_index * 6 : (stratum_index + 1) * 6]
        for offset in (0, 4):
            rows = bucket[offset : offset + 4]
            size = len(rows)
            batches.append(
                subject._TokenBatch(
                    model_inputs={
                        "input_ids": torch.zeros((size, 1), dtype=torch.long),
                        "attention_mask": torch.ones(
                            (size, 1), dtype=torch.long
                        ),
                    },
                    example_ids=tuple(row.prompt_id for row in rows),
                    family_ids=tuple(
                        subject._confirmation_family_id(row) for row in rows
                    ),
                    stratum_ids=tuple(str(row.stratum_id) for row in rows),
                    choice_ids=torch.tensor(
                        [[0, 1, 2, 3] for _row in rows], dtype=torch.long
                    ),
                    gold_choice=torch.tensor(
                        [row.correct_choice for row in rows], dtype=torch.long
                    ),
                    last_query_indices=torch.zeros(size, dtype=torch.long),
                    target_valid_tokens=target,
                )
            )
    tokenization = subject._AuthenticatedTokenization(
        batches=tuple(batches),
        receipt={"tokenization_binding_sha256": "a" * 64},
    )
    preflight = SimpleNamespace(
        confirmation_records=records,
        predecessor={
            "measurement": {
                "mode_state_fingerprint_sha256_by_mode": state_fingerprints,
                "terminal_norm_state_fingerprint_sha256": terminal_fingerprint,
            }
        },
        receipts={
            "setup_file_sha256": "1" * 64,
            "setup_sha256": "2" * 64,
            "typed_response_report_file_sha256": "3" * 64,
            "typed_response_report_sha256": "4" * 64,
            "corpus_file_sha256": "5" * 64,
            "panel_file_sha256": "6" * 64,
            "panel_audit_file_sha256": "7" * 64,
            "panel_audit_payload_sha256": "8" * 64,
            "confirmation_panel_module_sha256": "9" * 64,
            "confirmation_worker_source_path": str(
                Path(subject.__file__).resolve()
            ),
            "confirmation_worker_source_sha256": subject._sha256_file(
                Path(subject.__file__).resolve()
            ),
        },
    )
    measurement = subject._measure_responses_live(
        runtime=runtime,
        preflight=preflight,
        tokenization=tokenization,
        dependencies=SimpleNamespace(torch=torch),
    )
    assert model.forward_count == subject.PINNED_FULL_MODEL_FORWARD_COUNT == 800
    assert measurement.receipt["token_id_baseline_forward_count"] == 20
    assert measurement.receipt["inputs_embeds_baseline_forward_count"] == 20
    assert measurement.receipt["logical_intervention_forward_count"] == 760
    assert set(
        measurement.receipt[
            "logical_intervention_forward_count_by_mode"
        ].values()
    ) == {20}
    assert set(
        measurement.receipt[
            "target_hook_call_count_by_mutation_mode"
        ].values()
    ) == {20}
    assert measurement.receipt[
        "lexical_egress_occurrence_hook_call_count"
    ] == 20
    assert measurement.receipt["tied_lexical_responses_byte_identical"] is True
