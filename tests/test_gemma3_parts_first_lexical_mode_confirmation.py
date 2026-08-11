from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import fisher_graph.gemma3_parts_first_lexical_modes as lexical_modes
import fisher_graph.gemma3_parts_first_lexical_mode_confirmation as confirmation
from fisher_graph.gemma3_parts_first_lexical_mode_confirmation import (
    LexicalModeBConfirmationReceipt,
    build_lexical_mode_B_source,
    compile_frozen_B_lexical_mode_confirmation,
    compile_published_frozen_B_lexical_mode_confirmation,
    load_lexical_mode_B_confirmation,
    main,
    parse_lexical_mode_B_confirmation_json,
    validate_lexical_mode_B_confirmation,
    write_lexical_mode_B_confirmation,
)
from fisher_graph.gemma3_parts_first_supermode_experiment import (
    AtomResponseAxisEntry,
    CaptureRole,
    CaptureRow,
    PartsFirstCaptureBundle,
    build_demo_capture_bundle,
)


_TEST_CONFIG = lexical_modes.LexicalModeCompilerConfig(
    permutation_count=19,
    permutation_seed=20260811,
)


def _role(
    role: str,
    *,
    seed: int,
    A_signal: bool = False,
    B_signal: bool = False,
) -> CaptureRole:
    rng = np.random.default_rng(seed)
    rows: list[CaptureRow] = []
    rows_per_family = 6 if role == "A_discovery" else 2
    for family_ordinal in range(6):
        for local in range(rows_per_family):
            ingress = rng.normal(0.0, 0.2, size=640)
            readout = rng.normal(0.0, 0.2, size=640)
            if A_signal:
                centered = float(local) - 2.5
                alternate = (
                    centered * centered - (35.0 / 12.0)
                ) * (1.0 + 0.1 * family_ordinal)
                ingress[0] = 25.0 * centered
                ingress[1] = 25.0 * centered
                readout[0] = 0.01 * alternate
                readout[1] = -0.02 * centered
                readout[2] = 25.0 * alternate
                readout[3] = 25.0 * alternate
                ingress[2] = 0.01 * centered
                ingress[3] = -0.02 * alternate
            if B_signal:
                contrast = (-1.0 if local == 0 else 1.0) * (
                    1.0 + 0.1 * family_ordinal
                )
                ingress[0] = 25.0 * contrast
                ingress[1] = 25.0 * contrast
                readout[2] = 25.0 * contrast
                readout[3] = 25.0 * contrast
            responses = np.concatenate((ingress, readout))
            rows.append(
                CaptureRow(
                    row_id=f"{role.lower()}.{family_ordinal}.{local}",
                    family_id=f"family.{family_ordinal}",
                    chart_coordinates=(float(family_ordinal), float(local)),
                    target_score=float(local - family_ordinal),
                    atom_responses=tuple(float(item) for item in responses),
                )
            )
    return CaptureRole(
        role=role,
        rows=tuple(rows),
        exact_observer_passed=False,
    )


def _capture(*, B_signal: bool = True, C_seed: int = 3) -> PartsFirstCaptureBundle:
    base = build_demo_capture_bundle()
    axis = tuple(
        [AtomResponseAxisEntry(column, 0) for column in range(640)]
        + [AtomResponseAxisEntry(column, 38) for column in range(640)]
    )
    return replace(
        base,
        response_axis=axis,
        roles=(
            _role("A_discovery", seed=1, A_signal=True),
            _role("B_confirmation", seed=2, B_signal=B_signal),
            _role("C_intervention", seed=C_seed),
        ),
        capture_sha256="",
    )


def _write_json(path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _published_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    B_signal: bool = True,
    C_seed: int = 3,
):
    capture = _capture(B_signal=B_signal, C_seed=C_seed)
    capture_path = tmp_path / "capture.json"
    dictionary_path = tmp_path / "dictionary.json"
    _write_json(capture_path, capture.to_dict())
    A_source = lexical_modes.build_lexical_mode_A_source(capture)
    unauthenticated = lexical_modes.compile_lexical_mode_dictionary(
        A_source,
        _TEST_CONFIG,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_PARTS_FIRST_CAPTURE_SHA256",
        capture.capture_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_ROLE_SHA256",
        A_source.A_role.role_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_ROW_IDS_SHA256",
        A_source.A_role.row_ids_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_SOURCE_SHA256",
        A_source.A_source_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_MODE_CONFIG_SHA256",
        _TEST_CONFIG.config_sha256,
    )
    dictionary_payload = unauthenticated._payload()
    dictionary_payload["published_A_anchor_authenticated"] = True
    artifact_sha256 = lexical_modes._json_sha256(
        dictionary_payload,
        domain=lexical_modes._ARTIFACT_DOMAIN,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_MODE_ARTIFACT_SHA256",
        artifact_sha256,
    )
    monkeypatch.setattr(lexical_modes, "_published_config", lambda: _TEST_CONFIG)
    B_source = build_lexical_mode_B_source(capture)
    monkeypatch.setattr(
        confirmation,
        "PINNED_LEXICAL_B_ROLE_SHA256",
        B_source.B_role.role_sha256,
    )
    monkeypatch.setattr(
        confirmation,
        "PINNED_LEXICAL_B_ROW_IDS_SHA256",
        B_source.B_role.row_ids_sha256,
    )
    monkeypatch.setattr(
        confirmation,
        "PINNED_LEXICAL_B_SOURCE_SHA256",
        B_source.B_source_sha256,
    )
    dictionary = lexical_modes.compile_published_lexical_mode_dictionary(
        capture_path
    )
    _write_json(dictionary_path, dictionary.to_dict())
    authenticated_B = build_lexical_mode_B_source(
        capture,
        authenticate_published_capture=True,
    )
    test_receipt = compile_frozen_B_lexical_mode_confirmation(
        dictionary,
        authenticated_B,
    )
    monkeypatch.setattr(
        confirmation,
        "PINNED_LEXICAL_B_CONFIRMATION_RECEIPT_SHA256",
        test_receipt.receipt_sha256,
    )
    return capture, dictionary, authenticated_B, capture_path, dictionary_path


def test_frozen_B_confirms_only_frozen_roles_with_exact_independent_maxT(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, dictionary, B_source, _, _ = _published_inputs(monkeypatch, tmp_path)
    receipt = compile_frozen_B_lexical_mode_confirmation(dictionary, B_source)

    assert receipt.evaluated_candidate_count == 2
    assert receipt.confirmed_candidate_count == 2
    assert receipt.all_candidates_rejected is False
    assert receipt.roles_statistically_read == ("B_confirmation",)
    assert receipt.A_rows_read_for_dictionary_provenance_replay == 36
    assert receipt.A_rows_read_for_qualification == receipt.C_rows_read == 0
    assert receipt.attention_promotion_candidate_count == 0
    assert receipt.attention_promotion_performed is False
    assert receipt.B_was_visible_during_protocol_finalization is True
    assert receipt.clean_prospective_confirmation is False
    assert receipt.attention_promotion_authorized is False
    assert receipt.fresh_untouched_panel_required is True
    by_members = {
        item.member_columns: item for item in receipt.candidate_qualifications
    }
    assert tuple(
        role.role_scope for role in by_members[(0, 1)].role_qualifications
    ) == ("lexical_ingress",)
    assert tuple(
        role.role_scope for role in by_members[(2, 3)].role_qualifications
    ) == ("lexical_readout",)
    for candidate in receipt.candidate_qualifications:
        pair = candidate.role_qualifications[0].pair_evidence[0]
        assert pair.raw_exceedance_count == 1
        assert pair.raw_exact_p_value == 1 / 64
        assert pair.familywise_exceedance_count == 127
        assert pair.familywise_exact_p_value == 127 / 4096
        assert len(pair.familywise_max_pattern_cosines) == 4096


def test_failed_exact_evidence_never_promotes_or_claims_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, dictionary, B_source, _, _ = _published_inputs(
        monkeypatch,
        tmp_path,
        B_signal=False,
    )
    receipt = compile_frozen_B_lexical_mode_confirmation(dictionary, B_source)
    assert receipt.confirmed_candidate_count == 0
    assert receipt.all_candidates_rejected
    assert all(not item.passed for item in receipt.candidate_qualifications)
    assert receipt.semantic_mode_confirmed is False
    assert receipt.attention_chart_structure_claimed is False
    assert receipt.compression_attempted is False
    assert receipt.compression_claimed is False
    assert receipt.attention_promotion_candidate_count == 0


def test_C_poisoning_cannot_change_B_candidate_evaluations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first = _published_inputs(monkeypatch, first_dir, C_seed=3)
    first_receipt = compile_frozen_B_lexical_mode_confirmation(first[1], first[2])
    monkeypatch.undo()
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _published_inputs(monkeypatch, second_dir, C_seed=999)
    second_receipt = compile_frozen_B_lexical_mode_confirmation(second[1], second[2])

    assert first[0].capture_sha256 != second[0].capture_sha256
    assert (
        first_receipt.candidate_qualifications
        == second_receipt.candidate_qualifications
    )
    assert first_receipt.C_rows_read == second_receipt.C_rows_read == 0


def test_strict_roundtrip_replays_from_dictionary_and_B(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, dictionary, B_source, _, _ = _published_inputs(monkeypatch, tmp_path)
    receipt = compile_frozen_B_lexical_mode_confirmation(dictionary, B_source)
    encoded = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":"))
    parsed = parse_lexical_mode_B_confirmation_json(
        encoded,
        dictionary=dictionary,
        B_source=B_source,
    )
    assert parsed == receipt
    assert LexicalModeBConfirmationReceipt.from_dict(receipt.to_dict()) == receipt
    assert validate_lexical_mode_B_confirmation(
        receipt,
        dictionary=dictionary,
        B_source=B_source,
    ) is receipt


def test_coordinated_membership_rehash_rejects_on_source_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, dictionary, B_source, _, _ = _published_inputs(monkeypatch, tmp_path)
    receipt = compile_frozen_B_lexical_mode_confirmation(dictionary, B_source)
    first = receipt.candidate_qualifications[0]
    forged = replace(
        first,
        member_columns=(0, 2),
        candidate_sha256_B="",
    )
    attacked = replace(
        receipt,
        candidate_qualifications=(forged, *receipt.candidate_qualifications[1:]),
        frozen_candidate_inventory_sha256=confirmation._json_sha256(
            [
                {
                    "candidate_id": item.candidate_id,
                    "candidate_sha256": item.candidate_sha256,
                    "member_columns": list(item.member_columns),
                    "frozen_role_scope": item.frozen_role_scope,
                }
                for item in (forged, *receipt.candidate_qualifications[1:])
            ],
            domain=confirmation._INVENTORY_DOMAIN,
        ),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="does not replay"):
        validate_lexical_mode_B_confirmation(
            attacked,
            dictionary=dictionary,
            B_source=B_source,
        )


def test_strict_loader_rejects_duplicate_and_nonfinite_JSON(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, dictionary, B_source, _, _ = _published_inputs(monkeypatch, tmp_path)
    receipt = compile_frozen_B_lexical_mode_confirmation(dictionary, B_source)
    encoded = json.dumps(receipt.to_dict(), sort_keys=True)
    duplicate = encoded.replace(
        '"schema":',
        '"schema":"duplicate", "schema":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_lexical_mode_B_confirmation_json(
            duplicate,
            dictionary=dictionary,
            B_source=B_source,
        )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        parse_lexical_mode_B_confirmation_json(
            '{"payload":NaN}',
            dictionary=dictionary,
            B_source=B_source,
        )


def test_preflight_writes_nothing_and_writer_roundtrips_no_clobber(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, _, capture_path, dictionary_path = _published_inputs(
        monkeypatch,
        tmp_path,
    )
    output = tmp_path / "confirmation.json"
    assert main(
        [
            "--preflight-only",
            "--capture",
            str(capture_path),
            "--dictionary",
            str(dictionary_path),
            "--output",
            str(output),
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["published"] is False
    assert not output.exists()

    receipt = compile_published_frozen_B_lexical_mode_confirmation(
        capture=capture_path,
        dictionary=dictionary_path,
    )
    written = write_lexical_mode_B_confirmation(
        receipt,
        output,
        capture=capture_path,
        dictionary=dictionary_path,
    )
    assert written == receipt
    assert output.stat().st_mode & 0o777 == 0o600
    assert load_lexical_mode_B_confirmation(
        output,
        capture=capture_path,
        dictionary=dictionary_path,
    ) == receipt
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_lexical_mode_B_confirmation(
            receipt,
            output,
            capture=capture_path,
            dictionary=dictionary_path,
        )


def test_writer_rejects_a_validated_temporary_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, _, _, capture_path, dictionary_path = _published_inputs(
        monkeypatch,
        tmp_path,
    )
    output = tmp_path / "confirmation.json"
    receipt = compile_published_frozen_B_lexical_mode_confirmation(
        capture=capture_path,
        dictionary=dictionary_path,
    )
    original_loader = confirmation._load_receipt_path

    def replace_after_validation(path, *, dictionary, B_source):
        loaded = original_loader(
            path,
            dictionary=dictionary,
            B_source=B_source,
        )
        path.unlink()
        path.write_text('{"corrupt":true}\n', encoding="utf-8")
        return loaded

    monkeypatch.setattr(
        confirmation,
        "_load_receipt_path",
        replace_after_validation,
    )
    with pytest.raises(RuntimeError, match="temporary identity drifted"):
        write_lexical_mode_B_confirmation(
            receipt,
            output,
            capture=capture_path,
            dictionary=dictionary_path,
        )
    assert not output.exists()
    replacements = tuple(tmp_path.glob(".confirmation.json.*.tmp"))
    assert len(replacements) == 1
    assert replacements[0].read_text(encoding="utf-8") == '{"corrupt":true}\n'
    replacements[0].unlink()


def test_writer_rejects_a_post_reload_target_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, _, _, capture_path, dictionary_path = _published_inputs(
        monkeypatch,
        tmp_path,
    )
    output = tmp_path / "confirmation.json"
    receipt = compile_published_frozen_B_lexical_mode_confirmation(
        capture=capture_path,
        dictionary=dictionary_path,
    )
    original_loader = confirmation._load_receipt_path

    def replace_after_reload(path, *, dictionary, B_source):
        loaded = original_loader(
            path,
            dictionary=dictionary,
            B_source=B_source,
        )
        if path == output:
            path.unlink()
            path.write_text('{"corrupt":true}\n', encoding="utf-8")
        return loaded

    monkeypatch.setattr(
        confirmation,
        "_load_receipt_path",
        replace_after_reload,
    )
    with pytest.raises(RuntimeError, match="changed after canonical reload"):
        write_lexical_mode_B_confirmation(
            receipt,
            output,
            capture=capture_path,
            dictionary=dictionary_path,
        )
    assert output.read_text(encoding="utf-8") == '{"corrupt":true}\n'


def test_paths_fail_closed_before_writes() -> None:
    with pytest.raises(ValueError, match="output must use a .json suffix"):
        confirmation._validate_paths(
            capture="capture.json",
            dictionary="dictionary.json",
            output="receipt.txt",
        )
    with pytest.raises(ValueError, match="paths must differ"):
        confirmation._validate_paths(
            capture="same.json",
            dictionary="dictionary.json",
            output="same.json",
        )


def test_published_confirmation_receipt_pin_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, _, _, capture_path, dictionary_path = _published_inputs(
        monkeypatch,
        tmp_path,
    )
    monkeypatch.setattr(
        confirmation,
        "PINNED_LEXICAL_B_CONFIRMATION_RECEIPT_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="receipt pin drifted"):
        compile_published_frozen_B_lexical_mode_confirmation(
            capture=capture_path,
            dictionary=dictionary_path,
        )


def test_published_writer_and_loader_reject_receipt_pin_drift_before_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _, _, _, capture_path, dictionary_path = _published_inputs(
        monkeypatch,
        tmp_path,
    )
    receipt = compile_published_frozen_B_lexical_mode_confirmation(
        capture=capture_path,
        dictionary=dictionary_path,
    )
    output = tmp_path / "confirmation.json"
    monkeypatch.setattr(
        confirmation,
        "PINNED_LEXICAL_B_CONFIRMATION_RECEIPT_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="receipt pin drifted"):
        write_lexical_mode_B_confirmation(
            receipt,
            output,
            capture=capture_path,
            dictionary=dictionary_path,
        )
    assert not output.exists()

    _write_json(output, receipt.to_dict())
    with pytest.raises(ValueError, match="receipt pin drifted"):
        load_lexical_mode_B_confirmation(
            output,
            capture=capture_path,
            dictionary=dictionary_path,
        )
