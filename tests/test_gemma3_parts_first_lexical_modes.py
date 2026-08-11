from __future__ import annotations

from dataclasses import fields, replace
import json

import numpy as np
import pytest
import torch

import fisher_graph.gemma3_parts_first_lexical_modes as lexical_modes
from fisher_graph.gemma3_parts_first_lexical_modes import (
    LexicalModeCompilerConfig,
    LexicalModeDictionary,
    LexicalModeASource,
    build_lexical_mode_A_source,
    compile_lexical_mode_dictionary,
    compile_published_lexical_mode_dictionary,
    load_lexical_mode_capture_strict,
    load_published_lexical_mode_dictionary,
    main,
    parse_lexical_mode_dictionary_json,
    validate_lexical_mode_dictionary,
    write_published_lexical_mode_dictionary,
)
from fisher_graph.gemma3_parts_first_supermode_experiment import (
    AtomResponseAxisEntry,
    CaptureRole,
    CaptureRow,
    PartsFirstCaptureBundle,
    build_demo_capture_bundle,
)
from fisher_graph.gemma3_parts_first_supermode_surfaces import (
    decompose_tied_lexical_egress_column_groups,
    decompose_tied_lexical_ingress_column_groups,
)


_TEST_CONFIG = LexicalModeCompilerConfig(
    permutation_count=19,
    permutation_seed=20260811,
)


def _synthetic_role(
    role: str,
    *,
    seed: int,
    A_signal: bool,
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


def _capture(*, B_seed: int = 2, C_seed: int = 3) -> PartsFirstCaptureBundle:
    base = build_demo_capture_bundle()
    axis = tuple(
        [AtomResponseAxisEntry(column, 0) for column in range(640)]
        + [AtomResponseAxisEntry(column, 38) for column in range(640)]
    )
    roles = (
        _synthetic_role("A_discovery", seed=1, A_signal=True),
        _synthetic_role("B_confirmation", seed=B_seed, A_signal=False),
        _synthetic_role("C_intervention", seed=C_seed, A_signal=False),
    )
    return replace(
        base,
        response_axis=axis,
        roles=roles,
        capture_sha256="",
    )


def _write_capture(capture: PartsFirstCaptureBundle, path) -> None:
    path.write_text(
        json.dumps(
            capture.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _configure_test_publication(
    monkeypatch: pytest.MonkeyPatch,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> str:
    source = build_lexical_mode_A_source(capture)
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_PARTS_FIRST_CAPTURE_SHA256",
        capture.capture_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_ROLE_SHA256",
        source.A_role.role_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_ROW_IDS_SHA256",
        source.A_role.row_ids_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_SOURCE_SHA256",
        source.A_source_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_MODE_CONFIG_SHA256",
        _TEST_CONFIG.config_sha256,
    )
    payload = dictionary._payload()
    payload["published_A_anchor_authenticated"] = True
    artifact_sha256 = lexical_modes._json_sha256(
        payload,
        domain=lexical_modes._ARTIFACT_DOMAIN,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_MODE_ARTIFACT_SHA256",
        artifact_sha256,
    )
    monkeypatch.setattr(lexical_modes, "_published_config", lambda: _TEST_CONFIG)
    return artifact_sha256


@pytest.fixture(scope="module")
def capture() -> PartsFirstCaptureBundle:
    return _capture()


@pytest.fixture(scope="module")
def dictionary(capture: PartsFirstCaptureBundle) -> LexicalModeDictionary:
    return compile_lexical_mode_dictionary(
        build_lexical_mode_A_source(capture),
        _TEST_CONFIG,
    )


def test_compiler_preserves_physical_basis_and_discovers_role_scoped_modes(
    dictionary: LexicalModeDictionary,
) -> None:
    assert dictionary.physical_basis_count == 640
    assert dictionary.physical_basis_kind == "640_tied_hidden_column_singletons"
    by_members = {item.member_columns: item for item in dictionary.candidates}
    assert by_members[(0, 1)].role_scope == "lexical_ingress"
    assert by_members[(2, 3)].role_scope == "lexical_readout"
    assert all(item.storage_credit_scalar_count == 0 for item in dictionary.candidates)
    assert dictionary.roles_read == ("A_discovery",)
    assert dictionary.B_rows_read == dictionary.C_rows_read == 0
    assert dictionary.compression_attempted is False
    assert dictionary.compression_claimed is False
    assert dictionary.semantic_confirmation_claimed is False
    assert dictionary.attention_chart_structure_claimed is False


def test_B_and_C_poisoning_cannot_change_A_dictionary(
    dictionary: LexicalModeDictionary,
) -> None:
    poisoned = _capture(B_seed=200, C_seed=300)
    assert poisoned.capture_sha256 != _capture().capture_sha256
    replay = compile_lexical_mode_dictionary(
        build_lexical_mode_A_source(poisoned),
        _TEST_CONFIG,
    )
    assert replay.to_dict() == dictionary.to_dict()


def test_dictionary_strict_roundtrip_and_source_replay(
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    encoded = json.dumps(dictionary.to_dict(), sort_keys=True, separators=(",", ":"))
    A_source = build_lexical_mode_A_source(capture)
    parsed = parse_lexical_mode_dictionary_json(encoded, A_source=A_source)
    assert parsed == dictionary
    assert LexicalModeDictionary.from_dict(dictionary.to_dict()) == dictionary
    assert validate_lexical_mode_dictionary(dictionary, A_source) is dictionary


def test_coordinated_role_relabel_rehash_fails_source_replay(
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    original = dictionary.candidates[0]
    replacement_scope = (
        "lexical_ingress"
        if original.role_scope == "lexical_readout"
        else "lexical_readout"
    )
    forged_candidate = replace(
        original,
        role_scope=replacement_scope,
        candidate_sha256="",
    )
    with pytest.raises(ValueError, match="role scope does not close"):
        replace(
            dictionary,
            candidates=(forged_candidate, *dictionary.candidates[1:]),
            artifact_sha256="",
        )


def test_A_response_drift_rejects_dictionary_replay(
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    A_role = capture.roles[0]
    first = A_role.rows[0]
    responses = list(first.atom_responses)
    responses[0] += 1.0
    changed_A = replace(
        A_role,
        rows=(replace(first, atom_responses=tuple(responses)), *A_role.rows[1:]),
    )
    drifted = replace(
        capture,
        roles=(changed_A, *capture.roles[1:]),
        capture_sha256="",
    )
    with pytest.raises(ValueError, match="does not replay"):
        validate_lexical_mode_dictionary(
            dictionary,
            build_lexical_mode_A_source(drifted),
        )


def test_missing_lexical_occurrence_fails_closed(
    capture: PartsFirstCaptureBundle,
) -> None:
    shortened = replace(
        capture,
        response_axis=capture.response_axis[:-1],
        roles=tuple(
            replace(
                role,
                rows=tuple(
                    replace(row, atom_responses=row.atom_responses[:-1])
                    for row in role.rows
                ),
            )
            for role in capture.roles
        ),
        capture_sha256="",
    )
    with pytest.raises(ValueError, match="all 640 tied lexical columns"):
        compile_lexical_mode_dictionary(
            build_lexical_mode_A_source(shortened),
            _TEST_CONFIG,
        )


def test_candidate_unit_sums_close_ingress_and_egress(
    dictionary: LexicalModeDictionary,
) -> None:
    candidate = next(
        item for item in dictionary.candidates if item.member_columns == (0, 1)
    )
    assert candidate.ingress_partition.subset_columns == candidate.member_columns
    assert candidate.readout_partition.subset_columns == candidate.member_columns
    assert (
        candidate.ingress_partition.membership_blob_sha256
        == candidate.readout_partition.membership_blob_sha256
    )
    groups = (
        candidate.member_columns,
        candidate.ingress_partition.complement_columns,
    )
    generator = torch.Generator().manual_seed(7)
    tied_weight = torch.randn(17, 640, generator=generator, dtype=torch.float32)
    input_ids = torch.tensor([[1, 3, 9]], dtype=torch.int64)
    scale = 2.5
    native_ingress = tied_weight[input_ids] * scale
    ingress = decompose_tied_lexical_ingress_column_groups(
        input_ids=input_ids,
        tied_weight=tied_weight,
        embed_scale=scale,
        groups=groups,
        reference_output=native_ingress,
    )
    assert ingress.additive_closure_passed
    hidden = torch.randn(2, 3, 640, generator=generator, dtype=torch.float32)
    native_egress = hidden @ tied_weight.transpose(0, 1)
    egress = decompose_tied_lexical_egress_column_groups(
        hidden_states=hidden,
        tied_weight=tied_weight,
        groups=groups,
        reference_output=native_egress,
    )
    assert egress.additive_closure_passed


def test_null_receipt_is_deterministic_and_calibrates_effective_floor(
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    replay = compile_lexical_mode_dictionary(
        build_lexical_mode_A_source(capture),
        _TEST_CONFIG,
    )
    assert replay.null_receipt == dictionary.null_receipt
    receipt = dictionary.null_receipt
    assert receipt.sorted_max_absolute_cosines == tuple(
        sorted(receipt.sorted_max_absolute_cosines)
    )
    assert receipt.effective_joint_similarity_floor == max(
        _TEST_CONFIG.signed_similarity_floor,
        receipt.max_absolute_cosine_q95,
    )


def test_invalid_config_and_exact_type_attacks_reject(
    capture: PartsFirstCaptureBundle,
) -> None:
    with pytest.raises(ValueError, match="integer >= 19"):
        LexicalModeCompilerConfig(permutation_count=18)

    class CaptureSubclass(PartsFirstCaptureBundle):
        pass

    attacked = CaptureSubclass(
        **{item.name: getattr(capture, item.name) for item in fields(capture)}
    )
    with pytest.raises(TypeError, match="redacted A source"):
        compile_lexical_mode_dictionary(attacked, _TEST_CONFIG)

    A_source = build_lexical_mode_A_source(capture)
    assert type(A_source) is LexicalModeASource
    assert A_source.A_role.role == "A_discovery"
    assert not hasattr(A_source, "roles")


def test_null_config_semantic_and_membership_rehash_attacks_reject(
    dictionary: LexicalModeDictionary,
) -> None:
    forged_null = replace(
        dictionary.null_receipt,
        signed_similarity_floor=0.01,
        effective_joint_similarity_floor=(
            dictionary.null_receipt.max_absolute_cosine_q95
        ),
        null_sha256="",
    )
    with pytest.raises(ValueError, match="similarity floor drifted"):
        replace(
            dictionary,
            null_receipt=forged_null,
            artifact_sha256="",
        )

    candidate = dictionary.candidates[0]
    forged_candidate = replace(
        candidate,
        joint_min_signed_cosine=-1.0,
        parity_joint_signed_cosines=(-1.0, -1.0),
        lofo_joint_signed_cosines=(-1.0,) * 6,
        lofo_pass_count=0,
        active_family_count=0,
        balanced_empirical_fisher_mass=2.0,
        candidate_sha256="",
    )
    with pytest.raises(ValueError, match="calibrated A gate"):
        replace(
            dictionary,
            candidates=(forged_candidate, *dictionary.candidates[1:]),
            artifact_sha256="",
        )

    forged_readout = replace(
        candidate.readout_partition,
        membership_blob_sha256="1" * 64,
        partition_sha256="",
    )
    divergent_candidate = replace(
        candidate,
        readout_partition=forged_readout,
        candidate_sha256="",
    )
    with pytest.raises(ValueError, match="membership diverged"):
        replace(
            dictionary,
            candidates=(divergent_candidate, *dictionary.candidates[1:]),
            artifact_sha256="",
        )


def test_role_qualified_graphs_preserve_mixed_role_submodes() -> None:
    stable_pairs = tuple(
        (left, right, (0.9, 0.9), (0.9,) * 6, 6)
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    ingress = np.eye(3)
    readout = np.eye(3)
    ingress[0, 1] = ingress[1, 0] = 0.72
    ingress[0, 2] = ingress[2, 0] = 0.98
    ingress[1, 2] = ingress[2, 1] = 0.81
    readout[0, 1] = readout[1, 0] = 0.92
    readout[0, 2] = readout[2, 0] = 0.73
    readout[1, 2] = readout[2, 1] = 0.94
    ingress_pairs = tuple(
        pair
        for pair in stable_pairs
        if lexical_modes._role_edge_passes(
            pair[0],
            pair[1],
            full_cosine=ingress,
            parity_cosines=(ingress, ingress),
            lofo_cosines=(ingress,) * 6,
            config=_TEST_CONFIG,
        )
    )
    readout_pairs = tuple(
        pair
        for pair in stable_pairs
        if lexical_modes._role_edge_passes(
            pair[0],
            pair[1],
            full_cosine=readout,
            parity_cosines=(readout, readout),
            lofo_cosines=(readout,) * 6,
            config=_TEST_CONFIG,
        )
    )
    ingress_cliques = lexical_modes._all_stable_cliques(
        ingress_pairs,
        maximum_size=8,
        maximum_count=256,
    )
    readout_cliques = lexical_modes._all_stable_cliques(
        readout_pairs,
        maximum_size=8,
        maximum_count=256,
    )
    assert set(ingress_cliques).union(readout_cliques) == {
        (0, 1),
        (0, 2),
        (1, 2),
    }


def test_all_subcliques_survive_a_rotating_LOFO_triangle() -> None:
    threshold = _TEST_CONFIG.role_stability_floor
    full = np.full((3, 3), 0.95, dtype=np.float64)
    np.fill_diagonal(full, 1.0)
    lofo = [full.copy() for _ in range(6)]
    # Every edge fails a different held-family panel, so each pair passes 5/6.
    # The triangle's per-panel minimum fails three panels and must not erase
    # those independently qualified pair candidates.
    for panel, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
        lofo[panel][left, right] = lofo[panel][right, left] = threshold - 0.01
    stable_pairs = tuple(
        (
            left,
            right,
            (0.95, 0.95),
            tuple(float(matrix[left, right]) for matrix in lofo),
            6,
        )
        for left, right in ((0, 1), (0, 2), (1, 2))
    )

    cliques = lexical_modes._all_stable_cliques(
        stable_pairs,
        maximum_size=8,
        maximum_count=256,
    )
    assert cliques == ((0, 1), (0, 1, 2), (0, 2), (1, 2))
    aggregate_passes = tuple(
        members
        for members in cliques
        if sum(
            lexical_modes._minimum_pair_cosine(matrix, members) >= threshold
            for matrix in lofo
        )
        >= _TEST_CONFIG.minimum_lofo_pass_count
    )
    assert aggregate_passes == ((0, 1), (0, 2), (1, 2))


def test_all_subcliques_fail_closed_before_dense_graph_expansion() -> None:
    stable_pairs = tuple(
        (left, right, (0.9, 0.9), (0.9,) * 6, 6)
        for left in range(5)
        for right in range(left + 1, 5)
    )
    with pytest.raises(ValueError, match="maximum_candidate_count"):
        lexical_modes._all_stable_cliques(
            stable_pairs,
            maximum_size=8,
            maximum_count=3,
        )


def test_authenticated_A_source_rejects_exact_axis_swap(
    monkeypatch: pytest.MonkeyPatch,
    capture: PartsFirstCaptureBundle,
) -> None:
    source = build_lexical_mode_A_source(capture)
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_PARTS_FIRST_CAPTURE_SHA256",
        capture.capture_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_ROLE_SHA256",
        source.A_role.role_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_ROW_IDS_SHA256",
        source.A_role.row_ids_sha256,
    )
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_A_SOURCE_SHA256",
        source.A_source_sha256,
    )
    authenticated = replace(
        source,
        published_A_anchor_authenticated=True,
        A_source_sha256="",
    )
    assert authenticated.A_source_sha256 == source.A_source_sha256

    swapped = list(authenticated.response_axis)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="does not match its pins"):
        replace(
            authenticated,
            response_axis=tuple(swapped),
            A_source_sha256="",
        )


def test_published_compile_write_and_fresh_replay_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    expected_artifact = _configure_test_publication(
        monkeypatch,
        capture,
        dictionary,
    )
    capture_path = tmp_path / "capture.json"
    output_path = tmp_path / "dictionary.json"
    _write_capture(capture, capture_path)

    compiled = compile_published_lexical_mode_dictionary(capture_path)
    assert compiled.artifact_sha256 == expected_artifact
    written = write_published_lexical_mode_dictionary(
        compiled,
        output_path,
        capture=capture_path,
    )
    assert written == compiled
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert load_published_lexical_mode_dictionary(
        output_path,
        capture=capture_path,
    ) == compiled


def test_preflight_cli_performs_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    expected_artifact = _configure_test_publication(
        monkeypatch,
        capture,
        dictionary,
    )
    capture_path = tmp_path / "capture.json"
    output_path = tmp_path / "dictionary.json"
    _write_capture(capture, capture_path)

    assert main(
        [
            "--preflight-only",
            "--capture",
            str(capture_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["artifact_sha256"] == expected_artifact
    assert receipt["published"] is False
    assert not output_path.exists()
    assert tuple(tmp_path.iterdir()) == (capture_path,)


def test_published_writer_preserves_an_existing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    _configure_test_publication(monkeypatch, capture, dictionary)
    capture_path = tmp_path / "capture.json"
    output_path = tmp_path / "dictionary.json"
    _write_capture(capture, capture_path)
    compiled = compile_published_lexical_mode_dictionary(capture_path)
    output_path.write_text("existing sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_published_lexical_mode_dictionary(
            compiled,
            output_path,
            capture=capture_path,
        )
    assert output_path.read_text(encoding="utf-8") == "existing sentinel\n"
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == (
        "capture.json",
        "dictionary.json",
    )


def test_published_writer_rejects_a_validated_temporary_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    _configure_test_publication(monkeypatch, capture, dictionary)
    capture_path = tmp_path / "capture.json"
    output_path = tmp_path / "dictionary.json"
    _write_capture(capture, capture_path)
    compiled = compile_published_lexical_mode_dictionary(capture_path)
    original_loader = lexical_modes._load_lexical_mode_dictionary_path

    def replace_after_validation(path, *, A_source):
        loaded = original_loader(path, A_source=A_source)
        path.unlink()
        path.write_text('{"corrupt":true}\n', encoding="utf-8")
        return loaded

    monkeypatch.setattr(
        lexical_modes,
        "_load_lexical_mode_dictionary_path",
        replace_after_validation,
    )
    with pytest.raises(RuntimeError, match="temporary identity drifted"):
        write_published_lexical_mode_dictionary(
            compiled,
            output_path,
            capture=capture_path,
        )
    assert not output_path.exists()
    replacements = tuple(tmp_path.glob(".dictionary.json.*.tmp"))
    assert len(replacements) == 1
    assert replacements[0].read_text(encoding="utf-8") == '{"corrupt":true}\n'
    replacements[0].unlink()


def test_published_writer_rejects_a_post_reload_target_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    _configure_test_publication(monkeypatch, capture, dictionary)
    capture_path = tmp_path / "capture.json"
    output_path = tmp_path / "dictionary.json"
    _write_capture(capture, capture_path)
    compiled = compile_published_lexical_mode_dictionary(capture_path)
    original_loader = lexical_modes._load_lexical_mode_dictionary_path

    def replace_after_reload(path, *, A_source):
        loaded = original_loader(path, A_source=A_source)
        if path == output_path:
            path.unlink()
            path.write_text('{"corrupt":true}\n', encoding="utf-8")
        return loaded

    monkeypatch.setattr(
        lexical_modes,
        "_load_lexical_mode_dictionary_path",
        replace_after_reload,
    )
    with pytest.raises(RuntimeError, match="changed after canonical reload"):
        write_published_lexical_mode_dictionary(
            compiled,
            output_path,
            capture=capture_path,
        )
    assert output_path.read_text(encoding="utf-8") == '{"corrupt":true}\n'


def test_strict_file_loaders_reject_duplicate_and_nonfinite_json(
    tmp_path,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    capture_text = json.dumps(capture.to_dict(), sort_keys=True)
    duplicate_capture = tmp_path / "duplicate-capture.json"
    duplicate_capture.write_text(
        capture_text.replace(
            '"schema":',
            '"schema":"duplicate", "schema":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_lexical_mode_capture_strict(duplicate_capture)
    nonfinite_capture = tmp_path / "nonfinite-capture.json"
    nonfinite_capture.write_text('{"payload":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_lexical_mode_capture_strict(nonfinite_capture)

    A_source = build_lexical_mode_A_source(capture)
    dictionary_text = json.dumps(dictionary.to_dict(), sort_keys=True)
    duplicate_dictionary = tmp_path / "duplicate-dictionary.json"
    duplicate_dictionary.write_text(
        dictionary_text.replace(
            '"schema":',
            '"schema":"duplicate", "schema":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        lexical_modes.load_lexical_mode_dictionary(
            duplicate_dictionary,
            A_source=A_source,
        )
    nonfinite_dictionary = tmp_path / "nonfinite-dictionary.json"
    nonfinite_dictionary.write_text('{"payload":Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        lexical_modes.load_lexical_mode_dictionary(
            nonfinite_dictionary,
            A_source=A_source,
        )


def test_production_paths_reject_wrong_suffix_and_alias(
    tmp_path,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    wrong_capture = tmp_path / "capture.txt"
    _write_capture(capture, wrong_capture)
    with pytest.raises(ValueError, match="capture must use a .json suffix"):
        compile_published_lexical_mode_dictionary(wrong_capture)
    with pytest.raises(ValueError, match="output must use a .json suffix"):
        write_published_lexical_mode_dictionary(
            dictionary,
            tmp_path / "dictionary.txt",
            capture=tmp_path / "capture.json",
        )
    same_path = tmp_path / "same.json"
    with pytest.raises(ValueError, match="paths must differ"):
        main(
            [
                "--write",
                "--capture",
                str(same_path),
                "--output",
                str(same_path),
            ]
        )


def test_published_artifact_pin_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capture: PartsFirstCaptureBundle,
    dictionary: LexicalModeDictionary,
) -> None:
    _configure_test_publication(monkeypatch, capture, dictionary)
    monkeypatch.setattr(
        lexical_modes,
        "PINNED_LEXICAL_MODE_ARTIFACT_SHA256",
        "0" * 64,
    )
    capture_path = tmp_path / "capture.json"
    _write_capture(capture, capture_path)
    with pytest.raises(ValueError, match="artifact pin drifted"):
        compile_published_lexical_mode_dictionary(capture_path)


def test_duplicate_json_key_rejects(
    dictionary: LexicalModeDictionary,
) -> None:
    encoded = json.dumps(dictionary.to_dict(), sort_keys=True)
    attacked = encoded.replace(
        '"schema":',
        '"schema":"duplicate", "schema":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_lexical_mode_dictionary_json(
            attacked,
            A_source=build_lexical_mode_A_source(_capture()),
        )
