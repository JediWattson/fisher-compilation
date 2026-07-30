from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fisher_graph.gemma3_l3_l4_progressive_a_corpus as corpus_module
import fisher_graph.gemma3_l3_l4_progressive_guard_ledger as guard_ledger
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
    frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest,
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpusIntegrityError,
    Gemma3L3L4ProgressiveAGuardClosedError,
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    gemma3_l3_l4_progressive_a_fit_replacement_lineage,
    load_gemma3_l3_l4_progressive_a_corpus,
    load_gemma3_l3_l4_progressive_a_fit_role,
    replace_gemma3_l3_l4_progressive_a_fit_role,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)
from fisher_graph.gemma3_l3_l4_progressive_guard_ledger import (
    claim_gemma3_l3_l4_progressive_guard,
)


_ROLES = (
    "calibration_a_fit",
    "calibration_a_selection",
    "calibration_a_guard",
)
_TOKENIZER_CONTRACT = {
    "model_id": "google/gemma-3-270m",
    "tokenizer_revision": "fixture-revision",
    "add_special_tokens": True,
    "padding_side": "left",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(index: int) -> str:
    return f"{index:064x}"


def _write_role_inputs(
    root: Path,
    *,
    profile: str = "pilot",
    corpus_id: str = "three-role-a-v1",
    role_members: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
    | None = None,
) -> dict[str, Path]:
    members = role_members or {
        "calibration_a_fit": (
            (
                "Fit example: name the smallest prime.",
                "Fit example: reverse the sequence 3, 4, 5.",
            ),
            ("math-primes", "sequence-reversal"),
        ),
        "calibration_a_selection": (
            (
                "Selection example: give a two-word weather summary.",
                "Selection example: classify oak as plant or mineral.",
            ),
            ("weather-summary", "taxonomy-plants"),
        ),
        "calibration_a_guard": (
            (
                "Guard example: identify the direct object in this sentence.",
                "Guard example: continue the alternating bit pattern.",
            ),
            ("grammar-objects", "binary-patterns"),
        ),
    }
    paths: dict[str, Path] = {}
    for role in _ROLES:
        prompts, families = members[role]
        path = root / f"{profile}-{role}.json"
        write_gemma3_l3_l4_progressive_a_role_input(
            path,
            corpus_id=corpus_id,
            profile=profile,
            role=role,
            prompts=prompts,
            family_ids=families,
        )
        paths[role] = path
    return paths


def _freeze(
    root: Path,
    *,
    profile: str = "pilot",
    tokenizer_contract: dict[str, object] | None = None,
    role_members: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
    | None = None,
):
    paths = _write_role_inputs(
        root,
        profile=profile,
        role_members=role_members,
    )
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id="three-role-a-v1",
        profile=profile,
        tokenizer_contract=tokenizer_contract or _TOKENIZER_CONTRACT,
        role_input_paths=paths,
    )
    artifact_path = root / f"{profile}-corpus.json"
    write_gemma3_l3_l4_progressive_a_corpus_artifact(
        artifact_path,
        artifact,
    )
    return artifact, artifact_path, paths


@pytest.fixture
def private_guard_ledger(tmp_path, monkeypatch):
    path = tmp_path / "private-guard-ledger"
    monkeypatch.setattr(
        guard_ledger,
        "_FROZEN_LEDGER_ROOT",
        path,
    )
    return path


@pytest.mark.parametrize("profile", ("pilot", "full"))
def test_freezes_deterministic_prompt_free_three_role_artifact(
    tmp_path,
    profile: str,
) -> None:
    artifact, artifact_path, paths = _freeze(
        tmp_path / "first",
        profile=profile,
    )
    repeated, repeated_path, _ = _freeze(
        tmp_path / "second",
        profile=profile,
    )

    assert artifact.to_dict() == repeated.to_dict()
    assert artifact_path.read_bytes() == repeated_path.read_bytes()
    assert artifact.profile == profile
    assert tuple(view.role for view in artifact.role_views) == _ROLES
    assert len(
        {view.manifest_sha256 for view in artifact.role_views}
    ) == 3

    encoded = artifact_path.read_bytes()
    assert encoded == _canonical_bytes(json.loads(encoded))
    for path in paths.values():
        role_payload = json.loads(path.read_bytes())
        for prompt in role_payload["prompts"]:
            assert prompt.encode("utf-8") not in encoded
    for view in artifact.role_views:
        role_payload = json.loads(paths[view.role].read_bytes())
        assert view.ordered_prompt_sha256s == tuple(
            gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
            for prompt in role_payload["prompts"]
        )
        assert view.ordered_family_ids == tuple(
            role_payload["family_ids"]
        )
        assert view.role_input_file_sha256 == hashlib.sha256(
            paths[view.role].read_bytes()
        ).hexdigest()

    protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    expected_b_manifest = protocol.metadata()["corpus"][
        "calibration_b_manifest"
    ]["artifact_sha256"]
    assert artifact.forbidden_assessment_manifest_sha256s == (
        expected_b_manifest,
    )


def test_role_manifest_binds_ordered_members_and_tokenizer_contract(
    tmp_path,
) -> None:
    artifact, _, _ = _freeze(tmp_path / "base")
    changed_tokenizer, _, _ = _freeze(
        tmp_path / "tokenizer",
        tokenizer_contract={
            **_TOKENIZER_CONTRACT,
            "tokenizer_revision": "different-revision",
        },
    )
    reordered_members = {
        "calibration_a_fit": (
            (
                "Fit example: reverse the sequence 3, 4, 5.",
                "Fit example: name the smallest prime.",
            ),
            ("sequence-reversal", "math-primes"),
        ),
        "calibration_a_selection": (
            (
                "Selection example: give a two-word weather summary.",
                "Selection example: classify oak as plant or mineral.",
            ),
            ("weather-summary", "taxonomy-plants"),
        ),
        "calibration_a_guard": (
            (
                "Guard example: identify the direct object in this sentence.",
                "Guard example: continue the alternating bit pattern.",
            ),
            ("grammar-objects", "binary-patterns"),
        ),
    }
    reordered, _, _ = _freeze(
        tmp_path / "reordered",
        role_members=reordered_members,
    )

    original_manifests = {
        view.role: view.manifest_sha256
        for view in artifact.role_views
    }
    assert all(
        changed_tokenizer.role_view(role).manifest_sha256
        != original_manifests[role]
        for role in _ROLES
    )
    assert (
        reordered.role_view("calibration_a_fit").manifest_sha256
        != original_manifests["calibration_a_fit"]
    )
    assert (
        reordered.role_view("calibration_a_selection").manifest_sha256
        == original_manifests["calibration_a_selection"]
    )
    assert (
        reordered.role_view("calibration_a_guard").manifest_sha256
        == original_manifests["calibration_a_guard"]
    )


def test_load_and_preclaim_do_not_open_any_role_json(
    tmp_path,
    monkeypatch,
) -> None:
    artifact, artifact_path, paths = _freeze(tmp_path / "corpus")
    observed: list[Path] = []
    original_read = corpus_module._read_json_bytes

    def tracked_read(path: Path, *, label: str) -> bytes:
        observed.append(path)
        return original_read(path, label=label)

    monkeypatch.setattr(corpus_module, "_read_json_bytes", tracked_read)
    loaded = load_gemma3_l3_l4_progressive_a_corpus(
        artifact_path,
        role_input_paths=paths,
        expected_artifact_sha256=artifact.artifact_sha256,
        tokenizer_contract=_TOKENIZER_CONTRACT,
    )
    assert observed == [artifact_path]

    for role in _ROLES:
        view = loaded.preclaim_view(role)
        assert view.role == role
        assert not hasattr(view, "prompts")
    assert observed == [artifact_path]

    fit = loaded.open_development_role("calibration_a_fit")
    assert fit.role == "calibration_a_fit"
    assert observed == [artifact_path, paths["calibration_a_fit"]]
    assert (
        loaded.open_development_role("calibration_a_fit")
        is fit
    )
    assert observed == [artifact_path, paths["calibration_a_fit"]]
    with pytest.raises(
        Gemma3L3L4ProgressiveAGuardClosedError,
        match="open_guard_after_claim",
    ):
        loaded.open_development_role("calibration_a_guard")  # type: ignore[arg-type]
    assert observed == [artifact_path, paths["calibration_a_fit"]]


def test_fit_only_loader_requires_and_opens_only_the_fit_role(
    tmp_path,
    monkeypatch,
) -> None:
    artifact, artifact_path, paths = _freeze(tmp_path / "corpus")
    paths["calibration_a_selection"].unlink()
    paths["calibration_a_guard"].unlink()
    observed: list[Path] = []
    original_read = corpus_module._read_json_bytes

    def tracked_read(path: Path, *, label: str) -> bytes:
        observed.append(path)
        return original_read(path, label=label)

    monkeypatch.setattr(corpus_module, "_read_json_bytes", tracked_read)
    loaded_artifact, fit = load_gemma3_l3_l4_progressive_a_fit_role(
        artifact_path,
        fit_input_path=paths["calibration_a_fit"],
        expected_artifact_sha256=artifact.artifact_sha256,
        tokenizer_contract=_TOKENIZER_CONTRACT,
    )

    assert loaded_artifact == artifact
    assert fit.role == "calibration_a_fit"
    assert fit.ordered_prompt_sha256s == artifact.role_view(
        "calibration_a_fit"
    ).ordered_prompt_sha256s
    assert observed == [
        artifact_path,
        paths["calibration_a_fit"],
    ]


def test_fit_role_replacement_preserves_unopened_protected_views(
    tmp_path,
    monkeypatch,
) -> None:
    parent, artifact_path, paths = _freeze(tmp_path / "corpus")
    replacement_fit_path = tmp_path / "replacement-fit.json"
    write_gemma3_l3_l4_progressive_a_role_input(
        replacement_fit_path,
        corpus_id=parent.corpus_id,
        profile=parent.profile,
        role="calibration_a_fit",
        prompts=(
            "Replacement fit prompt one.",
            "Replacement fit prompt two.",
        ),
        family_ids=(
            "replacement-fit-family-one",
            "replacement-fit-family-two",
        ),
    )
    paths["calibration_a_selection"].unlink()
    paths["calibration_a_guard"].unlink()
    observed: list[Path] = []
    original_read = corpus_module._read_json_bytes

    def tracked_read(path: Path, *, label: str) -> bytes:
        observed.append(path)
        return original_read(path, label=label)

    monkeypatch.setattr(corpus_module, "_read_json_bytes", tracked_read)
    replacement = replace_gemma3_l3_l4_progressive_a_fit_role(
        artifact_path,
        fit_input_path=replacement_fit_path,
        expected_parent_artifact_sha256=parent.artifact_sha256,
        tokenizer_contract=_TOKENIZER_CONTRACT,
    )

    assert observed == [artifact_path, replacement_fit_path]
    assert replacement.artifact_sha256 != parent.artifact_sha256
    assert replacement.role_view("calibration_a_fit") != parent.role_view(
        "calibration_a_fit"
    )
    for role in ("calibration_a_selection", "calibration_a_guard"):
        assert replacement.role_view(role).to_dict() == parent.role_view(
            role
        ).to_dict()
    lineage = gemma3_l3_l4_progressive_a_fit_replacement_lineage(
        parent,
        replacement,
    )
    assert lineage["kind"] == "fit_role_only_replacement"
    assert lineage["parent_corpus_artifact_sha256"] == (
        parent.artifact_sha256
    )
    assert lineage["replacement_corpus_artifact_sha256"] == (
        replacement.artifact_sha256
    )

    replacement_artifact_path = tmp_path / "replacement-corpus.json"
    write_gemma3_l3_l4_progressive_a_corpus_artifact(
        replacement_artifact_path,
        replacement,
    )
    loaded, fit = load_gemma3_l3_l4_progressive_a_fit_role(
        replacement_artifact_path,
        fit_input_path=replacement_fit_path,
        expected_artifact_sha256=replacement.artifact_sha256,
        tokenizer_contract=_TOKENIZER_CONTRACT,
    )
    assert loaded == replacement
    assert fit.family_ids == (
        "replacement-fit-family-one",
        "replacement-fit-family-two",
    )


def test_fit_role_replacement_rejects_parent_or_protected_overlap(
    tmp_path,
) -> None:
    parent, artifact_path, paths = _freeze(tmp_path / "corpus")
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="expected logical",
    ):
        replace_gemma3_l3_l4_progressive_a_fit_role(
            artifact_path,
            fit_input_path=paths["calibration_a_fit"],
            expected_parent_artifact_sha256=_sha(999),
        )
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="must change",
    ):
        replace_gemma3_l3_l4_progressive_a_fit_role(
            artifact_path,
            fit_input_path=paths["calibration_a_fit"],
            expected_parent_artifact_sha256=parent.artifact_sha256,
        )

    colliding_fit = tmp_path / "colliding-fit.json"
    write_gemma3_l3_l4_progressive_a_role_input(
        colliding_fit,
        corpus_id=parent.corpus_id,
        profile=parent.profile,
        role="calibration_a_fit",
        prompts=("New prompt with a protected family.",),
        family_ids=(
            parent.role_view("calibration_a_selection").family_ids[0],
        ),
    )
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="family disjoint",
    ):
        replace_gemma3_l3_l4_progressive_a_fit_role(
            artifact_path,
            fit_input_path=colliding_fit,
            expected_parent_artifact_sha256=parent.artifact_sha256,
        )


def test_guard_requires_matching_durable_claim_and_opens_once(
    tmp_path,
    private_guard_ledger,
) -> None:
    artifact, artifact_path, paths = _freeze(tmp_path / "corpus")
    loaded = load_gemma3_l3_l4_progressive_a_corpus(
        artifact_path,
        role_input_paths=paths,
    )

    with pytest.raises(TypeError, match="durable"):
        loaded.open_guard_after_claim(object())  # type: ignore[arg-type]
    assert not loaded.guard_consumed
    assert not loaded.guard_opened

    unrelated = claim_gemma3_l3_l4_progressive_guard(
        protocol_sha256=_sha(1),
        guard_manifest_sha256=_sha(2),
        challenger_receipt_sha256=_sha(3),
    )
    with pytest.raises(
        Gemma3L3L4ProgressiveAGuardClosedError,
        match="another manifest",
    ):
        loaded.open_guard_after_claim(unrelated)
    assert not loaded.guard_consumed

    guard_view = loaded.preclaim_view("calibration_a_guard")
    receipt = claim_gemma3_l3_l4_progressive_guard(
        protocol_sha256=_sha(4),
        guard_manifest_sha256=guard_view.manifest_sha256,
        challenger_receipt_sha256=_sha(5),
    )
    guard = loaded.open_guard_after_claim(receipt)

    assert guard.role == "calibration_a_guard"
    assert loaded.guard_consumed
    assert loaded.guard_opened
    with pytest.raises(
        Gemma3L3L4ProgressiveAGuardClosedError,
        match="already been consumed",
    ):
        loaded.open_guard_after_claim(receipt)


def test_valid_guard_claim_is_spent_even_if_role_opening_fails(
    tmp_path,
    private_guard_ledger,
) -> None:
    artifact, artifact_path, paths = _freeze(tmp_path / "corpus")
    loaded = load_gemma3_l3_l4_progressive_a_corpus(
        artifact_path,
        role_input_paths=paths,
    )
    guard_view = loaded.preclaim_view("calibration_a_guard")
    receipt = claim_gemma3_l3_l4_progressive_guard(
        protocol_sha256=_sha(11),
        guard_manifest_sha256=guard_view.manifest_sha256,
        challenger_receipt_sha256=_sha(12),
    )
    paths["calibration_a_guard"].write_bytes(b"{broken")

    with pytest.raises(Gemma3L3L4ProgressiveACorpusIntegrityError):
        loaded.open_guard_after_claim(receipt)
    assert loaded.guard_consumed
    assert not loaded.guard_opened
    with pytest.raises(
        Gemma3L3L4ProgressiveAGuardClosedError,
        match="already been consumed",
    ):
        loaded.open_guard_after_claim(receipt)


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "calibration_b",
        "validation",
        "test",
        "calibration_b_prompts",
        "validation_prompts",
        "test_prompts",
    ),
)
def test_role_input_rejects_assessment_fields(
    tmp_path,
    forbidden_field: str,
) -> None:
    paths = _write_role_inputs(tmp_path / "corpus")
    fit_path = paths["calibration_a_fit"]
    payload = json.loads(fit_path.read_bytes())
    payload[forbidden_field] = []
    fit_path.write_bytes(_canonical_bytes(payload))

    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="A-only schema",
    ):
        build_gemma3_l3_l4_progressive_a_corpus_artifact(
            corpus_id="three-role-a-v1",
            profile="pilot",
            tokenizer_contract=_TOKENIZER_CONTRACT,
            role_input_paths=paths,
        )


@pytest.mark.parametrize(
    "forbidden_role",
    ("calibration_b", "validation", "test"),
)
def test_artifact_rejects_assessment_roles(
    tmp_path,
    forbidden_role: str,
) -> None:
    _, artifact_path, paths = _freeze(tmp_path / "corpus")
    payload = json.loads(artifact_path.read_bytes())
    payload["roles"][forbidden_role] = payload["roles"][
        "calibration_a_fit"
    ]
    artifact_path.write_bytes(_canonical_bytes(payload))

    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="exactly the three Calibration-A roles",
    ):
        load_gemma3_l3_l4_progressive_a_corpus(
            artifact_path,
            role_input_paths=paths,
        )


def test_roles_must_be_pairwise_prompt_and_family_disjoint(
    tmp_path,
) -> None:
    shared_prompt_members = {
        "calibration_a_fit": (
            ("Shared prompt.",),
            ("fit-family",),
        ),
        "calibration_a_selection": (
            ("Shared prompt.",),
            ("selection-family",),
        ),
        "calibration_a_guard": (
            ("Guard prompt.",),
            ("guard-family",),
        ),
    }
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="prompt-identity disjoint",
    ):
        _freeze(
            tmp_path / "shared-prompt",
            role_members=shared_prompt_members,
        )

    shared_family_members = {
        "calibration_a_fit": (
            ("Fit prompt.",),
            ("shared-family",),
        ),
        "calibration_a_selection": (
            ("Selection prompt.",),
            ("shared-family",),
        ),
        "calibration_a_guard": (
            ("Guard prompt.",),
            ("guard-family",),
        ),
    }
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="family disjoint",
    ):
        _freeze(
            tmp_path / "shared-family",
            role_members=shared_family_members,
        )


def test_roles_reject_frozen_b_hash_or_family_overlap(
    tmp_path,
    monkeypatch,
) -> None:
    frozen_b = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    forbidden_family = next(iter(frozen_b.values()))
    b_family_members = {
        "calibration_a_fit": (
            ("Fit prompt.",),
            (forbidden_family,),
        ),
        "calibration_a_selection": (
            ("Selection prompt.",),
            ("selection-family",),
        ),
        "calibration_a_guard": (
            ("Guard prompt.",),
            ("guard-family",),
        ),
    }
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="frozen Calibration-B hash-only manifest",
    ):
        _freeze(
            tmp_path / "b-family",
            role_members=b_family_members,
        )

    forbidden_prompt_hash = next(iter(frozen_b))
    original_hash = (
        corpus_module.gemma3_l3_l4_graph_organized_svd_prompt_sha256
    )

    def prompt_hash(prompt: str) -> str:
        if prompt == "Synthetic exact B hash collision.":
            return forbidden_prompt_hash
        return original_hash(prompt)

    monkeypatch.setattr(
        corpus_module,
        "gemma3_l3_l4_graph_organized_svd_prompt_sha256",
        prompt_hash,
    )
    b_hash_members = {
        "calibration_a_fit": (
            ("Synthetic exact B hash collision.",),
            ("fit-family",),
        ),
        "calibration_a_selection": (
            ("Selection prompt.",),
            ("selection-family",),
        ),
        "calibration_a_guard": (
            ("Guard prompt.",),
            ("guard-family",),
        ),
    }
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="frozen Calibration-B hash-only manifest",
    ):
        _freeze(
            tmp_path / "b-hash",
            role_members=b_hash_members,
        )


def test_load_rejects_tampering_profile_mismatch_and_noncanonical_json(
    tmp_path,
) -> None:
    artifact, artifact_path, paths = _freeze(tmp_path / "corpus")

    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="live tokenizer contract",
    ):
        load_gemma3_l3_l4_progressive_a_corpus(
            artifact_path,
            role_input_paths=paths,
            tokenizer_contract={
                **_TOKENIZER_CONTRACT,
                "padding_side": "right",
            },
        )

    role_payload = json.loads(paths["calibration_a_fit"].read_bytes())
    role_payload["profile"] = "full"
    paths["calibration_a_fit"].write_bytes(
        _canonical_bytes(role_payload)
    )
    loaded = load_gemma3_l3_l4_progressive_a_corpus(
        artifact_path,
        role_input_paths=paths,
    )
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="metadata differs",
    ):
        loaded.open_development_role("calibration_a_fit")

    payload = json.loads(artifact_path.read_bytes())
    artifact_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(
        Gemma3L3L4ProgressiveACorpusIntegrityError,
        match="canonical JSON",
    ):
        load_gemma3_l3_l4_progressive_a_corpus(
            artifact_path,
            role_input_paths=paths,
            expected_artifact_sha256=artifact.artifact_sha256,
        )
