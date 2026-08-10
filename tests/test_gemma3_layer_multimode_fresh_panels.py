from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "src/fisher_graph/gemma3_layer_multimode_fresh_panels.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_gemma3_layer_multimode_fresh_panels_test_module",
    MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
panels = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = panels
_SPEC.loader.exec_module(panels)


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Whitespace tokenizer with a BOS and stable content-derived IDs."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> FakeEncoding:
        ids = []
        if add_special_tokens:
            ids.append(2)
        for word in text.split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            ids.append(3 + int.from_bytes(digest[:4], "big"))
        return FakeEncoding(ids)


def _prior_catalogs() -> tuple[panels.PriorSemanticLineageCatalog, ...]:
    return tuple(
        panels.make_prior_lineage_catalog(generation, [f"{index:064x}"])
        for index, generation in enumerate(panels.PRIOR_GENERATIONS, start=1)
    )


@pytest.fixture(scope="module")
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture(scope="module")
def bundle(tokenizer: FakeTokenizer) -> panels.AuthoredFreshPanelBundle:
    return panels.author_fresh_panel_bundle(
        tokenizer,
        prior_lineage_catalogs=_prior_catalogs(),
    )


def _manifest(
    bundle: panels.AuthoredFreshPanelBundle,
) -> dict[str, object]:
    return panels._decode_canonical_mapping(
        bundle.manifest_bytes,
        label="test manifest",
    )


def _preflight(
    bundle: panels.AuthoredFreshPanelBundle,
) -> panels.OpaquePanelPreflightReceipt:
    return panels.opaque_preflight(
        manifest_bytes=bundle.manifest_bytes,
        receipt_bytes=bundle.receipt_bytes,
        role_corpus_bytes=bundle.role_corpus_bytes,
        expected_manifest_file_sha256=bundle.manifest_file_sha256,
        expected_receipt_file_sha256=bundle.receipt_file_sha256,
        expected_role_file_sha256s=bundle.role_file_sha256s,
    )


def _selection_sha() -> str:
    return "a" * 64


def _stage_sha(index: int) -> str:
    return f"{index:064x}"


def _source(
    bundle: panels.AuthoredFreshPanelBundle,
) -> panels.SequentialPanelSource:
    return panels.SequentialPanelSource.from_frozen_a_selection(
        _preflight(bundle),
        bundle.role_corpus_bytes,
        selection_receipt_sha256=_selection_sha(),
        selected_layer_count=18,
    )


def _prepare_publication_root(root: Path) -> None:
    (root / "examples").mkdir()
    (root / "artifacts/research").mkdir(parents=True)


def _assert_no_published_panel_files(root: Path) -> None:
    for role in panels.ORDERED_ROLES:
        assert not (root / panels.DEFAULT_ROLE_CORPORA[role]).exists()
    assert not (root / panels.DEFAULT_MANIFEST).exists()
    assert not (root / panels.DEFAULT_RECEIPT).exists()


def test_module_is_standard_library_only_and_import_inert() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
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
        "importlib",
        "json",
        "os",
        "pathlib",
        "re",
        "stat",
        "tempfile",
        "types",
        "typing",
        "unicodedata",
        "weakref",
    }
    assert not panels.DEFAULT_MANIFEST.is_absolute()
    assert not panels.DEFAULT_RECEIPT.is_absolute()


def test_authoring_is_deterministic_and_does_not_write(
    tokenizer: FakeTokenizer,
    tmp_path: Path,
) -> None:
    first = panels.author_fresh_panel_bundle(
        tokenizer,
        prior_lineage_catalogs=_prior_catalogs(),
    )
    second = panels.author_fresh_panel_bundle(
        tokenizer,
        prior_lineage_catalogs=_prior_catalogs(),
    )
    assert first == second
    assert first.manifest_file_sha256 == hashlib.sha256(
        first.manifest_bytes
    ).hexdigest()
    assert first.receipt_file_sha256 == hashlib.sha256(
        first.receipt_bytes
    ).hexdigest()
    assert list(tmp_path.iterdir()) == []
    receipt = panels.validate_receipt(
        panels._decode_canonical_mapping(
            first.receipt_bytes,
            label="test receipt",
        )
    )
    assert receipt["claim_boundary"]["model_execution_performed"] is False
    assert receipt["claim_boundary"]["reserve_semantically_opened"] is False


def test_exact_six_family_role_geometry_and_d3_strata(
    bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    manifest = panels.validate_manifest(_manifest(bundle))
    assert tuple(manifest["family_ids"]) == panels.FAMILY_IDS
    assert len(manifest["family_ids"]) == 6
    assert tuple(manifest["ordered_roles"]) == panels.ORDERED_ROLES
    assert manifest["D3_all_12_family_x_hard_long_strata_nonempty"] is True
    assert len(manifest["D3_bootstrap_strata"]) == 12
    assert set(manifest["D3_bootstrap_strata"].values()) == {3, 5}

    for role in panels.EXECUTABLE_ROLES:
        descriptor = manifest["roles"][role]
        assert descriptor["example_count"] == 48
        assert descriptor["examples_per_family"] == 8
        assert descriptor["target_token_counts_per_family"] == [
            96,
            511,
            512,
            513,
            514,
            576,
            640,
            768,
        ]
        assert descriptor["predicted_token_count"] == 24_732
        assert descriptor["hard_long_example_count"] == 18
        assert descriptor["non_hard_long_example_count"] == 30
        assert descriptor["family_example_counts"] == {
            family: 8 for family in panels.FAMILY_IDS
        }
        for member in descriptor["ordered_members"]:
            target = member["target_token_count"]
            if member["hard_long"]:
                assert member["semantic_prefix_token_count"] <= target - 512

    reserve = manifest["roles"][panels.RESERVE_ROLE]
    assert reserve["example_count"] == 12
    assert reserve["examples_per_family"] == 2
    assert reserve["target_token_counts_per_family"] == [576, 768]
    assert reserve["access_class"] == (
        "opaque_bytes_only_never_decode_tokenize_or_model"
    )


def test_pairwise_and_prior_semantic_lineage_firewalls(
    bundle: panels.AuthoredFreshPanelBundle,
    tokenizer: FakeTokenizer,
) -> None:
    manifest = _manifest(bundle)
    audit = manifest["overlap_audit"]
    assert audit["all_intersection_counts_zero"] is True
    assert set(audit["role_pair_intersection_counts"].values()) == {0}
    assert set(
        audit[
            "prior_generation_semantic_lineage_intersection_counts"
        ].values()
    ) == {0}
    assert audit["declared_lineage_parent_lists_all_empty"] is True

    first_lineage = manifest["roles"][panels.B3_ROLE]["ordered_members"][0][
        "semantic_lineage_sha256"
    ]
    colliding = (
        panels.make_prior_lineage_catalog("v1", [first_lineage]),
        _prior_catalogs()[1],
    )
    with pytest.raises(
        panels.FreshPanelIntegrityError,
        match="overlap",
    ):
        panels.author_fresh_panel_bundle(
            tokenizer,
            prior_lineage_catalogs=colliding,
        )


def test_manifest_rejects_pairwise_lineage_and_stratum_tampering(
    bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    manifest = _manifest(bundle)
    b_member = manifest["roles"][panels.B3_ROLE]["ordered_members"][0]
    c_role = manifest["roles"][panels.C3_ROLE]
    c_member = c_role["ordered_members"][0]
    old = c_member["semantic_lineage_sha256"]
    c_member["semantic_lineage_sha256"] = b_member[
        "semantic_lineage_sha256"
    ]
    catalog = c_role["identity_catalog"]["semantic_lineage_sha256s"]
    catalog.remove(old)
    catalog.append(b_member["semantic_lineage_sha256"])
    catalog.sort()
    manifest["manifest_sha256"] = panels._domain_hash(
        panels._MANIFEST_DOMAIN,
        {key: value for key, value in manifest.items() if key != "manifest_sha256"},
    )
    with pytest.raises(
        panels.FreshPanelIntegrityError,
        match="overlap",
    ):
        panels.validate_manifest(manifest)

    manifest = _manifest(bundle)
    manifest["D3_all_12_family_x_hard_long_strata_nonempty"] = False
    manifest["manifest_sha256"] = panels._domain_hash(
        panels._MANIFEST_DOMAIN,
        {key: value for key, value in manifest.items() if key != "manifest_sha256"},
    )
    with pytest.raises(panels.FreshPanelIntegrityError, match="stratum"):
        panels.validate_manifest(manifest)


def test_opaque_preflight_never_decodes_raw_role_corpora(
    bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    original = panels._decode_canonical_mapping

    def audited_decode(encoded: bytes, *, label: str):
        labels.append(label)
        return original(encoded, label=label)

    monkeypatch.setattr(panels, "_decode_canonical_mapping", audited_decode)
    preflight = _preflight(bundle)
    assert labels == ["prompt-free receipt", "prompt-free manifest"]
    assert preflight.reserve_semantically_decoded is False
    assert preflight.role_file_sha256s == bundle.role_file_sha256s

    corrupted = dict(bundle.role_corpus_bytes)
    corrupted[panels.RESERVE_ROLE] += b" "
    with pytest.raises(panels.FreshPanelIntegrityError, match="reserve"):
        panels.opaque_preflight(
            manifest_bytes=bundle.manifest_bytes,
            receipt_bytes=bundle.receipt_bytes,
            role_corpus_bytes=corrupted,
            expected_manifest_file_sha256=bundle.manifest_file_sha256,
            expected_receipt_file_sha256=bundle.receipt_file_sha256,
            expected_role_file_sha256s=bundle.role_file_sha256s,
        )
    assert all("corpus" not in label for label in labels)


def test_preflight_pins_and_original_live_authority_fail_closed(
    bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    with pytest.raises(panels.FreshPanelIntegrityError, match="receipt"):
        panels.opaque_preflight(
            manifest_bytes=bundle.manifest_bytes,
            receipt_bytes=bundle.receipt_bytes,
            role_corpus_bytes=bundle.role_corpus_bytes,
            expected_manifest_file_sha256=bundle.manifest_file_sha256,
            expected_receipt_file_sha256="0" * 64,
            expected_role_file_sha256s=bundle.role_file_sha256s,
        )
    preflight = _preflight(bundle)
    copied = copy.copy(preflight)
    with pytest.raises(panels.FreshPanelFirewallError, match="original live"):
        panels.SequentialPanelSource.from_frozen_a_selection(
            copied,
            bundle.role_corpus_bytes,
            selection_receipt_sha256=_selection_sha(),
            selected_layer_count=18,
        )
    with pytest.raises(panels.FreshPanelFirewallError, match="18"):
        panels.SequentialPanelSource.from_frozen_a_selection(
            preflight,
            bundle.role_corpus_bytes,
            selection_receipt_sha256=_selection_sha(),
            selected_layer_count=17,
        )


def test_sequential_role_order_tokenization_and_terminal_state(
    bundle: panels.AuthoredFreshPanelBundle,
    tokenizer: FakeTokenizer,
) -> None:
    source = _source(bundle)
    assert source.terminal_state == "ready_B3"
    assert source.next_role == panels.B3_ROLE
    assert source.reserve_open_count == 0
    with pytest.raises(panels.FreshPanelFirewallError, match="expected"):
        source.open_role(panels.C3_ROLE)
    with pytest.raises(panels.FreshPanelFirewallError, match="no semantic-open"):
        source.open_role(panels.RESERVE_ROLE)

    for index, role in enumerate(panels.EXECUTABLE_ROLES, start=1):
        opened = source.open_role(role)
        receipt = panels.authenticate_open_role_tokenization(
            opened,
            role=role,
            manifest=source.manifest,
            tokenizer=tokenizer,
        )
        assert receipt.example_count == 48
        assert receipt.predicted_token_count == 24_732
        if role != panels.D3_ROLE:
            with pytest.raises(
                panels.FreshPanelFirewallError,
                match="must be frozen",
            ):
                source.open_role(panels.EXECUTABLE_ROLES[index])
        with pytest.raises(panels.FreshPanelFirewallError, match="stage kind"):
            source.freeze_current(
                "wrong_stage",
                _stage_sha(index),
                terminal_passed=True,
            )
        source.freeze_current(
            panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[role],
            _stage_sha(index),
            terminal_passed=role != panels.D3_ROLE,
        )

    assert source.opened_roles == panels.EXECUTABLE_ROLES
    assert source.reserve_open_count == 0
    assert source.next_role is None
    assert source.terminal_state == "terminal_D3_failed"
    assert tuple(source.freeze_receipts) == panels.EXECUTABLE_ROLES
    with pytest.raises(panels.FreshPanelFirewallError, match="terminal"):
        source.open_role(panels.B3_ROLE)


def test_failed_b3_is_terminal_and_reserve_refusal_precedes_decode(
    bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(bundle)
    source.open_role(panels.B3_ROLE)
    source.freeze_current(
        panels.REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[panels.B3_ROLE],
        _stage_sha(1),
        terminal_passed=False,
    )
    assert source.terminal_state == "terminal_B3_failed"
    assert source.next_role is None
    with pytest.raises(panels.FreshPanelFirewallError, match="terminal"):
        source.open_role(panels.C3_ROLE)

    fresh = _source(bundle)

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("reserve refusal must precede JSON decode")

    monkeypatch.setattr(panels, "_decode_canonical_mapping", forbidden_decode)
    with pytest.raises(panels.FreshPanelFirewallError, match="no semantic-open"):
        fresh.open_role(panels.RESERVE_ROLE)
    with pytest.raises(panels.FreshPanelFirewallError, match="never"):
        fresh.open_reserve()
    assert fresh.reserve_open_count == 0


def test_durable_no_clobber_publication_and_strict_loader(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    published_order: list[Path] = []
    original_publish = panels._publish_one_no_clobber

    def audited_publish(
        target: Path,
        payload: bytes,
        *,
        created: list[tuple[Path, tuple[int, int]]],
    ):
        published_order.append(target.relative_to(tmp_path))
        return original_publish(target, payload, created=created)

    monkeypatch.setattr(panels, "_publish_one_no_clobber", audited_publish)
    published = panels.publish_fresh_panel_bundle(
        bundle,
        repository_root=tmp_path,
    )
    assert published.receipt_file_sha256 == bundle.receipt_file_sha256
    assert published_order[:4] == [
        panels.DEFAULT_ROLE_CORPORA[role] for role in panels.ORDERED_ROLES
    ]
    assert published_order[-2:] == [
        panels.DEFAULT_MANIFEST,
        panels.DEFAULT_RECEIPT,
    ]
    for relative in published_order:
        assert (tmp_path / relative).is_file()

    labels: list[str] = []
    original_decode = panels._decode_canonical_mapping

    def audited_decode(encoded: bytes, *, label: str):
        labels.append(label)
        return original_decode(encoded, label=label)

    monkeypatch.setattr(panels, "_decode_canonical_mapping", audited_decode)
    preflight = panels.check_frozen_panels(
        repository_root=tmp_path,
        expected_receipt_file_sha256=bundle.receipt_file_sha256,
    )
    assert preflight.bundle_sha256 == bundle.bundle_sha256
    assert preflight.reserve_semantically_decoded is False
    assert labels == [
        "published panel receipt",
        "prompt-free receipt",
        "prompt-free manifest",
    ]
    assert all("corpus" not in label for label in labels)

    before = {
        relative: (tmp_path / relative).read_bytes()
        for relative in published_order
    }
    with pytest.raises(FileExistsError, match="clobber"):
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)
    assert before == {
        relative: (tmp_path / relative).read_bytes()
        for relative in published_order
    }


@pytest.mark.parametrize("fault_after", [1, 4, 5, 6])
def test_publication_fault_rolls_back_only_created_files(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    fault_after: int,
) -> None:
    _prepare_publication_root(tmp_path)
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(panels.FreshPanelIntegrityError, match="injected"):
        panels.publish_fresh_panel_bundle(
            bundle,
            repository_root=tmp_path,
            _fault_after_publish_count=fault_after,
    )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    _assert_no_published_panel_files(tmp_path)


def test_publication_requires_preexisting_canonical_parents(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
) -> None:
    with pytest.raises(panels.FreshPanelIntegrityError, match="already exist"):
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)
    _assert_no_published_panel_files(tmp_path)


def test_post_link_lstat_failure_rolls_back_registered_inode(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    target = tmp_path / panels.DEFAULT_ROLE_CORPORA[panels.B3_ROLE]
    original = panels.os.lstat
    fired = False

    def flaky_lstat(path: object):
        nonlocal fired
        if Path(path) == target and target.exists() and not fired:
            fired = True
            raise OSError("injected post-link lstat failure")
        return original(path)

    monkeypatch.setattr(panels.os, "lstat", flaky_lstat)
    with pytest.raises(OSError, match="post-link lstat"):
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)
    assert fired is True
    _assert_no_published_panel_files(tmp_path)


def test_post_link_temp_unlink_failure_rolls_back_registered_inode(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    target = tmp_path / panels.DEFAULT_ROLE_CORPORA[panels.B3_ROLE]
    original = panels.Path.unlink
    original_fsync = panels._fsync_directory
    fired = False
    fsync_target_existence: list[bool] = []

    def flaky_unlink(path: Path, *args: object, **kwargs: object):
        nonlocal fired
        is_target_temp = path.name.startswith(f".{target.name}.tmp-")
        if is_target_temp and target.exists() and not fired:
            fired = True
            raise OSError("injected post-link temp unlink failure")
        return original(path, *args, **kwargs)

    def audited_fsync(path: Path) -> None:
        if path == target.parent:
            fsync_target_existence.append(target.exists())
        original_fsync(path)

    monkeypatch.setattr(panels.Path, "unlink", flaky_unlink)
    monkeypatch.setattr(panels, "_fsync_directory", audited_fsync)
    with pytest.raises(OSError, match="post-link temp unlink"):
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)
    assert fired is True
    assert fsync_target_existence == [True, False]
    _assert_no_published_panel_files(tmp_path)


def test_persistent_temp_unlink_failure_does_not_mask_primary(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    target = tmp_path / panels.DEFAULT_ROLE_CORPORA[panels.B3_ROLE]
    original = panels.Path.unlink
    attempts: list[Path] = []

    def stubborn_unlink(path: Path, *args: object, **kwargs: object):
        is_target_temp = path.name.startswith(f".{target.name}.tmp-")
        if is_target_temp and target.exists():
            attempts.append(path)
            raise OSError(f"persistent temp unlink attempt {len(attempts)}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(panels.Path, "unlink", stubborn_unlink)
    with pytest.raises(OSError, match="persistent temp unlink attempt 1") as caught:
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)

    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert any(
        "temporary file unlink also failed: OSError: "
        "persistent temp unlink attempt 2" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    _assert_no_published_panel_files(tmp_path)
    assert attempts[0].is_file()
    original(attempts[0])


def test_replaced_temp_is_preserved_while_linked_target_rolls_back(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    target = tmp_path / panels.DEFAULT_ROLE_CORPORA[panels.B3_ROLE]
    original_lstat = panels.os.lstat
    original_unlink = panels.Path.unlink
    replacement = b"do not delete replacement"
    replaced_path: Path | None = None

    def replacing_lstat(path: object):
        nonlocal replaced_path
        candidate = Path(path)
        is_target_temp = candidate.name.startswith(f".{target.name}.tmp-")
        if is_target_temp and target.exists() and replaced_path is None:
            original_unlink(candidate)
            candidate.write_bytes(replacement)
            replaced_path = candidate
        return original_lstat(path)

    monkeypatch.setattr(panels.os, "lstat", replacing_lstat)
    with pytest.raises(
        panels.FreshPanelIntegrityError,
        match="temporary identity drifted",
    ) as caught:
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)

    assert replaced_path is not None
    assert replaced_path.read_bytes() == replacement
    assert any(
        "refusing to unlink a replacement publication temporary" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    _assert_no_published_panel_files(tmp_path)
    original_unlink(replaced_path)


def test_post_link_directory_fsync_failure_rolls_back_registered_inode(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    target = tmp_path / panels.DEFAULT_ROLE_CORPORA[panels.B3_ROLE]
    original = panels._fsync_directory
    fired = False

    def flaky_fsync(path: Path) -> None:
        nonlocal fired
        if path == target.parent and target.exists() and not fired:
            fired = True
            raise OSError("injected post-link directory fsync failure")
        original(path)

    monkeypatch.setattr(panels, "_fsync_directory", flaky_fsync)
    with pytest.raises(OSError, match="post-link directory fsync"):
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)
    assert fired is True
    _assert_no_published_panel_files(tmp_path)


def test_rollback_unlink_failure_preserves_primary_and_continues(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    stubborn = tmp_path / panels.DEFAULT_ROLE_CORPORA[panels.RESERVE_ROLE]
    original = panels.Path.unlink
    fired = False

    def flaky_unlink(path: Path, *args: object, **kwargs: object):
        nonlocal fired
        if path == stubborn and not fired:
            fired = True
            raise OSError("injected rollback unlink failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(panels.Path, "unlink", flaky_unlink)
    with pytest.raises(
        panels.FreshPanelIntegrityError,
        match="injected publication fault",
    ) as caught:
        panels.publish_fresh_panel_bundle(
            bundle,
            repository_root=tmp_path,
            _fault_after_publish_count=4,
        )
    assert fired is True
    assert stubborn.exists()
    for role in panels.ORDERED_ROLES[:3]:
        assert not (tmp_path / panels.DEFAULT_ROLE_CORPORA[role]).exists()
    assert any(
        "rollback failure: unlink" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_rollback_fsync_failure_is_not_allowed_to_mask_primary(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_publication_root(tmp_path)
    original = panels._fsync_directory
    call_count = 0

    def flaky_fsync(path: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            raise OSError("injected rollback fsync failure")
        original(path)

    monkeypatch.setattr(panels, "_fsync_directory", flaky_fsync)
    with pytest.raises(
        panels.FreshPanelIntegrityError,
        match="injected publication fault",
    ) as caught:
        panels.publish_fresh_panel_bundle(
            bundle,
            repository_root=tmp_path,
            _fault_after_publish_count=4,
        )
    _assert_no_published_panel_files(tmp_path)
    assert any(
        "rollback failure: fsync" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_strict_loader_rejects_symlink_and_receipt_drift(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
) -> None:
    _prepare_publication_root(tmp_path)
    panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)
    receipt = tmp_path / panels.DEFAULT_RECEIPT
    original = receipt.read_bytes()
    receipt.unlink()
    target = tmp_path / "receipt-target.json"
    target.write_bytes(original)
    receipt.symlink_to(target)
    with pytest.raises(panels.FreshPanelIntegrityError, match="regular"):
        panels.load_frozen_preflight(
            repository_root=tmp_path,
            expected_receipt_file_sha256=bundle.receipt_file_sha256,
        )

    receipt.unlink()
    receipt.write_bytes(original + b" ")
    with pytest.raises(panels.FreshPanelIntegrityError, match="root pin"):
        panels.load_frozen_preflight(
            repository_root=tmp_path,
            expected_receipt_file_sha256=bundle.receipt_file_sha256,
        )


def test_publication_refuses_symlinked_parent_without_external_write(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "artifacts/research").mkdir(parents=True)
    (tmp_path / "examples").symlink_to(outside, target_is_directory=True)
    with pytest.raises(panels.FreshPanelIntegrityError, match="directory"):
        panels.publish_fresh_panel_bundle(bundle, repository_root=tmp_path)
    assert list(outside.iterdir()) == []


def test_cli_authoring_path_requires_pinned_tokenizer_assets(
    tmp_path: Path,
) -> None:
    drifted = tmp_path / "tokenizer.json"
    drifted.write_text("{}", encoding="utf-8")
    args = panels.argparse.Namespace(
        tokenizer_json=str(drifted),
        tokenizer_config=str(drifted),
        special_tokens_map=str(drifted),
        v1_manifest=str(drifted),
        v2_manifest=str(drifted),
    )
    with pytest.raises(panels.FreshPanelIntegrityError, match="tokenizer JSON"):
        panels._build_from_cli(args)


def test_cli_build_is_read_only_and_check_never_opens_reserve(
    bundle: panels.AuthoredFreshPanelBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(panels, "_build_from_cli", lambda _args: bundle)
    assert panels.main(["--build", "--repository-root", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["canonical_files_written"] is False
    assert summary["model_executed"] is False
    assert list(tmp_path.iterdir()) == []

    _prepare_publication_root(tmp_path)
    assert panels.main(["--write", "--repository-root", str(tmp_path)]) == 0
    written = json.loads(capsys.readouterr().out)
    assert written["canonical_files_written"] is True
    assert panels.main(
        [
            "--check",
            "--repository-root",
            str(tmp_path),
            "--expected-receipt-file-sha256",
            bundle.receipt_file_sha256,
        ]
    ) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["reserve_semantically_decoded"] is False
