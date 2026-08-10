from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

import fisher_graph.gemma3_layer_multimode_fresh_panels as panels
import fisher_graph.gemma3_layer_multimode_implementation_qualification as qualification
import fisher_graph.gemma3_layer_multimode_setup as setup


_ROOT = Path(__file__).resolve().parents[1]


def _panel_helpers() -> object:
    path = _ROOT / "tests/test_gemma3_layer_multimode_fresh_panels.py"
    spec = importlib.util.spec_from_file_location(
        "_multimode_panel_qualification_helpers",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def panel_bundle() -> panels.AuthoredFreshPanelBundle:
    receipt_bytes = (_ROOT / panels.DEFAULT_RECEIPT).read_bytes()
    receipt = json.loads(receipt_bytes)
    return panels.AuthoredFreshPanelBundle(
        manifest_bytes=(_ROOT / panels.DEFAULT_MANIFEST).read_bytes(),
        role_corpus_items=tuple(
            (role, (_ROOT / panels.DEFAULT_ROLE_CORPORA[role]).read_bytes())
            for role in panels.ORDERED_ROLES
        ),
        receipt_bytes=receipt_bytes,
        bundle_sha256=receipt["bundle_sha256"],
    )


def _copy_regular(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _prepared_root(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> Path:
    root = tmp_path / "repository"
    closure = qualification._discover_source_closure(_ROOT)
    for relative in closure.values():
        _copy_regular(_ROOT / relative, root / relative)
    for relative in qualification._TEST_PATHS:
        _copy_regular(_ROOT / relative, root / relative)
    for relative in (setup.PREDECESSOR_V2_SETUP, setup.PREDECESSOR_V2_REPORT):
        _copy_regular(_ROOT / relative, root / relative)
    setup_path = root / setup.DEFAULT_OUTPUT
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_bytes(setup.render_setup_bytes())
    setup_path.chmod(0o600)
    panel_path = root / panels.DEFAULT_RECEIPT
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_bytes(panel_bundle.receipt_bytes)
    manifest_path = root / panels.DEFAULT_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(panel_bundle.manifest_bytes)
    for role, raw in panel_bundle.role_corpus_items:
        role_path = root / panels.DEFAULT_ROLE_CORPORA[role]
        role_path.parent.mkdir(parents=True, exist_ok=True)
        role_path.write_bytes(raw)
    return root


def test_source_closure_is_transitive_exact_unique_and_includes_entrypoints() -> None:
    closure = qualification._discover_source_closure(_ROOT)
    required = set(qualification._SOURCE_ENTRYPOINTS)
    assert required.issubset(closure)
    assert len(set(closure.values())) == len(closure)
    assert closure[
        "fisher_graph.gemma3_layer_multimode_implementation_qualification"
    ] == qualification.TRUST_ANCHOR_PATH
    for module, relative in closure.items():
        path = _ROOT / relative
        assert path.is_file(), module
        assert not path.is_symlink()


def test_build_is_result_independent_and_has_acyclic_identity_boundary(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    state = qualification.build_qualification(repository_root=root)
    assert state["trust_anchor"][
        "coordinated_source_and_artifact_rehash_tamper_proof_claimed"
    ] is False
    boundary = state["identity_boundary"]
    assert boundary["setup_pins_qualification_file_sha256"] is False
    assert boundary["experiment_report_bytes_excluded"] is True
    assert boundary["coefficient_bundle_bytes_excluded"] is True
    assert boundary["summary_sidecar_bytes_excluded"] is True
    assert boundary["measured_result_values_excluded"] is True
    encoded = qualification.canonical_json_bytes(state)
    assert b"B3_evaluation" not in encoded
    assert b"D3_passed" not in encoded


def test_cycle_guard_rejects_setup_pin_and_panel_result_binding(
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    setup_state = setup.build_setup()
    readiness = dict(setup_state["implementation_readiness"])
    readiness["implementation_qualification_file_sha256"] = "a" * 64
    cycled_setup = {**setup_state, "implementation_readiness": readiness}
    receipt = panels.validate_receipt(
        panels._decode_canonical_mapping(
            panel_bundle.receipt_bytes,
            label="test panel receipt",
        )
    )
    with pytest.raises(ValueError, match="hash cycle"):
        qualification._assert_acyclic_identity_boundary(
            setup_state=cycled_setup,
            panel_state=receipt,
        )
    poisoned_panel = {**receipt, "result_sha256": "b" * 64}
    with pytest.raises(ValueError, match="hash cycle"):
        qualification._assert_acyclic_identity_boundary(
            setup_state=setup_state,
            panel_state=poisoned_panel,
        )


def test_write_load_is_canonical_durable_no_clobber_and_source_sensitive(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = qualification.write_qualification(
        destination,
        repository_root=root,
    )
    raw = destination.read_bytes()
    assert raw == qualification.canonical_json_bytes(state)
    assert qualification.load_qualification(
        destination,
        repository_root=root,
    ) == state
    with pytest.raises(FileExistsError):
        qualification.write_qualification(
            destination,
            repository_root=root,
        )

    report_source = root / (
        "src/fisher_graph/gemma3_layer_multimode_report.py"
    )
    report_source.write_bytes(report_source.read_bytes() + b"\n# drift\n")
    with pytest.raises(ValueError, match="live implementation closure"):
        qualification.load_qualification(
            destination,
            repository_root=root,
        )


def test_loader_rejects_duplicate_noncanonical_and_self_rehashed_tamper(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = qualification.write_qualification(
        destination,
        repository_root=root,
    )
    duplicate = destination.read_bytes()[:-1] + b',"schema":"duplicate"}'
    destination.unlink()
    destination.write_bytes(duplicate)
    with pytest.raises(ValueError, match="invalid"):
        qualification.load_qualification(destination, repository_root=root)

    tampered = dict(state)
    tampered["qualification_policy"] = {
        **tampered["qualification_policy"],
        "strict_live_rebuild_required_by_preflight": False,
    }
    tampered["qualification_sha256"] = qualification._digest(tampered)
    destination.write_bytes(qualification.canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="live implementation closure"):
        qualification.load_qualification(destination, repository_root=root)


def test_publication_link_failure_rolls_back_output_and_temporary(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)

    def fail_link(
        source: object,
        target: object,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        del source, target
        raise OSError("injected link failure")

    monkeypatch.setattr(qualification.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        qualification.write_qualification(
            destination,
            repository_root=root,
        )
    assert not destination.exists()
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


@pytest.mark.parametrize("failure_stage", ("directory_fsync", "strict_reload"))
def test_qualification_post_link_failures_roll_back_owned_output(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if failure_stage == "directory_fsync":
        real_fsync = qualification.os.fsync
        calls = 0

        def fail_first_fsync(descriptor: int) -> None:
            nonlocal calls
            if qualification.stat.S_ISDIR(
                qualification.os.fstat(descriptor).st_mode
            ):
                calls += 1
                if calls == 1:
                    raise OSError("injected qualification directory fsync")
            real_fsync(descriptor)

        monkeypatch.setattr(qualification.os, "fsync", fail_first_fsync)
        expected = "directory fsync"
    else:

        def fail_reload(*_args: object, **_kwargs: object) -> object:
            raise ValueError("injected qualification strict reload")

        monkeypatch.setattr(qualification, "_read_regular_at", fail_reload)
        expected = "strict reload"

    with pytest.raises((OSError, ValueError), match=expected):
        qualification.write_qualification(destination, repository_root=root)
    assert not destination.exists()
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_qualification_post_link_side_effect_is_registered_for_rollback(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    real_link = qualification.os.link

    def link_then_fail(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        real_link(source, target, *args, **kwargs)
        raise OSError("injected post-link qualification fault")

    monkeypatch.setattr(qualification.os, "link", link_then_fail)
    with pytest.raises(OSError, match="post-link qualification fault"):
        qualification.write_qualification(destination, repository_root=root)
    assert not destination.exists()
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_qualification_post_link_stat_fault_rolls_back_registered_target(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    real_stat_at = qualification._stat_at
    target_observations = 0

    def fail_first_post_link_stat(descriptor: int, name: str) -> object:
        nonlocal target_observations
        if name == destination.name:
            target_observations += 1
            if target_observations == 2:
                raise OSError("injected post-link qualification stat fault")
        return real_stat_at(descriptor, name)

    monkeypatch.setattr(qualification, "_stat_at", fail_first_post_link_stat)
    with pytest.raises(OSError, match="post-link qualification stat fault"):
        qualification.write_qualification(destination, repository_root=root)
    assert not destination.exists()
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_qualification_temporary_replacement_is_preserved(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    real_link = qualification.os.link
    real_unlink = qualification.os.unlink
    real_open = qualification.os.open
    replacement = b"unowned qualification temporary replacement"

    def replace_temporary_after_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        real_link(source, target, *args, **kwargs)
        source_descriptor = kwargs["src_dir_fd"]
        real_unlink(source, dir_fd=source_descriptor)
        descriptor = real_open(
            source,
            qualification.os.O_WRONLY
            | qualification.os.O_CREAT
            | qualification.os.O_EXCL,
            0o600,
            dir_fd=source_descriptor,
        )
        try:
            qualification.os.write(descriptor, replacement)
        finally:
            qualification.os.close(descriptor)
        raise OSError("injected qualification temporary replacement")

    monkeypatch.setattr(qualification.os, "link", replace_temporary_after_link)
    with pytest.raises(OSError, match="temporary replacement"):
        qualification.write_qualification(destination, repository_root=root)
    assert not destination.exists()
    preserved = list(destination.parent.glob(f".{destination.name}.*.tmp"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == replacement
    real_unlink(preserved[0])


def test_qualification_strict_reload_rejects_target_inode_replacement(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    real_open = qualification.os.open
    real_unlink = qualification.os.unlink
    replacement = b"unowned qualification target replacement"
    replaced = False

    def replace_when_reload_opens(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == destination.name and not replaced:
            replaced = True
            parent_descriptor = kwargs["dir_fd"]
            real_unlink(path, dir_fd=parent_descriptor)
            replacement_descriptor = real_open(
                path,
                qualification.os.O_WRONLY
                | qualification.os.O_CREAT
                | qualification.os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            try:
                qualification.os.write(replacement_descriptor, replacement)
            finally:
                qualification.os.close(replacement_descriptor)
        return descriptor

    monkeypatch.setattr(qualification.os, "open", replace_when_reload_opens)
    with pytest.raises(ValueError, match="identity or size"):
        qualification.write_qualification(destination, repository_root=root)
    assert destination.read_bytes() == replacement
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_qualification_created_temporary_fstat_failure_uses_safe_fallback(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    real_fstat = qualification.os.fstat
    injected = False

    def fail_first_empty_regular_fstat(descriptor: int) -> object:
        nonlocal injected
        observed = real_fstat(descriptor)
        if (
            not injected
            and qualification.stat.S_ISREG(observed.st_mode)
            and observed.st_size == 0
        ):
            injected = True
            raise OSError("injected qualification temporary fstat failure")
        return observed

    monkeypatch.setattr(qualification.os, "fstat", fail_first_empty_regular_fstat)
    state = qualification.write_qualification(destination, repository_root=root)
    assert injected is True
    assert qualification.load_qualification(
        destination,
        repository_root=root,
    ) == state


def test_qualification_rejects_missing_or_drifted_predecessor_evidence(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    predecessor_setup = root / setup.PREDECESSOR_V2_SETUP
    predecessor_report = root / setup.PREDECESSOR_V2_REPORT
    setup_raw = predecessor_setup.read_bytes()

    predecessor_setup.write_bytes(setup_raw + b"\n")
    with pytest.raises(ValueError, match="predecessor v2 setup"):
        qualification.build_qualification(repository_root=root)
    predecessor_setup.write_bytes(setup_raw)

    report_size = predecessor_report.stat().st_size
    with predecessor_report.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="predecessor v2 report"):
        qualification.build_qualification(repository_root=root)
    with predecessor_report.open("r+b") as stream:
        stream.truncate(report_size)
    missing = predecessor_report.with_name(f"{predecessor_report.name}.missing")
    predecessor_report.rename(missing)
    with pytest.raises(FileNotFoundError):
        qualification.build_qualification(repository_root=root)
    missing.rename(predecessor_report)


def test_qualification_rehashes_manifest_and_every_panel_role_file(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    paths = [
        root / panels.DEFAULT_MANIFEST,
        *(root / panels.DEFAULT_ROLE_CORPORA[role] for role in panels.ORDERED_ROLES),
    ]
    for path in paths:
        raw = path.read_bytes()
        path.write_bytes(raw + b"\n")
        with pytest.raises(panels.FreshPanelIntegrityError):
            qualification.build_qualification(repository_root=root)
        path.write_bytes(raw)
        qualification.build_qualification(repository_root=root)


def test_qualification_link_race_does_not_delete_unowned_target(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    sentinel = b"concurrent unowned target"

    def race_link(
        _source: object,
        target: object,
        *_args: object,
        **kwargs: object,
    ) -> None:
        descriptor = qualification.os.open(
            target,
            qualification.os.O_WRONLY
            | qualification.os.O_CREAT
            | qualification.os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        try:
            qualification.os.write(descriptor, sentinel)
        finally:
            qualification.os.close(descriptor)
        raise FileExistsError(target)

    monkeypatch.setattr(qualification.os, "link", race_link)
    with pytest.raises(FileExistsError):
        qualification.write_qualification(destination, repository_root=root)
    assert destination.read_bytes() == sentinel
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_qualification_rejects_symlink_in_output_parent_ancestry(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    real_parent = root / "real-output"
    real_parent.mkdir()
    real_destination = real_parent / qualification.DEFAULT_OUTPUT.name
    qualification.write_qualification(real_destination, repository_root=root)
    redirected = root / "redirected-output"
    redirected.symlink_to(real_parent, target_is_directory=True)
    destination = redirected / qualification.DEFAULT_OUTPUT.name
    with pytest.raises(ValueError, match="ancestry"):
        qualification.write_qualification(destination, repository_root=root)
    with pytest.raises(ValueError, match="ancestry"):
        qualification.load_qualification(destination, repository_root=root)
    assert list(real_parent.iterdir()) == [real_destination]


def test_qualification_preserves_primary_through_persistent_cleanup_faults(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    destination = root / qualification.DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    real_unlink = qualification.os.unlink

    def fail_link(
        _source: object,
        _target: object,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise OSError("primary qualification link failure")

    def fail_temporary_unlink(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if Path(path).name.endswith(".tmp"):
            raise OSError("persistent qualification temporary unlink failure")
        real_unlink(path, *args, **kwargs)

    real_fsync = qualification.os.fsync

    def fail_parent_fsync(descriptor: int) -> None:
        if qualification.stat.S_ISDIR(
            qualification.os.fstat(descriptor).st_mode
        ):
            raise OSError("persistent qualification parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(qualification.os, "link", fail_link)
    monkeypatch.setattr(qualification.os, "unlink", fail_temporary_unlink)
    monkeypatch.setattr(qualification.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="primary qualification link failure") as caught:
        qualification.write_qualification(destination, repository_root=root)
    assert isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert any("cleanup also failed" in note for note in caught.value.__notes__)
    assert not destination.exists()
    for temporary in destination.parent.glob(f".{destination.name}.*.tmp"):
        real_unlink(temporary)


def test_every_qualified_source_and_test_digest_matches_live_bytes(
    tmp_path: Path,
    panel_bundle: panels.AuthoredFreshPanelBundle,
) -> None:
    root = _prepared_root(tmp_path, panel_bundle)
    state = qualification.build_qualification(repository_root=root)
    for row in (
        state["source_closure"]
        + state["model_free_adversarial_test_closure"]
    ):
        expected = hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
        assert row["file_sha256"] == expected
    assert len(state["source_closure"]) == len(
        {row["path"] for row in state["source_closure"]}
    )
    assert len(state["model_free_adversarial_test_closure"]) == len(
        {row["path"] for row in state["model_free_adversarial_test_closure"]}
    )
