from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from fisher_graph import gemma3_chart_native_neighbor_setup as v1_setup
from fisher_graph import (
    gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup as diagnostic_setup,
)
from fisher_graph import gemma3_chart_native_neighbor_v2_setup as v2_setup
from fisher_graph import (
    gemma3_chart_native_neighbor_v2_implementation_qualification as qualification,
)


_ROOT = Path(__file__).resolve().parents[1]
_WORKER = _ROOT / qualification.V2_WORKER_PATH
_V1_SETUP_SOURCE = (
    "src/fisher_graph/gemma3_chart_native_neighbor_setup.py"
)
_QUALIFICATION_SOURCE_NAME = (
    "chart_native_neighbor_v2_implementation_qualification"
)
_V1_SETUP_SOURCE_NAME = "chart_native_neighbor_v1_setup"

# These are observations from the separately sealed A-only result.  They are
# deliberately test-only negative anchors: none may enter the implementation
# qualification identity, directly or through a qualification field.
_DIAGNOSTIC_RESULT_SHA256 = (
    "df1c7884172c46489791385364e536f9f781dc836d949cace524cfcf44101081"
)
_DIAGNOSTIC_RESULT_FILE_SHA256 = (
    "3675ae1c26a673cdb82f6a2e8fdcc4c1b0d1bef3436bce7c1e953b5bb0f33642"
)
_DIAGNOSTIC_DIAGNOSIS_RECEIPT_SHA256 = (
    "2dff197e6e996e4d0783d550e6fc9c9ef0e301927a050a302f69f3dedaecf93a"
)
_DIAGNOSTIC_CANDIDATE_RECEIPT_SHA256 = (
    "254f29d294947c29d8677961eb8153a4198557c60e74e724561267e7b2da287f"
)
_DIAGNOSTIC_HARD_COUNT_VALUES = (
    276,
    1_122,
    915_084,
    167,
    826,
    7_351,
    11_238,
    3_752,
)
_DIAGNOSTIC_HARD_MASS_VALUES = (
    "3f3bf682bfc9cf9f",
    "3f5ca0ecb0edf8e9",
    "3fef286bca1af28f",
    "3f31433a2c59ca4d",
    "3f556c8f5170bd6e",
    "3f7e4308c7d50800",
    "3f874ffe04a845b0",
    "3f6f2036a40e8416",
)


pytestmark = pytest.mark.skipif(
    not _WORKER.is_file(),
    reason="v2 implementation qualification is held until the v2 worker exists",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_qualification_inputs(destination_root: Path) -> None:
    relative_paths = set(qualification._source_inventory().values())
    relative_paths.update(
        {
            qualification.TRUST_ANCHOR_PATH,
            _V1_SETUP_SOURCE,
            str(v1_setup.DEFAULT_OUTPUT),
            str(v2_setup.DEFAULT_OUTPUT),
            str(v2_setup.A_DIAGNOSTIC_SETUP_PATH),
        }
    )
    for relative in sorted(relative_paths):
        _copy_file(_ROOT / relative, destination_root / relative)


def _reseal(value: dict[str, object]) -> bytes:
    value["qualification_sha256"] = qualification._digest(value)
    return qualification.canonical_json_bytes(value)


def _first_party_module_path(
    package_root: Path,
    module_name: str,
) -> Path | None:
    if module_name == "fisher_graph":
        candidate = package_root / "__init__.py"
    elif module_name.startswith("fisher_graph."):
        relative = Path(*module_name.split(".")[1:])
        candidate = (package_root / relative).with_suffix(".py")
        if not candidate.is_file():
            candidate = package_root / relative / "__init__.py"
    else:
        return None
    return candidate if candidate.is_file() else None


def _literal_module_strings(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            result[target.id] = value.value
    return result


def _literal_import_argument(
    argument: ast.expr,
    constants: dict[str, str],
) -> str | None:
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    if isinstance(argument, ast.Name):
        return constants.get(argument.id)
    return None


def _invoked_first_party_source_closure(
    repository_root: Path,
) -> frozenset[str]:
    """Conservatively derive v2's first-party import closure from its AST."""

    package_root = repository_root / "src/fisher_graph"
    pending = ["fisher_graph.gemma3_chart_native_neighbor_v2_experiment"]
    observed: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in observed:
            continue
        source = _first_party_module_path(package_root, module_name)
        if source is None:
            continue
        observed.add(module_name)
        # Importing any submodule executes the package root.  Record that
        # boundary, but do not expand its eager re-export fanout.
        if module_name == "fisher_graph":
            continue
        package_name = (
            module_name
            if source.name == "__init__.py"
            else module_name.rsplit(".", 1)[0]
        )
        tree = ast.parse(source.read_bytes(), filename=str(source))
        constants = _literal_module_strings(tree)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
                continue
            if isinstance(node, ast.ImportFrom):
                try:
                    base = (
                        importlib.util.resolve_name(
                            "." * node.level + (node.module or ""),
                            package_name,
                        )
                        if node.level
                        else (node.module or "")
                    )
                except (ImportError, ValueError):
                    continue
                imported.add(base)
                imported.update(
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                    if alias.name != "*"
                )
                continue
            if not isinstance(node, ast.Call) or not node.args:
                continue
            callback = node.func
            is_import_module = (
                isinstance(callback, ast.Attribute)
                and isinstance(callback.value, ast.Name)
                and callback.value.id == "importlib"
                and callback.attr == "import_module"
            )
            is_dunder_import = (
                isinstance(callback, ast.Name) and callback.id == "__import__"
            )
            if not (is_import_module or is_dunder_import):
                continue
            target = _literal_import_argument(node.args[0], constants)
            if target is None:
                continue
            try:
                imported.add(
                    importlib.util.resolve_name(target, package_name)
                    if target.startswith(".")
                    else target
                )
            except (ImportError, ValueError):
                continue
        pending.extend(
            candidate
            for candidate in imported
            if candidate not in observed
            and _first_party_module_path(package_root, candidate) is not None
        )
    return frozenset(
        str(
            _first_party_module_path(package_root, module_name).relative_to(
                repository_root
            )
        )
        for module_name in observed
    )


def test_build_is_canonical_and_binds_exact_frozen_lineage() -> None:
    state = qualification.build_qualification(repository_root=_ROOT)
    raw = qualification.canonical_json_bytes(state)
    assert json.loads(raw) == state
    assert raw == qualification.canonical_json_bytes(json.loads(raw))
    assert state["qualification_sha256"] == qualification._digest(state)
    assert qualification.validate_qualification(
        deepcopy(state), repository_root=_ROOT
    ) == state

    assert state["frozen_v2_setup"] == {
        "path": str(v2_setup.DEFAULT_OUTPUT),
        "setup_sha256": v2_setup.PINNED_SETUP_SHA256,
        "file_sha256": v2_setup.PINNED_SETUP_FILE_SHA256,
    }
    assert state["predecessor_v1_setup"] == {
        "path": str(v1_setup.DEFAULT_OUTPUT),
        "setup_sha256": v2_setup.V1_SETUP_SHA256,
        "file_sha256": v2_setup.V1_SETUP_FILE_SHA256,
    }
    assert state["frozen_A_K8_diagnostic_setup"] == {
        "path": str(diagnostic_setup.DEFAULT_OUTPUT),
        "setup_sha256": v2_setup.PINNED_A_DIAGNOSTIC_SETUP_SHA256,
        "file_sha256": v2_setup.PINNED_A_DIAGNOSTIC_SETUP_FILE_SHA256,
    }


def test_qualification_source_has_no_duplicate_literal_mapping_keys() -> None:
    tree = ast.parse(
        (_ROOT / qualification.TRUST_ANCHOR_PATH).read_bytes(),
        filename=qualification.TRUST_ANCHOR_PATH,
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        literal_keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        assert len(literal_keys) == len(set(literal_keys))


def test_source_inventory_is_exact_unique_safe_and_byte_authenticated() -> None:
    expected = {
        name: str(receipt["path"])
        for name, receipt in v1_setup.PINNED_WORKER_IMPLEMENTATION_SOURCES.items()
    }
    expected.update(
        {
            _V1_SETUP_SOURCE_NAME: _V1_SETUP_SOURCE,
            _QUALIFICATION_SOURCE_NAME: qualification.TRUST_ANCHOR_PATH,
            "chart_native_neighbor_v2_worker": qualification.V2_WORKER_PATH,
            "chart_native_neighbor_v2_report": (
                "src/fisher_graph/gemma3_chart_native_neighbor_v2_report.py"
            ),
            "chart_native_neighbor_v2_setup": (
                "src/fisher_graph/gemma3_chart_native_neighbor_v2_setup.py"
            ),
            "chart_native_neighbor_A_K8_diagnostic_runner": (
                "src/fisher_graph/"
                "gemma3_chart_native_neighbor_a_occupancy_diagnostic.py"
            ),
            "chart_native_neighbor_A_K8_diagnostic_setup": (
                "src/fisher_graph/"
                "gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup.py"
            ),
            "streaming_shared_chart_atlas_selection_v1": (
                "src/fisher_graph/streaming_shared_chart_atlas_selection_v1.py"
            ),
        }
    )
    assert len(v1_setup.PINNED_WORKER_IMPLEMENTATION_SOURCES) == 60
    assert qualification._source_inventory() == dict(sorted(expected.items()))
    assert len(expected) == len(set(expected)) == len(set(expected.values())) == 68

    state = qualification.build_qualification(repository_root=_ROOT)
    observed = {
        receipt["name"]: {
            "path": receipt["path"],
            "file_sha256": receipt["file_sha256"],
        }
        for receipt in state["source_closure"]
    }
    assert set(observed) == set(expected)
    assert [row["name"] for row in state["source_closure"]] == sorted(expected)
    assert all(
        not Path(relative).is_absolute()
        and ".." not in Path(relative).parts
        and Path(relative).as_posix() == relative
        and relative.startswith("src/fisher_graph/")
        for relative in expected.values()
    )
    assert observed["streaming_shared_chart_atlas_selection_v1"][
        "file_sha256"
    ] == v2_setup.PINNED_SELECTION_SOURCE_SHA256
    for name, relative in expected.items():
        assert observed[name] == {
            "path": relative,
            "file_sha256": _file_sha256(_ROOT / relative),
        }


def test_invoked_first_party_AST_closure_has_no_missing_or_extra_source() -> None:
    derived = _invoked_first_party_source_closure(_ROOT)
    qualified = frozenset(qualification._source_inventory().values())
    assert qualification.TRUST_ANCHOR_PATH in derived
    assert _V1_SETUP_SOURCE in derived
    assert qualification.V2_WORKER_PATH in derived
    assert derived == qualified


def test_reviewed_bootstrap_is_pinned_not_excluded_from_closure() -> None:
    state = qualification.build_qualification(repository_root=_ROOT)
    trust_anchor = state["trust_anchor"]
    assert trust_anchor["path"] == qualification.TRUST_ANCHOR_PATH
    assert trust_anchor["explicitly_reviewed_bootstrap"] is True
    assert trust_anchor["included_in_source_closure"] is True
    assert trust_anchor["coordinated_self_rehash_tamper_proof_claimed"] is False
    assert "explicitly_reviewed_non_self_hashed" not in trust_anchor
    assert "excluded_from_source_closure" not in trust_anchor
    receipts = {
        row["path"]: row["file_sha256"] for row in state["source_closure"]
    }
    assert receipts[qualification.TRUST_ANCHOR_PATH] == _file_sha256(
        _ROOT / qualification.TRUST_ANCHOR_PATH
    )


def test_write_load_round_trip_is_canonical_durable_and_no_clobber(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification.json"
    expected = qualification.write_qualification(
        output, repository_root=_ROOT
    )
    assert output.read_bytes() == qualification.canonical_json_bytes(expected)
    assert qualification.load_qualification(
        output, repository_root=_ROOT
    ) == expected
    with pytest.raises(FileExistsError):
        qualification.write_qualification(output, repository_root=_ROOT)

    occupied = tmp_path / "occupied.json"
    occupied.write_bytes(b"occupied")
    alias = tmp_path / "qualification-link.json"
    alias.symlink_to(occupied)
    with pytest.raises(FileExistsError):
        qualification.write_qualification(alias, repository_root=_ROOT)


def test_default_frozen_qualification_artifact_equals_live_build() -> None:
    artifact = _ROOT / qualification.DEFAULT_OUTPUT
    if not artifact.is_file():
        pytest.skip(
            "freeze gate: default qualification artifact is published only after "
            "the final worker bytes pass independent audit"
        )
    assert artifact.read_bytes() == qualification.canonical_json_bytes(
        qualification.build_qualification(repository_root=_ROOT)
    )
    assert qualification.load_qualification(
        artifact, repository_root=_ROOT
    ) == qualification.build_qualification(repository_root=_ROOT)


def test_publication_failure_rolls_back_destination_and_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "qualification.json"
    fsync_directory_calls = 0
    real_fsync = qualification.os.fsync

    def fail_reload(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise RuntimeError("injected strict-reload failure")

    def record_fsync(descriptor: int) -> None:
        nonlocal fsync_directory_calls
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            fsync_directory_calls += 1
        real_fsync(descriptor)

    monkeypatch.setattr(qualification, "load_qualification", fail_reload)
    monkeypatch.setattr(qualification.os, "fsync", record_fsync)
    with pytest.raises(RuntimeError, match="injected strict-reload failure"):
        qualification.write_qualification(output, repository_root=_ROOT)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    assert fsync_directory_calls >= 2


def test_first_post_link_identity_failure_cleans_owned_output_and_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "qualification.json"
    real_link = qualification.os.link
    real_lstat = Path.lstat
    linked = False
    failed = False

    def record_link(source: Path | str, destination: Path | str) -> None:
        nonlocal linked
        real_link(source, destination)
        if Path(destination) == output:
            linked = True

    def fail_first_post_link_lstat(path: Path) -> os.stat_result:
        nonlocal failed
        if path == output and linked and not failed:
            failed = True
            raise OSError("injected qualification post-link identity failure")
        return real_lstat(path)

    monkeypatch.setattr(qualification.os, "link", record_link)
    monkeypatch.setattr(Path, "lstat", fail_first_post_link_lstat)
    with pytest.raises(OSError, match="post-link identity failure"):
        qualification.write_qualification(output, repository_root=_ROOT)
    assert failed is True
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_post_temp_unlink_identity_failure_uses_owner_inode_for_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "qualification.json"
    real_unlink = qualification.os.unlink
    real_lstat = Path.lstat
    temp_was_unlinked = False
    failed = False

    def record_unlink(path: Path | str) -> None:
        nonlocal temp_was_unlinked
        candidate = Path(path)
        real_unlink(candidate)
        if candidate.name.endswith(".tmp"):
            temp_was_unlinked = True

    def fail_post_temp_unlink_lstat(path: Path) -> os.stat_result:
        nonlocal failed
        if path == output and temp_was_unlinked and not failed:
            failed = True
            raise OSError("injected qualification post-temp-unlink failure")
        return real_lstat(path)

    monkeypatch.setattr(qualification.os, "unlink", record_unlink)
    monkeypatch.setattr(Path, "lstat", fail_post_temp_unlink_lstat)
    with pytest.raises(OSError, match="post-temp-unlink failure"):
        qualification.write_qualification(output, repository_root=_ROOT)
    assert failed is True
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_replaced_qualification_temporary_is_never_unlinked_as_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "qualification.json"
    real_link = qualification.os.link
    real_unlink = qualification.os.unlink
    replaced_temporary: Path | None = None

    def replace_before_link(source: Path | str, destination: Path | str) -> None:
        nonlocal replaced_temporary
        source_path = Path(source)
        real_unlink(source_path)
        source_path.write_bytes(b"attacker replacement qualification temp")
        replaced_temporary = source_path
        real_link(source_path, destination)

    monkeypatch.setattr(qualification.os, "link", replace_before_link)
    with pytest.raises(RuntimeError, match="temporary changed while linking") as error:
        qualification.write_qualification(output, repository_root=_ROOT)
    assert any(
        "temporary cleanup also failed" in note
        for note in getattr(error.value, "__notes__", [])
    )
    assert replaced_temporary is not None
    assert replaced_temporary.read_bytes() == (
        b"attacker replacement qualification temp"
    )
    assert output.read_bytes() == b"attacker replacement qualification temp"


@pytest.mark.parametrize("cleanup_fault", ["unlink", "fsync"])
def test_qualification_rollback_preserves_primary_and_attempts_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fault: str,
) -> None:
    output = tmp_path / "qualification.json"
    real_unlink = qualification.os.unlink
    real_fsync_directory = qualification._fsync_directory
    fsync_calls = 0

    def fail_reload(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise RuntimeError("original qualification reload failure")

    def maybe_fail_unlink(path: Path | str) -> None:
        if cleanup_fault == "unlink" and Path(path) == output:
            raise OSError("injected qualification unlink failure")
        real_unlink(path)

    def maybe_fail_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if cleanup_fault == "fsync" and fsync_calls == 2:
            raise OSError("injected qualification cleanup fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(qualification, "load_qualification", fail_reload)
    monkeypatch.setattr(qualification.os, "unlink", maybe_fail_unlink)
    monkeypatch.setattr(qualification, "_fsync_directory", maybe_fail_fsync)
    with pytest.raises(
        RuntimeError, match="original qualification reload failure"
    ) as error:
        qualification.write_qualification(output, repository_root=_ROOT)
    assert isinstance(error.value.__cause__, BaseExceptionGroup)
    assert fsync_calls >= 2
    assert output.exists() is (cleanup_fault == "unlink")
    assert not list(tmp_path.glob(".*.tmp"))


def test_only_setup_binds_diagnostic_path_and_measurements_do_not_enter_identity(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "repository"
    _copy_qualification_inputs(mirror)
    absent = qualification.build_qualification(repository_root=mirror)
    result_path = mirror / v2_setup.A_DIAGNOSTIC_RESULT_PATH
    assert not result_path.exists()

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(b"not even a valid diagnostic result")
    arbitrary = qualification.build_qualification(repository_root=mirror)
    result_path.write_bytes(b'{"result_sha256":"arbitrary replacement"}')
    replaced = qualification.build_qualification(repository_root=mirror)
    assert absent == arbitrary == replaced

    encoded = qualification.canonical_json_bytes(absent)
    forbidden = (
        str(v2_setup.A_DIAGNOSTIC_RESULT_PATH),
        _DIAGNOSTIC_RESULT_SHA256,
        _DIAGNOSTIC_RESULT_FILE_SHA256,
        _DIAGNOSTIC_DIAGNOSIS_RECEIPT_SHA256,
        _DIAGNOSTIC_CANDIDATE_RECEIPT_SHA256,
    )
    assert all(anchor.encode("utf-8") not in encoded for anchor in forbidden)
    assert qualification.canonical_json_bytes(
        list(_DIAGNOSTIC_HARD_COUNT_VALUES)
    ) not in encoded
    assert qualification.canonical_json_bytes(
        list(_DIAGNOSTIC_HARD_MASS_VALUES)
    ) not in encoded

    measured_source_anchors = tuple(
        anchor.encode("utf-8") for anchor in forbidden[1:]
    )
    result_path_sources: list[str] = []
    for relative in qualification._source_inventory().values():
        source_raw = (mirror / relative).read_bytes()
        assert all(anchor not in source_raw for anchor in measured_source_anchors)
        tree = ast.parse(source_raw, filename=relative)
        if any(
            isinstance(node, ast.Constant) and node.value == forbidden[0]
            for node in ast.walk(tree)
        ):
            result_path_sources.append(relative)
        literal_sequences: list[tuple[object, ...]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
            if isinstance(value, (list, tuple)):
                literal_sequences.append(tuple(value))
        assert _DIAGNOSTIC_HARD_COUNT_VALUES not in literal_sequences
        assert _DIAGNOSTIC_HARD_MASS_VALUES not in literal_sequences
    # The route was prospectively frozen in both immutable setup modules.  It
    # must not be copied into the later worker/validator or artifact fields.
    assert set(result_path_sources) == {
        (
            "src/fisher_graph/"
            "gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup.py"
        ),
        "src/fisher_graph/gemma3_chart_native_neighbor_v2_setup.py",
    }
    policy = absent["qualification_policy"]
    assert policy[
        "canonical_result_path_bound_indirectly_by_v2_setup"
    ] is True
    assert policy[
        "diagnostic_result_file_sha256_excluded_from_identity"
    ] is True
    assert policy["diagnostic_result_sha256_excluded_from_identity"] is True
    assert policy["diagnostic_measurements_excluded_from_identity"] is True


def test_inventory_rejects_duplicate_unsafe_and_trust_anchor_path_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_v1_path = next(
        str(receipt["path"])
        for receipt in v1_setup.PINNED_WORKER_IMPLEMENTATION_SOURCES.values()
    )
    monkeypatch.setitem(
        qualification._EXTRA_SOURCE_PATHS,
        "chart_native_neighbor_v2_worker",
        first_v1_path,
    )
    with pytest.raises(RuntimeError, match="paths collide"):
        qualification._source_inventory()
    monkeypatch.undo()

    first_v1_name = next(iter(v1_setup.PINNED_WORKER_IMPLEMENTATION_SOURCES))
    monkeypatch.setitem(
        qualification._EXTRA_SOURCE_PATHS,
        first_v1_name,
        qualification.V2_WORKER_PATH,
    )
    with pytest.raises(RuntimeError, match="name collides"):
        qualification._source_inventory()
    monkeypatch.undo()

    monkeypatch.setitem(
        qualification._EXTRA_SOURCE_PATHS,
        "chart_native_neighbor_v2_worker",
        "../outside-worker.py",
    )
    with pytest.raises(
        (ValueError, RuntimeError),
        match="path|relative|canonical|unsafe",
    ):
        qualification._source_inventory()
    monkeypatch.undo()

    monkeypatch.setitem(
        qualification._EXTRA_SOURCE_PATHS,
        "chart_native_neighbor_v2_worker",
        qualification.TRUST_ANCHOR_PATH,
    )
    with pytest.raises(RuntimeError, match="paths collide"):
        qualification._source_inventory()


def test_loader_rejects_duplicate_keys_noncanonical_and_self_rehashed_tamper(
    tmp_path: Path,
) -> None:
    state = qualification.build_qualification(repository_root=_ROOT)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"first","schema":"second"}')
    with pytest.raises(ValueError, match="invalid"):
        qualification.load_qualification(duplicate, repository_root=_ROOT)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical|invalid"):
        qualification.load_qualification(noncanonical, repository_root=_ROOT)

    tampered = deepcopy(state)
    tampered["qualification_id"] = "attacker-resealed-qualification"
    tamper_path = tmp_path / "tampered.json"
    tamper_path.write_bytes(_reseal(tampered))
    with pytest.raises(ValueError, match="differs from live source closure"):
        qualification.load_qualification(tamper_path, repository_root=_ROOT)

    bool_for_int = deepcopy(state)
    bool_for_int["format_version"] = True
    bool_for_int_path = tmp_path / "bool-for-int.json"
    bool_for_int_path.write_bytes(_reseal(bool_for_int))
    with pytest.raises(ValueError, match="differs from live source closure"):
        qualification.load_qualification(bool_for_int_path, repository_root=_ROOT)


def test_loader_rejects_duplicate_or_mutated_source_closure_paths(
    tmp_path: Path,
) -> None:
    state = qualification.build_qualification(repository_root=_ROOT)

    duplicate = deepcopy(state)
    duplicate["source_closure"].append(deepcopy(duplicate["source_closure"][0]))
    duplicate_path = tmp_path / "duplicate-source.json"
    duplicate_path.write_bytes(_reseal(duplicate))
    with pytest.raises(ValueError, match="differs from live source closure"):
        qualification.load_qualification(duplicate_path, repository_root=_ROOT)

    path_mutation = deepcopy(state)
    path_mutation["source_closure"][0]["path"] = qualification.V2_WORKER_PATH
    path_mutation_path = tmp_path / "path-mutation.json"
    path_mutation_path.write_bytes(_reseal(path_mutation))
    with pytest.raises(ValueError, match="differs from live source closure"):
        qualification.load_qualification(
            path_mutation_path, repository_root=_ROOT
        )


@pytest.mark.parametrize(
    "relative",
    [
        qualification.V2_WORKER_PATH,
        "src/fisher_graph/gemma3_chart_native_neighbor_v2_report.py",
        qualification.TRUST_ANCHOR_PATH,
        _V1_SETUP_SOURCE,
        "src/fisher_graph/gemma3_chart_native_neighbor_v2_setup.py",
        (
            "src/fisher_graph/"
            "gemma3_chart_native_neighbor_a_occupancy_diagnostic.py"
        ),
        (
            "src/fisher_graph/"
            "gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup.py"
        ),
        "src/fisher_graph/streaming_shared_chart_atlas_selection_v1.py",
        "src/fisher_graph/gemma3_chart_native_neighbor_runtime.py",
    ],
)
def test_strict_reload_rejects_any_pinned_source_byte_mutation(
    tmp_path: Path,
    relative: str,
) -> None:
    mirror = tmp_path / "repository"
    _copy_qualification_inputs(mirror)
    output = mirror / "qualification.json"
    qualification.write_qualification(output, repository_root=mirror)
    source = mirror / relative
    source.write_bytes(source.read_bytes() + b"\n# qualification test mutation\n")
    with pytest.raises(
        ValueError,
        match=(
            "differs from live source closure|source hash drifted|"
            "differs from v1 pin|differs from frozen v2 pin"
        ),
    ):
        qualification.load_qualification(output, repository_root=mirror)


def test_symlinked_artifact_and_symlinked_source_fail_closed(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "qualification.json"
    qualification.write_qualification(artifact, repository_root=_ROOT)
    artifact_link = tmp_path / "qualification-link.json"
    artifact_link.symlink_to(artifact)
    with pytest.raises(ValueError, match="regular file|opened safely"):
        qualification.load_qualification(artifact_link, repository_root=_ROOT)

    mirror = tmp_path / "repository"
    _copy_qualification_inputs(mirror)
    mirrored_worker = mirror / qualification.V2_WORKER_PATH
    mirrored_worker.unlink()
    mirrored_worker.symlink_to(_ROOT / qualification.V2_WORKER_PATH)
    with pytest.raises(ValueError, match="regular file|opened safely"):
        qualification.build_qualification(repository_root=mirror)
