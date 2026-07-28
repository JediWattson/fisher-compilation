from __future__ import annotations

import hashlib
import json
import os

import pytest
import torch

from fisher_graph.gemma3_l3_l4_basis_package import (
    _atomic_publish_bytes,
    _artifact,
    _basis_report,
    _build_package,
    _file_sha256,
    export_gemma3_l3_l4_basis_package,
    load_gemma3_l3_l4_basis_package,
)
import fisher_graph.gemma3_l3_l4_basis_package as basis_module
from fisher_graph.gemma3_l3_l4_spectral_mapping_experiment import (
    Gemma3L3L4SpectralReference,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reference() -> Gemma3L3L4SpectralReference:
    identity = torch.eye(4, dtype=torch.float64)
    return Gemma3L3L4SpectralReference(
        hierarchy_artifact_sha256=_sha("hierarchy"),
        source_model_sha256=_sha("model"),
        base_artifact_file_sha256=_sha("base-file"),
        base_scientific_payload_sha256=_sha("base-payload"),
        refit_artifact_file_sha256=_sha("refit-file"),
        refit_scientific_payload_sha256=_sha("refit-payload"),
        generator_plan_sha256s=(_sha("plan-0"), _sha("plan-1")),
        layer3_factor_sha256=_sha("factor-3"),
        layer4_factor_sha256=_sha("factor-4"),
        x3_mean=torch.tensor([0.1, -0.2, 0.3, -0.4]),
        y3_mean=torch.tensor([-0.4, 0.3, -0.2, 0.1]),
        x4_mean=torch.tensor([0.2, 0.1, -0.1, -0.2]),
        y4_mean=torch.tensor([-0.2, -0.1, 0.1, 0.2]),
        R3=identity,
        P3=identity,
        R4=identity,
        P4=identity,
        S4=torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64),
        x3_covariance=torch.diag(
            torch.tensor([1.0, 4.0, 9.0, 16.0])
        ),
        upstream_mean_prompt_local_kernel=torch.randn(3, 2, 2),
    )


def _save_pair(path, raw, package) -> str:
    torch.save(raw, path)
    file_sha256 = _file_sha256(path)
    report = _basis_report(
        package,
        tensor_file_sha256=file_sha256,
        tensor_file_bytes=path.stat().st_size,
    )
    path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return file_sha256


def test_basis_package_strips_prompt_local_state_and_roundtrips(
    tmp_path,
) -> None:
    reference = _reference()
    package = _build_package(reference)
    raw = _artifact(package)
    path = tmp_path / "basis.pt"
    file_sha256 = _save_pair(path, raw, package)

    restored = load_gemma3_l3_l4_basis_package(
        path,
        expected_file_sha256=file_sha256,
        expected_payload_sha256=package.basis_payload_sha256,
    )

    serialized_names = repr(raw).lower()
    assert raw["schema"].endswith("_v3")
    assert raw["format_version"] == 3
    assert "mean_prompt_local_kernel" not in serialized_names
    assert "edge_jvp_states" not in serialized_names
    assert "contains_activation_rows" in raw["safety"]
    assert raw["safety"]["contains_activation_rows"] is False
    assert raw["safety"]["contains_prompt_identifiers"] is False
    assert torch.equal(restored.R3, package.R3)
    assert torch.equal(restored.S4, package.S4)
    assert torch.equal(restored.x3_covariance, package.x3_covariance)
    assert torch.equal(
        restored.source_mode_standard_deviations(4),
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
    )


def test_basis_loader_rejects_pre_v3_schema(tmp_path) -> None:
    package = _build_package(_reference())
    raw = _artifact(package)
    raw["schema"] = "fisher_graph.gemma3_l3_l4_prompt_blind_basis"
    raw["format_version"] = 2
    path = tmp_path / "basis-v2.pt"
    file_sha256 = _save_pair(path, raw, package)

    with pytest.raises(ValueError, match="schema or version"):
        load_gemma3_l3_l4_basis_package(
            path,
            expected_file_sha256=file_sha256,
        )


def test_basis_loader_rejects_tensor_tampering_even_with_new_file_hash(
    tmp_path,
) -> None:
    reference = _reference()
    package = _build_package(reference)
    raw = _artifact(package)
    raw["tensors"]["R3"][0, 0] += 0.25
    path = tmp_path / "tampered.pt"
    file_sha256 = _save_pair(path, raw, package)

    with pytest.raises(ValueError, match="logical payload hash"):
        load_gemma3_l3_l4_basis_package(
            path,
            expected_file_sha256=file_sha256,
        )


def test_basis_loader_rejects_modal_spectrum_tampering(tmp_path) -> None:
    package = _build_package(_reference())
    raw = _artifact(package)
    raw["tensors"]["S4"][0] *= 0.5
    path = tmp_path / "tampered-spectrum.pt"
    file_sha256 = _save_pair(path, raw, package)

    with pytest.raises(ValueError, match="logical payload hash|spectrum"):
        load_gemma3_l3_l4_basis_package(
            path,
            expected_file_sha256=file_sha256,
        )


def test_basis_loader_rejects_extra_prompt_bearing_fields(tmp_path) -> None:
    reference = _reference()
    package = _build_package(reference)
    raw = _artifact(package)
    raw["prompt_ids"] = ("forbidden",)
    path = tmp_path / "unsafe.pt"
    file_sha256 = _save_pair(path, raw, package)

    with pytest.raises(ValueError, match="fields do not match"):
        load_gemma3_l3_l4_basis_package(
            path,
            expected_file_sha256=file_sha256,
        )


def test_basis_loader_rejects_nested_status_payloads(tmp_path) -> None:
    reference = _reference()
    package = _build_package(reference)
    raw = _artifact(package)
    raw["scientific_status"]["prompt_text"] = "forbidden"
    path = tmp_path / "nested-unsafe.pt"
    file_sha256 = _save_pair(path, raw, package)

    with pytest.raises(ValueError, match="source sites or safety"):
        load_gemma3_l3_l4_basis_package(
            path,
            expected_file_sha256=file_sha256,
        )


def test_excluded_hierarchy_diagnostics_do_not_change_basis_identity() -> None:
    first = _reference()
    second = Gemma3L3L4SpectralReference(
        **{
            **{
                name: getattr(first, name)
                for name in first.__dataclass_fields__
            },
            "hierarchy_artifact_sha256": _sha("different-hierarchy-file"),
            "upstream_mean_prompt_local_kernel": torch.randn(3, 2, 2),
        }
    )

    first_package = _build_package(first)
    second_package = _build_package(second)

    assert first_package.basis_payload_sha256 == (
        second_package.basis_payload_sha256
    )
    assert first_package.binding() == second_package.binding()
    for name, value in first_package.tensors().items():
        assert torch.equal(value, second_package.tensors()[name])


def test_basis_package_rejects_indefinite_covariance() -> None:
    reference = _reference()
    package = _build_package(reference)
    tensors = package.tensors()
    tensors["x3_covariance"][0, 0] = -1.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        type(package)(
            basis_payload_sha256=package.basis_payload_sha256,
            **package.binding(),
            **tensors,
        )

    tensors = package.tensors()
    tensors["x3_covariance"][:2, :2] = torch.tensor(
        [[1.7e308, 1.7e308], [1.7e308, -1.7e308]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="positive semidefinite|eigenspectrum"):
        type(package)(
            basis_payload_sha256=package.basis_payload_sha256,
            **package.binding(),
            **tensors,
        )


def test_basis_package_rejects_nonfinite_or_failed_eigensolvers(
    monkeypatch,
) -> None:
    package = _build_package(_reference())
    real_eigvalsh = torch.linalg.eigvalsh

    monkeypatch.setattr(
        torch.linalg,
        "eigvalsh",
        lambda value: torch.full(
            (value.shape[0],),
            torch.nan,
            dtype=value.dtype,
        ),
    )
    with pytest.raises(ValueError, match="eigenspectrum must be finite"):
        type(package)(
            basis_payload_sha256=package.basis_payload_sha256,
            **package.binding(),
            **package.tensors(),
        )

    def fail_eigensolver(value):
        raise RuntimeError("injected eigensolver failure")

    monkeypatch.setattr(torch.linalg, "eigvalsh", fail_eigensolver)
    with pytest.raises(ValueError, match="could not be computed"):
        type(package)(
            basis_payload_sha256=package.basis_payload_sha256,
            **package.binding(),
            **package.tensors(),
        )
    monkeypatch.setattr(torch.linalg, "eigvalsh", real_eigvalsh)


def test_basis_package_scales_near_float64_limit_before_symmetrizing() -> None:
    reference = _reference()
    maximum = torch.finfo(torch.float64).max
    reference = type(reference)(
        **{
            name: (
                torch.diag(
                    torch.tensor(
                        [maximum, maximum / 2.0, 2.0, 1.0],
                        dtype=torch.float64,
                    )
                )
                if name == "x3_covariance"
                else getattr(reference, name)
            )
            for name in reference.__dataclass_fields__
        }
    )

    package = _build_package(reference)

    assert torch.isfinite(package.x3_covariance).all()


def test_basis_loader_requires_commit_marker_report(tmp_path) -> None:
    package = _build_package(_reference())
    path = tmp_path / "uncommitted.pt"
    torch.save(_artifact(package), path)

    with pytest.raises(FileNotFoundError, match="commit-marker"):
        load_gemma3_l3_l4_basis_package(
            path,
            expected_file_sha256=_file_sha256(path),
        )


def test_publication_helper_never_retracts_an_observable_link_on_fsync_failure(
    tmp_path,
    monkeypatch,
) -> None:
    destination = tmp_path / "published.bin"
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="durability is uncertain"):
        _atomic_publish_bytes(
            destination=destination,
            writer=lambda path: path.write_bytes(b"complete-stage"),
        )

    assert destination.read_bytes() == b"complete-stage"


def test_staging_cleanup_failure_does_not_revoke_a_durable_link(
    tmp_path,
    monkeypatch,
) -> None:
    destination = tmp_path / "durable.bin"
    real_unlink = type(destination).unlink

    def fail_only_hidden_stage(path, *args, **kwargs):
        if path.name.startswith(".durable.bin.") and path.suffix == ".tmp":
            raise OSError("injected staging cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(destination), "unlink", fail_only_hidden_stage)
    identity = _atomic_publish_bytes(
        destination=destination,
        writer=lambda path: path.write_bytes(b"durable"),
    )

    assert destination.read_bytes() == b"durable"
    assert basis_module._path_identity(destination) == identity


def test_staging_cleanup_retries_only_the_owned_inode(
    tmp_path,
    monkeypatch,
) -> None:
    destination = tmp_path / "retried.bin"
    real_unlink = type(destination).unlink
    injected = False

    def fail_hidden_stage_once(path, *args, **kwargs):
        nonlocal injected
        if (
            not injected
            and path.name.startswith(".retried.bin.")
            and path.suffix == ".tmp"
        ):
            injected = True
            raise OSError("injected transient staging cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        type(destination),
        "unlink",
        fail_hidden_stage_once,
    )
    _atomic_publish_bytes(
        destination=destination,
        writer=lambda path: path.write_bytes(b"durable"),
    )

    assert injected
    assert destination.read_bytes() == b"durable"
    assert list(tmp_path.glob(".retried.bin.*.tmp")) == []


def test_owned_cleanup_does_not_unlink_a_replacement_inode(tmp_path) -> None:
    path = tmp_path / "owned"
    path.write_bytes(b"first")
    first_identity = basis_module._path_identity(path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"second")
    os.replace(replacement, path)

    assert first_identity is not None
    assert not basis_module._unlink_if_identity(path, first_identity)
    assert path.read_bytes() == b"second"


def _fake_catalog():
    return object.__new__(basis_module.Gemma3RefitRuntimeCatalog)


def test_export_constructs_quarantined_receipt_before_marker_publication(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "basis.pt"
    monkeypatch.setattr(
        basis_module,
        "load_gemma3_l3_l4_spectral_reference",
        lambda *args, **kwargs: _reference(),
    )

    def fail_receipt(*args, **kwargs):
        assert not output.with_suffix(".json").exists()
        raise RuntimeError("injected receipt construction failure")

    monkeypatch.setattr(basis_module, "_export_receipt", fail_receipt)
    with pytest.raises(RuntimeError, match="receipt construction"):
        export_gemma3_l3_l4_basis_package(
            catalog=_fake_catalog(),
            hierarchy_artifact_path=tmp_path / "ignored.pt",
            hierarchy_artifact_sha256=_sha("ignored"),
            output=output,
        )

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert not output.with_name(f".{output.name}.publish.lock").exists()


def test_export_does_not_fail_after_claim_removal_commit_point(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "basis.pt"
    monkeypatch.setattr(
        basis_module,
        "load_gemma3_l3_l4_spectral_reference",
        lambda *args, **kwargs: _reference(),
    )
    monkeypatch.setattr(
        basis_module,
        "_best_effort_fsync_directory",
        lambda path: False,
    )

    result = export_gemma3_l3_l4_basis_package(
        catalog=_fake_catalog(),
        hierarchy_artifact_path=tmp_path / "ignored.pt",
        hierarchy_artifact_sha256=_sha("ignored"),
        output=output,
    )

    assert result["basis_payload_sha256"] == _build_package(
        _reference()
    ).basis_payload_sha256
    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert not output.with_name(f".{output.name}.publish.lock").exists()


def test_export_retains_reservation_when_marker_durability_is_uncertain(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "basis.pt"
    monkeypatch.setattr(
        basis_module,
        "load_gemma3_l3_l4_spectral_reference",
        lambda *args, **kwargs: _reference(),
    )
    real_fsync = os.fsync
    calls = 0

    def fail_marker_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise OSError("injected marker directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_marker_directory_fsync)
    with pytest.raises(OSError, match="durability is uncertain"):
        export_gemma3_l3_l4_basis_package(
            catalog=_fake_catalog(),
            hierarchy_artifact_path=tmp_path / "ignored.pt",
            hierarchy_artifact_sha256=_sha("ignored"),
            output=output,
        )

    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert output.with_name(f".{output.name}.publish.lock").exists()


def test_loader_uses_the_same_bytes_it_hashed_during_a_path_swap(
    tmp_path,
    monkeypatch,
) -> None:
    reference = _reference()
    package = _build_package(reference)
    path = tmp_path / "basis.pt"
    expected_file = _save_pair(path, _artifact(package), package)
    replacement = tmp_path / "replacement.pt"
    torch.save({"not": "the authenticated basis"}, replacement)
    original_read = basis_module._read_regular_file
    swapped = False

    def swap_after_read(candidate):
        nonlocal swapped
        content = original_read(candidate)
        if candidate == path and not swapped:
            os.replace(replacement, path)
            swapped = True
        return content

    monkeypatch.setattr(basis_module, "_read_regular_file", swap_after_read)
    restored = load_gemma3_l3_l4_basis_package(
        path,
        expected_file_sha256=expected_file,
        expected_payload_sha256=package.basis_payload_sha256,
    )

    assert swapped
    assert restored.basis_payload_sha256 == package.basis_payload_sha256


def test_loader_rejects_an_active_publication_reservation(tmp_path) -> None:
    reference = _reference()
    package = _build_package(reference)
    path = tmp_path / "basis.pt"
    expected_file = _save_pair(path, _artifact(package), package)
    claim = path.with_name(f".{path.name}.publish.lock")
    claim.write_bytes(b"owned")

    with pytest.raises(RuntimeError, match="still in progress"):
        load_gemma3_l3_l4_basis_package(
            path,
            expected_file_sha256=expected_file,
        )
