from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from fisher_graph import gemma3_l3_l4_complete_h4_rank320_basis_sidecar as sidecar
from fisher_graph.gemma3_l3_l4_complete_h4_projection import (
    ImmutableFloat64Matrix,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
    gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256,
)


_TEST_BASIS = torch.zeros((320, 640), dtype=torch.float64)
_TEST_BASIS[:, :320] = torch.eye(320, dtype=torch.float64)
_TEST_BASIS = _TEST_BASIS.contiguous()
_TEST_MATRIX_SHA256 = ImmutableFloat64Matrix.from_tensor(
    _TEST_BASIS,
    label="test D320",
).matrix_sha256
_TEST_RUNTIME_SHA256 = _runtime_tensor_sha256(_TEST_BASIS)
_TEST_PROJECTION_ARTIFACT_SHA256 = (
    gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
        _TEST_BASIS,
        projection_rank=320,
        projection_ordering=sidecar.TAIL_INFORMED_PROJECTION_ORDERING,
    )
)


@pytest.fixture
def patched_contract(monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    monkeypatch.setattr(
        sidecar,
        "EXPECTED_BASIS_MATRIX_SHA256",
        _TEST_MATRIX_SHA256,
    )
    monkeypatch.setattr(
        sidecar,
        "EXPECTED_RUNTIME_TENSOR_SHA256",
        _TEST_RUNTIME_SHA256,
    )
    monkeypatch.setattr(
        sidecar,
        "EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256",
        _TEST_PROJECTION_ARTIFACT_SHA256,
    )
    return _TEST_BASIS.clone()


def _local_output(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / ".local-runs"
    return root, root / "google--gemma-3-270m" / "d320.pt"


def test_frozen_parent_and_selected_basis_contract_is_exact() -> None:
    assert sidecar.PARENT_FILE_SHA256 == (
        "30246112335e5fa6ab62f299405d1430c9e182684d4d13e4c6dd6decd0d82a0d"
    )
    assert sidecar.PARENT_REPORT_SHA256 == (
        "dd325816e21a095921b9aefc955e396c8d765e7c193cb3838e305540db4b68e6"
    )
    assert sidecar.SELECTED_ARM == "tail_informed.rank320"
    assert sidecar.TAIL_INFORMED_FIT_ARTIFACT_SHA256 == (
        "0a5328c5c2a5ec7f5dd39799c7cd349b7dc5a5d4f625d0d492ab552df48aecb7"
    )
    assert sidecar.EXPECTED_BASIS_MATRIX_SHA256 == (
        "93de87e8abd6242fa0ca2ab1683607aa7f7b89617c66a5c673d218d04a74097f"
    )
    assert sidecar.EXPECTED_RUNTIME_TENSOR_SHA256 == (
        "e345e71394a65f55f5fd9f96196d830f9dec1df7962216dd4c28f3835793d0cd"
    )
    assert sidecar.EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256 == (
        "5ad248ba6c0e4b5a208f2ad05517649ee728ad95a00d64ba8e95917e3a45fbb4"
    )
    assert sidecar.FIT_TO_PREFIX_LINEAGE_SHA256 == (
        "ad19b3a896c05e4d4f2de36ddd778020e1e4a1cf0733488ae3e919c041e67fad"
    )


def test_build_is_immutable_and_state_is_closed(
    patched_contract: torch.Tensor,
) -> None:
    basis = patched_contract
    artifact = sidecar.build_complete_h4_rank320_basis_sidecar(basis)
    before = artifact.basis_tensor()
    basis.zero_()
    returned = artifact.basis_tensor()
    returned.zero_()

    assert torch.equal(before, _TEST_BASIS)
    assert torch.equal(artifact.basis_tensor(), _TEST_BASIS)
    assert artifact.orthonormal_max_abs_error == 0.0
    assert artifact.basis_matrix_sha256 == _TEST_MATRIX_SHA256
    assert artifact.runtime_tensor_sha256 == _TEST_RUNTIME_SHA256
    assert (
        artifact.projection_basis_artifact_sha256
        == _TEST_PROJECTION_ARTIFACT_SHA256
    )

    state = artifact.state_dict()
    assert set(state) == sidecar._STATE_KEYS
    assert isinstance(state["basis"], torch.Tensor)
    assert state["contains_basis_coefficients"] is True
    for key in (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_logits",
        "contains_activation_rows",
        "contains_gradient_rows",
        "committable",
    ):
        assert state[key] is False
    assert all(
        isinstance(value, (str, int, bool, torch.Tensor))
        for value in state.values()
    )


def test_atomic_save_and_load_roundtrip(
    tmp_path: Path,
    patched_contract: torch.Tensor,
) -> None:
    artifact = sidecar.build_complete_h4_rank320_basis_sidecar(
        patched_contract
    )
    root, output = _local_output(tmp_path)
    receipt = sidecar.save_complete_h4_rank320_basis_sidecar(
        artifact,
        output,
        local_root=root,
    )
    payload = output.read_bytes()
    loaded = sidecar.load_complete_h4_rank320_basis_sidecar(
        output,
        local_root=root,
    )

    assert receipt == {
        "file": str(output),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "logical_artifact_sha256": artifact.artifact_sha256,
        "committable": False,
    }
    assert loaded.artifact_sha256 == artifact.artifact_sha256
    assert torch.equal(loaded.basis_tensor(), _TEST_BASIS)
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_state_loader_rejects_extra_or_changed_metadata(
    patched_contract: torch.Tensor,
) -> None:
    artifact = sidecar.build_complete_h4_rank320_basis_sidecar(
        patched_contract
    )
    extra = artifact.state_dict()
    extra["prompt"] = "forbidden"
    with pytest.raises(ValueError, match="unexpected keyset"):
        sidecar.CompleteH4Rank320BasisSidecar.from_state_dict(extra)

    changed = artifact.state_dict()
    changed["selected_arm"] = "tail_informed.rank319"
    with pytest.raises(ValueError, match="selected_arm differs"):
        sidecar.CompleteH4Rank320BasisSidecar.from_state_dict(changed)

    changed_hash = artifact.state_dict()
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="logical artifact hash differs"):
        sidecar.CompleteH4Rank320BasisSidecar.from_state_dict(changed_hash)


def test_state_loader_rejects_basis_tamper(
    patched_contract: torch.Tensor,
) -> None:
    artifact = sidecar.build_complete_h4_rank320_basis_sidecar(
        patched_contract
    )
    state = artifact.state_dict()
    tampered = state["basis"].clone()
    tampered[0, 0] = 0.5
    state["basis"] = tampered
    with pytest.raises(ValueError, match="basis matrix differs from the parent"):
        sidecar.CompleteH4Rank320BasisSidecar.from_state_dict(state)


def test_paths_are_confined_and_overwrite_is_rejected(
    tmp_path: Path,
    patched_contract: torch.Tensor,
) -> None:
    artifact = sidecar.build_complete_h4_rank320_basis_sidecar(
        patched_contract
    )
    root, output = _local_output(tmp_path)
    outside = tmp_path / "outside.pt"
    with pytest.raises(ValueError, match="remain under .local-runs"):
        sidecar.save_complete_h4_rank320_basis_sidecar(
            artifact,
            outside,
            local_root=root,
        )
    with pytest.raises(ValueError, match=".pt suffix"):
        sidecar.save_complete_h4_rank320_basis_sidecar(
            artifact,
            output.with_suffix(".json"),
            local_root=root,
        )
    with pytest.raises(ValueError, match="root must be named"):
        sidecar.save_complete_h4_rank320_basis_sidecar(
            artifact,
            output,
            local_root=tmp_path / "runs",
        )

    sidecar.save_complete_h4_rank320_basis_sidecar(
        artifact,
        output,
        local_root=root,
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sidecar.save_complete_h4_rank320_basis_sidecar(
            artifact,
            output,
            local_root=root,
        )
    assert output.read_bytes() == before


def test_load_rejects_symlink_and_non_mapping_payload(
    tmp_path: Path,
    patched_contract: torch.Tensor,
) -> None:
    artifact = sidecar.build_complete_h4_rank320_basis_sidecar(
        patched_contract
    )
    root, output = _local_output(tmp_path)
    sidecar.save_complete_h4_rank320_basis_sidecar(
        artifact,
        output,
        local_root=root,
    )
    link = output.with_name("linked.pt")
    link.symlink_to(output)
    with pytest.raises(ValueError, match="symbolic link"):
        sidecar.load_complete_h4_rank320_basis_sidecar(
            link,
            local_root=root,
        )

    invalid = output.with_name("invalid.pt")
    torch.save(("not", "a", "mapping"), invalid)
    with pytest.raises(ValueError, match="state mapping"):
        sidecar.load_complete_h4_rank320_basis_sidecar(
            invalid,
            local_root=root,
        )


def test_save_reauthenticates_before_publication(
    tmp_path: Path,
    patched_contract: torch.Tensor,
) -> None:
    artifact = sidecar.build_complete_h4_rank320_basis_sidecar(
        patched_contract
    )
    object.__setattr__(artifact, "artifact_sha256", "0" * 64)
    root, output = _local_output(tmp_path)
    with pytest.raises(ValueError, match="logical artifact hash differs"):
        sidecar.save_complete_h4_rank320_basis_sidecar(
            artifact,
            output,
            local_root=root,
        )
    assert not output.exists()
