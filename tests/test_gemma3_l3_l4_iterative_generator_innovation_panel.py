from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_panel as panel_module,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_panel import (
    _prompt_blind_forbidden_binding,
)


def _receipt():
    private = panel_module._canonical_bytes(panel_module._role_payload())
    forbidden, _prompts, _families = _prompt_blind_forbidden_binding()
    return panel_module.Gemma3L3L4GeneratorInnovationPanelReceipt(
        plan_sha256=panel_module.FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
        plan_file_sha256=(
            panel_module.FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
        ),
        expanded_fit_corpus_artifact_sha256=(
            panel_module._EXPANDED_FIT_CORPUS_ARTIFACT_SHA256
        ),
        expanded_fit_corpus_file_sha256=(
            panel_module._EXPANDED_FIT_CORPUS_FILE_SHA256
        ),
        prior_occupancy_panel_artifact_sha256=(
            panel_module._PRIOR_OCCUPANCY_PANEL_ARTIFACT_SHA256
        ),
        prior_occupancy_panel_file_sha256=(
            panel_module._PRIOR_OCCUPANCY_PANEL_FILE_SHA256
        ),
        forbidden_assessment_manifest_sha256s=(forbidden,),
        role_input_file_sha256=panel_module._file_sha256(private),
        ordered_prompt_sha256s=panel_module._prompt_sha256s(),
        ordered_family_ids=(
            panel_module.GENERATOR_INNOVATION_FAMILY_SCHEDULE
        ),
    )


def test_panel_is_exactly_sixteen_by_eight_and_receipt_is_prompt_free() -> None:
    assert len(panel_module.GENERATOR_INNOVATION_PROMPTS) == 16
    assert len(set(panel_module._prompt_sha256s())) == 16
    assert len(panel_module.GENERATOR_INNOVATION_FAMILIES) == 8
    assert all(
        panel_module.GENERATOR_INNOVATION_FAMILY_SCHEDULE.count(family)
        == 2
        for family in panel_module.GENERATOR_INNOVATION_FAMILIES
    )

    receipt = _receipt()
    serialized = json.loads(json.dumps(receipt.to_dict()))
    replay = (
        panel_module.Gemma3L3L4GeneratorInnovationPanelReceipt.from_dict(
            serialized
        )
    )
    assert replay == receipt
    encoded = panel_module._canonical_bytes(receipt.to_dict())
    assert all(
        prompt.encode("utf-8") not in encoded
        for prompt in panel_module.GENERATOR_INNOVATION_PROMPTS
    )
    assert receipt.receipt_sha256
    assert receipt.membership_receipt_sha256


def test_private_role_loader_opens_only_exact_receipt_bound_bytes(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    role_path = tmp_path / "panel.private.json"
    role_path.write_bytes(
        panel_module._canonical_bytes(panel_module._role_payload())
    )
    opened = (
        panel_module.load_gemma3_l3_l4_generator_innovation_role_input(
            role_path,
            receipt=receipt,
        )
    )
    assert opened.ordered_prompt_sha256s == receipt.ordered_prompt_sha256s
    role_path.write_bytes(role_path.read_bytes() + b" ")
    with pytest.raises(
        panel_module.Gemma3L3L4GeneratorInnovationPanelIntegrityError,
        match="differs from its receipt",
    ):
        panel_module.load_gemma3_l3_l4_generator_innovation_role_input(
            role_path,
            receipt=receipt,
        )


def test_pair_publication_rolls_back_private_file_if_second_link_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private.json"
    receipt = tmp_path / "receipt.json"
    original_link = os.link
    calls = 0

    def fail_second_link(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second install failure")
        original_link(source, destination)

    monkeypatch.setattr(panel_module.os, "link", fail_second_link)
    with pytest.raises(OSError, match="synthetic"):
        panel_module._publish_pair_once(
            private_path=private,
            private_encoded=b"private",
            receipt_path=receipt,
            receipt_encoded=b"receipt",
        )
    assert not private.exists()
    assert not receipt.exists()
    assert tuple(tmp_path.iterdir()) == ()
