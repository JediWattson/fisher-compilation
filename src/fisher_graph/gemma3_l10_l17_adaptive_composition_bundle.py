"""Freeze the qualified layer-10 plus adaptive layer-17 graph union.

This command is an artifact-only authority bridge.  It recovers the exact
layer-10 candidate and closed-guard evidence from the already authenticated
legacy composition bundle, binds them to the supplied layer-10 tensor file,
and combines that parent with the selected post-LOFO layer-17 refit.

No corpus role, prompt file, tokenizer, source model, or evaluator is accepted
by this API.  The adaptive result is used only as a source-safe, hash-bound
development assessment.  Consequently the resulting composition remains an
open-development executable and does not authorize heldout, serving, or
whole-model claims.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from .gemma3_layer17_open_a_capacity_evaluation import (
    DEFAULT_ADAPTIVE_A_FIT_CANDIDATE,
    DEFAULT_ADAPTIVE_OPEN_A_OUTPUT,
    load_gemma3_layer17_open_a_capacity_result,
)
from .gemma3_layer17_v8_all_family_refit import (
    GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
    load_gemma3_layer17_v8_all_family_refit_candidate,
)
from .gemma3_modal_graph_composition_bundle import (
    SourceSafeGuardEvidenceRecord,
    load_gemma3_layer10_layer17_composition_bundle,
    save_gemma3_layer10_layer17_composition_bundle,
)
from .gemma3_state_conditioned_modal_graph_artifact import (
    GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA,
    load_gemma3_state_conditioned_modal_graph_candidate,
)


__all__ = [
    "DEFAULT_ADAPTIVE_COMPOSITION_OUTPUT",
    "DEFAULT_LAYER10_CANDIDATE",
    "DEFAULT_LEGACY_COMPOSITION_BUNDLE",
    "build_parser",
    "freeze_gemma3_l10_l17_adaptive_composition_bundle",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_LEGACY_COMPOSITION_BUNDLE = (
    _LOCAL_ROOT / "layer10-layer17-modal-composition-v1.pt"
)
DEFAULT_LAYER10_CANDIDATE = (
    _LOCAL_ROOT / "layer10-shape-flow-gain-dev-v2.pt"
)
DEFAULT_ADAPTIVE_COMPOSITION_OUTPUT = (
    _LOCAL_ROOT / "layer10-layer17-adaptive-composition-open-a-v2.pt"
)

_ADAPTIVE_SCIENTIFIC_ROLE = (
    "already_open_adaptive_development_fixed_capacity_refit"
)
_ADAPTIVE_NEXT_ACTION = "retain_adaptive_candidate_for_open_development_only"


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _legacy_layer10_authority(
    *,
    legacy_bundle_path: Path | str,
    layer10_candidate_path: Path | str,
) -> SourceSafeGuardEvidenceRecord:
    """Authenticate the exact nested layer-10 parent and guard evidence."""

    legacy = load_gemma3_layer10_layer17_composition_bundle(
        legacy_bundle_path
    )
    parents = legacy.get("parents")
    if type(parents) is not tuple or len(parents) != 2:
        raise ValueError("legacy composition parent catalog is invalid")
    matches = tuple(
        parent
        for parent in parents
        if isinstance(parent, Mapping) and parent.get("role") == "layer10"
    )
    if len(matches) != 1:
        raise ValueError("legacy composition must contain one layer10 parent")
    parent = matches[0]
    nested = _mapping(parent.get("candidate"), label="nested layer10 candidate")
    supplied_path = Path(layer10_candidate_path)
    supplied = load_gemma3_state_conditioned_modal_graph_candidate(
        supplied_path
    )
    supplied_file_sha256 = _file_sha256(supplied_path)
    supplied_scientific_sha256 = supplied.get("scientific_payload_sha256")
    if (
        supplied.get("schema") != GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA
        or nested.get("schema") != GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA
        or supplied_path.name != parent.get("candidate_tensor_file")
        or supplied_file_sha256
        != parent.get("candidate_tensor_file_sha256")
        or supplied_scientific_sha256
        != parent.get("candidate_scientific_payload_sha256")
        or supplied_scientific_sha256
        != nested.get("scientific_payload_sha256")
    ):
        raise ValueError(
            "supplied layer10 candidate differs from the legacy authority"
        )
    return SourceSafeGuardEvidenceRecord.from_state_dict(
        _mapping(parent.get("guard_evidence"), label="layer10 guard evidence")
    )


def _adaptive_layer17_authority(
    *,
    layer17_candidate_path: Path | str,
    adaptive_result_path: Path | str,
) -> SourceSafeGuardEvidenceRecord:
    """Bind a passing hardened adaptive result to its exact challenger."""

    candidate_path = Path(layer17_candidate_path)
    candidate = load_gemma3_layer17_v8_all_family_refit_candidate(
        candidate_path
    )
    result_path = Path(adaptive_result_path)
    result = load_gemma3_layer17_open_a_capacity_result(result_path)
    candidate_file_sha256 = _file_sha256(candidate_path)
    candidate_scientific_sha256 = candidate.get("scientific_payload_sha256")

    pair = _mapping(result.get("candidate_pair"), label="adaptive candidate pair")
    challenger_label = pair.get("challenger_label")
    candidates = _mapping(result.get("candidates"), label="adaptive candidates")
    after = _mapping(
        result.get("candidate_tensor_file_sha256s_after"),
        label="adaptive candidate hashes after assessment",
    )
    authorization = _mapping(
        result.get("authorization"),
        label="adaptive authorization",
    )
    safety = _mapping(result.get("safety"), label="adaptive result safety")
    selection = _mapping(
        result.get("adaptive_selection"),
        label="adaptive selection",
    )
    if not isinstance(challenger_label, str) or not challenger_label:
        raise ValueError("adaptive challenger label is invalid")
    challenger = _mapping(
        candidates.get(challenger_label),
        label="adaptive challenger",
    )

    if (
        candidate.get("schema") != GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
        or result.get("scientific_role") != _ADAPTIVE_SCIENTIFIC_ROLE
        or result.get("heldout_confirmation") is not False
        or result.get("candidate_changed") is not False
        or result.get("fit_opened") is not False
        or result.get("selection_opened") is not True
        or result.get("guard_opened") is not False
        or result.get("calibration_b_opened") is not False
        or result.get("validation_opened") is not False
        or result.get("test_opened") is not False
        or safety.get("source_safe") is not True
        or safety.get("contains_prompt_text") is not False
        or safety.get("contains_token_ids") is not False
        or safety.get("contains_logits") is not False
        or safety.get("contains_model_or_candidate_weights") is not False
        or pair.get("comparison_kind") != "fixed_capacity_refit"
        or selection.get("all_required_gates_pass") is not True
        or selection.get("adaptive_candidate_selected") is not True
        or selection.get("next_action") != _ADAPTIVE_NEXT_ACTION
        or challenger.get("candidate_artifact_schema")
        != GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
        or challenger.get("tensor_file") != candidate_path.name
        or challenger.get("tensor_file_sha256") != candidate_file_sha256
        or challenger.get("scientific_payload_sha256")
        != candidate_scientific_sha256
        or after.get(challenger_label) != candidate_file_sha256
        or authorization.get("challenger_tensor_file_sha256")
        != candidate_file_sha256
        or authorization.get("challenger_scientific_payload_sha256")
        != candidate_scientific_sha256
        or authorization.get("selection_access_authorized") is not True
        or authorization.get("heldout_confirmation") is not False
        or authorization.get("serving_authorized") is not False
    ):
        raise ValueError(
            "adaptive result does not authorize the supplied layer17 candidate"
        )

    return SourceSafeGuardEvidenceRecord(
        evidence_file_sha256=_file_sha256(result_path),
        logical_sha256=result["result_sha256"],  # type: ignore[arg-type]
        status="passed",
        assessment_role="open_development_assessment",
        heldout_confirmation=False,
        fresh_validation=False,
    )


def freeze_gemma3_l10_l17_adaptive_composition_bundle(
    *,
    legacy_bundle_path: Path | str = DEFAULT_LEGACY_COMPOSITION_BUNDLE,
    layer10_candidate_path: Path | str = DEFAULT_LAYER10_CANDIDATE,
    layer17_candidate_path: Path | str = DEFAULT_ADAPTIVE_A_FIT_CANDIDATE,
    adaptive_result_path: Path | str = DEFAULT_ADAPTIVE_OPEN_A_OUTPUT,
    output: Path | str = DEFAULT_ADAPTIVE_COMPOSITION_OUTPUT,
) -> dict[str, object]:
    """Freeze one source-safe adaptive composition, refusing overwrite."""

    destination = Path(output)
    report_path = destination.with_suffix(".json")
    if destination.suffix != ".pt":
        raise ValueError("adaptive composition output must use .pt")
    if destination.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite adaptive composition")

    layer10_evidence = _legacy_layer10_authority(
        legacy_bundle_path=legacy_bundle_path,
        layer10_candidate_path=layer10_candidate_path,
    )
    layer17_evidence = _adaptive_layer17_authority(
        layer17_candidate_path=layer17_candidate_path,
        adaptive_result_path=adaptive_result_path,
    )
    return save_gemma3_layer10_layer17_composition_bundle(
        destination,
        layer10_candidate_path=layer10_candidate_path,
        layer10_guard_evidence=layer10_evidence,
        layer17_candidate_path=layer17_candidate_path,
        layer17_guard_evidence=layer17_evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "freeze the qualified layer10 plus selected adaptive layer17 "
            "source-safe graph composition"
        )
    )
    parser.add_argument(
        "--legacy-bundle",
        type=Path,
        default=DEFAULT_LEGACY_COMPOSITION_BUNDLE,
    )
    parser.add_argument(
        "--layer10-candidate",
        type=Path,
        default=DEFAULT_LAYER10_CANDIDATE,
    )
    parser.add_argument(
        "--layer17-candidate",
        type=Path,
        default=DEFAULT_ADAPTIVE_A_FIT_CANDIDATE,
    )
    parser.add_argument(
        "--adaptive-result",
        type=Path,
        default=DEFAULT_ADAPTIVE_OPEN_A_OUTPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ADAPTIVE_COMPOSITION_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = freeze_gemma3_l10_l17_adaptive_composition_bundle(
        legacy_bundle_path=arguments.legacy_bundle,
        layer10_candidate_path=arguments.layer10_candidate,
        layer17_candidate_path=arguments.layer17_candidate,
        adaptive_result_path=arguments.adaptive_result,
        output=arguments.output,
    )
    summary = {
        "output": str(arguments.output),
        "composition_payload_sha256": report["artifact"][  # type: ignore[index]
            "composition_payload_sha256"
        ],
        "report_sha256": report["report_sha256"],
        "source_safe": report["safety"]["source_safe"],  # type: ignore[index]
        "heldout_confirmation": False,
        "serving_authorized": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
