"""One-shot live finite-NLL selection for the frozen H4 damping candidate.

The campaign has two deliberately separate entry points:

``prepare_gemma_h4_damping_selection_campaign``
    Writes one local private 16-prompt role input and its prompt-free panel
    commitment.  The commitment is derived from the authenticated expanded-fit
    lineage recorded by the materialization report.

``run_gemma_h4_damping_selection_campaign``
    Opens that new input once, tokenizes it into a strict
    :class:`GemmaProgressivePanel`, and performs exactly four matched forwards
    per example: one direct native factorized-model authority pass followed by
    accepted-X4-only, matched-alpha0, and alpha=0.5 bridge executions.

There is no old selection, guard, Calibration-B, fit, or damping-search input
capability in the run boundary.  Logits and tokens remain transient; the
published report contains only scalar summaries and cryptographic identities.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol

import torch
from torch import Tensor

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .gemma3_experiment import (
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
    load_gemma3_graph_organized_svd_candidate,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
    derive_gemma3_l3_l4_supervised_boundary,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .gemma3_l3_l4_h4_damping_materialization import (
    _FORMAT_VERSION as _MATERIALIZATION_FORMAT_VERSION,
    _REPORT_DOMAIN as _MATERIALIZATION_REPORT_DOMAIN,
    _SCHEMA as _MATERIALIZATION_SCHEMA,
    load_gemma_h4_damping_materialization,
)
from .gemma3_l3_l4_h4_damping_selection_panel import (
    FRESH_DAMPING_SELECTION_FAMILIES,
    FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE,
    GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
    Gemma3L3L4H4DampingSelectionPanelArtifact,
    Gemma3L3L4H4DampingSelectionPanelSource,
    freeze_gemma3_l3_l4_h4_damping_selection_panel,
    load_gemma3_l3_l4_h4_damping_expanded_fit_lineage,
    load_gemma3_l3_l4_h4_damping_selection_panel_artifact,
    write_gemma3_l3_l4_h4_damping_selection_panel_artifact,
    write_gemma3_l3_l4_h4_damping_selection_role_input,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    ACCEPTED_X4_ONLY_ARM,
    CHALLENGER_ALPHA0_5_ARM,
    DAMPING_FINITE_NLL_ARM_IDS,
    DAMPING_FINITE_NLL_ARM_SEMANTICS,
    MATCHED_ALPHA0_ARM,
    GemmaH4DampingFiniteNLLArmInput,
    GemmaH4DampingFiniteNLLObservation,
    evaluate_gemma_h4_damping_finite_nll,
    measure_gemma_h4_damping_finite_nll_observation,
)
from .gemma3_l3_l4_h4_incremental_signal_diagnostic import (
    _accepted_x4_artifact,
    _canonical_json_bytes,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveARolePreclaimView,
    Gemma3L3L4ProgressiveARolePrompts,
    gemma3_l3_l4_progressive_a_tokenizer_contract_sha256,
)
from .gemma3_l3_l4_progressive_worker import GemmaProgressivePanel
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaL3L4TwoHeadArtifact,
    _tensor_sha256,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)
from .shadow_fidelity import ShadowFidelityExample


__all__ = [
    "FRESH_DAMPING_SELECTION_PROMPTS",
    "GemmaH4DampingLiveCollection",
    "build_parser",
    "build_preparation_parser",
    "collect_gemma_h4_damping_live_arms",
    "main",
    "prepare_gemma_h4_damping_selection_campaign",
    "preparation_main",
    "run_gemma_h4_damping_selection_campaign",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_h4_damping_selection_campaign"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-campaign:v1\0"
)
_PREPARATION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-preparation:v1\0"
)
_BOUNDARY_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-boundary:v1\0"
)
_COLLECTION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-live-collection:v1\0"
)
_CLAIM_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-claim:v1\0"
)
_CLAIM_IDENTITY_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-claim-identity:v1\0"
)
_CLAIM_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_h4_damping_selection_claim"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FACTORIZED_SCOPE = "factorized_refit"
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_CLAIM_LEDGER_ROOT = Path(__file__).resolve().parents[2] / _LOCAL_ROOT
_DEFAULT_PRIVATE_SELECTION_INPUT = (
    _LOCAL_ROOT
    / "progressive-a-h4-damping-selection-v1.private.json"
)
_DEFAULT_SELECTION_PANEL = (
    _LOCAL_ROOT / "progressive-a-h4-damping-selection-v1.panel.json"
)
_DEFAULT_MATERIALIZATION_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-damping-materialization-v1.report.json"
)
_DEFAULT_ALPHA0_CANDIDATE = (
    _LOCAL_ROOT
    / "progressive-a-h4-damping-materialization-v1.report.alpha0.candidate.pt"
)
_DEFAULT_ALPHA0_5_CANDIDATE = (
    _LOCAL_ROOT
    / "progressive-a-h4-damping-materialization-v1.report.alpha0_5.candidate.pt"
)
_DEFAULT_ACCEPTED_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-projected-state-v6.campaign.json"
)
_DEFAULT_ACCEPTED_CANDIDATE = (
    _LOCAL_ROOT
    / "progressive-a-h4-projected-state-v6.campaign.candidate.pt"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-h4-damping-selection-v1.report.json"
)


# The two rounds intentionally probe the same eight capabilities with
# different surface forms.  They are fixed before the one-shot selection is
# opened and are not generated from any observed candidate result.
FRESH_DAMPING_SELECTION_PROMPTS = (
    (
        "Execute this algorithm exactly and report the final ordered state. "
        "Start with queue [4, 1, 3, 2]. Remove the front item, append twice "
        "that item, rotate the queue left by two positions, then replace each "
        "remaining odd value n with n + 5. Show the intermediate queue after "
        "each operation."
    ),
    (
        "A service slowed after a deployment. The trace shows database time "
        "unchanged at 18 ms, serialization rising from 7 ms to 46 ms, and "
        "network time falling from 22 ms to 15 ms. Attribute the regression "
        "to the evidence-supported component and state which observations "
        "rule out the two alternatives."
    ),
    (
        "Determine whether this argument is formally valid: Every sealed "
        "record is immutable. Some audit entries are sealed records. No "
        "immutable object can be edited in place. Therefore some audit "
        "entries cannot be edited in place. Give a compact derivation or a "
        "countermodel."
    ),
    (
        "Apply the policy including its exception: requests over 500 units "
        "require manager approval, except verified emergency replacements "
        "may proceed up to 800 units when an incident ticket is open. A "
        "verified replacement requests 730 units with an open incident "
        "ticket and no manager approval. Decide whether it may proceed and "
        "identify the controlling clause."
    ),
    (
        "Extract a normalized record from this note: 'On 14 March, sensor "
        "R-17 in zone west reported 38.6 C at 09:42 UTC; operator Mina K. "
        "acknowledged alarm A55 and scheduled inspection for 11:15 UTC.' "
        "Return fields date, sensor, zone, temperature_c, observed_time_utc, "
        "operator, alarm_id, and inspection_time_utc."
    ),
    (
        "Decide whether the expressions are symbolically equivalent for all "
        "x where both are defined: (x^2 - 9)/(x - 3) and x + 3. Distinguish "
        "algebraic simplification from equality of the original domains."
    ),
    (
        "A pump moves 2.4 liters per minute for 35 seconds into a container "
        "that already holds 180 milliliters. Compute the final volume in "
        "milliliters, showing the time and volume conversions with units."
    ),
    (
        "Assess entailment: 'Whenever the archive is unlocked, either the "
        "curator or the deputy is present. The archive is unlocked, and the "
        "curator is not present.' Does it follow that the deputy is present? "
        "Explain using only the stated propositions."
    ),
    (
        "Run the procedure and return the final mapping. Begin with "
        "{a: 3, b: 5, c: 8}. In alphabetical key order, add the current value "
        "of the previous key to each later value, then delete keys whose final "
        "value is divisible by four, and finally insert d equal to the sum of "
        "the retained values. Show each update."
    ),
    (
        "Three facts accompany a failed batch: input row count stayed 12,000; "
        "validation rejects rose from 4 to 611; storage writes dropped by "
        "exactly 607; CPU utilization was unchanged. Identify the directly "
        "supported failure stage and separate direct evidence from an "
        "inference."
    ),
    (
        "Test the validity of this inference: If a token is expired then it "
        "is rejected. This token was rejected. Therefore it was expired. "
        "Name the logical form and provide one concrete counterexample if the "
        "conclusion does not follow."
    ),
    (
        "Use both rule priority and exception scope: bronze accounts may "
        "export at most 20 files daily; legal-hold exports are exempt from "
        "that limit only when signed by compliance; a global malware block "
        "overrides every exemption. A signed legal-hold export requests 31 "
        "files and no malware block is active. Decide the outcome and cite "
        "the precedence chain."
    ),
    (
        "Convert this incident description into a nested structure: "
        "'Case C204 affects services search and billing; severity is high; "
        "owners are I. Rao and J. Chen; mitigation cache-bypass began at "
        "16:08 UTC; next review is 16:40 UTC.' Return case_id, severity, "
        "services[], owners[], and mitigation{name, started_utc, review_utc}."
    ),
    (
        "For nonzero a and b, determine whether "
        "(a/b + b/a) / (1/a + 1/b) is always equal to (a^2 + b^2)/(a + b). "
        "Simplify carefully and state every additional domain restriction "
        "needed by either original expression."
    ),
    (
        "A vehicle travels 1.8 kilometers in 2 minutes 15 seconds. Compute "
        "its average speed in meters per second and kilometers per hour. "
        "Carry units through every conversion and round only the final values "
        "to two decimals."
    ),
    (
        "Evaluate whether the conclusion follows: 'No encrypted backup is "
        "readable without a key. At least one nightly backup is encrypted. "
        "Every nightly backup is stored off-site.' Conclusion: at least one "
        "off-site object is not readable without a key. Give the quantified "
        "reasoning."
    ),
)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _assert_scalar_hash_only(value: object, *, path: str = "report") -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} contains a tensor")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite float")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has a non-string key")
            _assert_scalar_hash_only(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_scalar_hash_only(nested, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported payload {type(value)!r}")


def _selection_claim_path(
    *,
    panel_artifact_sha256: str,
    private_input_file_sha256: str,
) -> Path:
    """Return the path-independent claim for one authenticated panel identity."""

    identity_sha256 = _sha256(
        _CLAIM_IDENTITY_DOMAIN,
        {
            "panel_artifact_sha256": _require_sha256(
                panel_artifact_sha256,
                label="claim identity panel artifact",
            ),
            "private_input_file_sha256": _require_sha256(
                private_input_file_sha256,
                label="claim identity private input file",
            ),
        },
    )
    return _CLAIM_LEDGER_ROOT / (
        f".h4-damping-selection-{identity_sha256}.consumed.claim.json"
    )


def _selection_claim_payload(
    *,
    panel_artifact_sha256: str,
    panel_file_sha256: str,
    private_input_file_sha256: str,
    materialization_report_sha256: str,
    materialization_report_file_sha256: str,
    artifacts: Mapping[str, _ArtifactLike],
) -> dict[str, object]:
    """Bind everything whose results this irreversible opening may reveal."""

    if set(artifacts) != set(DAMPING_FINITE_NLL_ARM_IDS):
        raise ValueError("selection claim candidate arm IDs differ")
    payload: dict[str, object] = {
        "schema": _CLAIM_SCHEMA,
        "format_version": 1,
        "semantics": {
            "operation": "durable_consume_before_prompt_read",
            "cross_process": True,
            "retry_after_failure_allowed": False,
            "alternate_output_rerun_allowed": False,
            "identity_key": (
                "panel_artifact_sha256_plus_private_input_file_sha256"
            ),
            "private_input_path_affects_identity": False,
        },
        "panel": {
            "artifact_sha256": _require_sha256(
                panel_artifact_sha256,
                label="claim panel artifact",
            ),
            "file_sha256": _require_sha256(
                panel_file_sha256,
                label="claim panel file",
            ),
            "private_input_file_sha256": _require_sha256(
                private_input_file_sha256,
                label="claim private input file",
            ),
        },
        "materialization": {
            "report_sha256": _require_sha256(
                materialization_report_sha256,
                label="claim materialization report",
            ),
            "report_file_sha256": _require_sha256(
                materialization_report_file_sha256,
                label="claim materialization report file",
            ),
        },
        "candidates": {
            arm_id: {
                "artifact_sha256": _require_sha256(
                    artifact.artifact_sha256,
                    label=f"claim {arm_id} artifact",
                ),
                "execution_sha256": _require_sha256(
                    artifact.execution_sha256,
                    label=f"claim {arm_id} execution",
                ),
                "runtime_binding_sha256": _require_sha256(
                    artifact.runtime_binding_sha256,
                    label=f"claim {arm_id} runtime binding",
                ),
                "bridge_binding_sha256": _require_sha256(
                    artifact.bridge_binding_sha256,
                    label=f"claim {arm_id} bridge binding",
                ),
            }
            for arm_id, artifact in sorted(artifacts.items())
        },
    }
    payload["claim_sha256"] = _sha256(_CLAIM_DOMAIN, payload)
    _assert_scalar_hash_only(payload, path="selection claim")
    return payload


def _create_selection_claim(
    payload: Mapping[str, object],
) -> tuple[Path, str]:
    """Irreversibly claim a private panel with one O_EXCL publication."""

    claim = dict(payload)
    observed_claim_sha256 = claim.pop("claim_sha256", None)
    if (
        claim.get("schema") != _CLAIM_SCHEMA
        or claim.get("format_version") != 1
        or observed_claim_sha256
        != _sha256(_CLAIM_DOMAIN, claim)
    ):
        raise ValueError("selection claim payload integrity differs")
    claim["claim_sha256"] = observed_claim_sha256
    _assert_scalar_hash_only(claim, path="selection claim")
    panel = _mapping(claim.get("panel"), label="selection claim panel")
    claim_path = _selection_claim_path(
        panel_artifact_sha256=str(panel.get("artifact_sha256")),
        private_input_file_sha256=str(
            panel.get("private_input_file_sha256")
        ),
    )
    if _CLAIM_LEDGER_ROOT.is_symlink() or not _CLAIM_LEDGER_ROOT.is_dir():
        raise ValueError(
            "fixed selection claim ledger must be a regular directory"
        )
    encoded = (
        json.dumps(
            claim,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            "fresh selection input already has a durable consume claim"
        ) from error
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("durable claim write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        directory = os.open(
            claim_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # A partial claim still consumes the one-shot capability.  Removing it
        # would create a result-dependent retry channel.
        raise
    return claim_path, _file_sha256(claim_path)


def _read_materialization_report(
    path: Path | str,
    *,
    expected_report_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError("materialization report must be a regular file")
    if _file_sha256(source) != _require_sha256(
        expected_file_sha256,
        label="materialization report file",
    ):
        raise ValueError("materialization report file hash differs")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("materialization report must contain an object")
    bound = dict(raw)
    observed = bound.pop("report_sha256", None)
    if (
        raw.get("schema") != _MATERIALIZATION_SCHEMA
        or raw.get("format_version") != _MATERIALIZATION_FORMAT_VERSION
        or observed
        != _require_sha256(
            expected_report_sha256,
            label="materialization report",
        )
        or observed != _sha256(_MATERIALIZATION_REPORT_DOMAIN, bound)
    ):
        raise ValueError("materialization report integrity differs")
    safety = _mapping(
        raw.get("safety"),
        label="materialization safety",
    )
    contract = _mapping(
        raw.get("contract"),
        label="materialization contract",
    )
    expected_safety = {
        "fit_only_inputs_opened": True,
        "selection_input_capability_present": False,
        "selection_role_opened": False,
        "guard_input_capability_present": False,
        "guard_role_opened": False,
        "calibration_b_loader_present": False,
        "calibration_b_opened": False,
        "alternate_alpha_fallback_present": False,
        "prompt_text_in_report": False,
        "token_ids_in_report": False,
        "logits_in_report": False,
        "activation_rows_in_report": False,
        "gradient_rows_in_report": False,
        "fit_sequences_in_report": False,
        "coefficient_tensors_in_report": False,
        "model_weights_in_artifacts": False,
        "selection_claim": False,
        "compression_claim": False,
        "latency_claim": False,
    }
    if (
        contract.get("candidate_alphas") not in ((0.0, 0.5), [0.0, 0.5])
        or contract.get("selected_alpha") != 0.5
        or contract.get("alpha_search_performed") is not False
        or dict(safety) != expected_safety
    ):
        raise ValueError("materialization promotion boundary differs")
    _assert_scalar_hash_only(raw)
    return raw


def _materialization_recollection(
    report: Mapping[str, object],
) -> Mapping[str, object]:
    recollection = _mapping(
        report.get("recollection"),
        label="materialization recollection",
    )
    for name in (
        "corpus_artifact_sha256",
        "fit_manifest_sha256",
        "fit_binding_sha256",
        "factorized_model_sha256",
        "factorized_execution_sha256",
    ):
        _require_sha256(recollection.get(name), label=name)
    return recollection


def prepare_gemma_h4_damping_selection_campaign(
    *,
    expanded_corpus_artifact_path: Path | str,
    materialization_report_path: Path | str,
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    private_selection_input_output: Path | str = (
        _DEFAULT_PRIVATE_SELECTION_INPUT
    ),
    selection_panel_output: Path | str = _DEFAULT_SELECTION_PANEL,
) -> dict[str, object]:
    """Write the only new private role input and prompt-free panel artifact."""

    private_path = Path(private_selection_input_output)
    panel_path = Path(selection_panel_output)
    if private_path == panel_path or private_path.exists() or panel_path.exists():
        raise FileExistsError("refusing to overwrite selection preparation")
    report = _read_materialization_report(
        materialization_report_path,
        expected_report_sha256=expected_materialization_report_sha256,
        expected_file_sha256=(
            expected_materialization_report_file_sha256
        ),
    )
    recollection = _materialization_recollection(report)
    lineage = load_gemma3_l3_l4_h4_damping_expanded_fit_lineage(
        expanded_corpus_artifact_path,
        expected_expanded_corpus_artifact_sha256=str(
            recollection["corpus_artifact_sha256"]
        ),
        fit_binding_sha256=str(recollection["fit_binding_sha256"]),
    )
    if lineage.fit_manifest_sha256 != recollection["fit_manifest_sha256"]:
        raise ValueError(
            "materialization and expanded-fit lineage manifests differ"
        )
    private_written = False
    try:
        private_file_sha256 = (
            write_gemma3_l3_l4_h4_damping_selection_role_input(
                private_path,
                prompts=FRESH_DAMPING_SELECTION_PROMPTS,
            )
        )
        private_written = True
        artifact = freeze_gemma3_l3_l4_h4_damping_selection_panel(
            expanded_fit_lineage=lineage,
            selection_input_path=private_path,
        )
        panel_file_sha256 = (
            write_gemma3_l3_l4_h4_damping_selection_panel_artifact(
                panel_path,
                artifact,
            )
        )
    except BaseException:
        if private_written:
            private_path.unlink(missing_ok=True)
        panel_path.unlink(missing_ok=True)
        raise

    payload: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_l3_l4_h4_damping_selection_preparation"
        ),
        "format_version": 1,
        "panel_id": GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
        "expanded_fit_lineage_receipt_sha256": lineage.receipt_sha256,
        "materialization_report_sha256": report["report_sha256"],
        "private_selection_input": {
            "path": str(private_path),
            "file_sha256": private_file_sha256,
            "prompt_count": len(FRESH_DAMPING_SELECTION_PROMPTS),
            "contains_prompt_text": True,
            "local_private": True,
            "committable": False,
        },
        "prompt_free_panel": {
            "path": str(panel_path),
            "file_sha256": panel_file_sha256,
            "artifact_sha256": artifact.artifact_sha256,
            "manifest_sha256": artifact.manifest_sha256,
            "membership_receipt_sha256": (
                artifact.membership_receipt_sha256
            ),
            "family_ids": artifact.family_ids,
            "contains_prompt_text": False,
        },
        "safety": {
            "old_selection_input_opened": False,
            "guard_input_opened": False,
            "calibration_b_input_opened": False,
            "candidate_result_observed_before_freeze": False,
        },
    }
    payload["preparation_sha256"] = _sha256(
        _PREPARATION_DOMAIN,
        payload,
    )
    _assert_scalar_hash_only(payload)
    return payload


@dataclass(frozen=True, slots=True)
class _LoadedMaterialization:
    report: Mapping[str, object]
    accepted_x4: GemmaL3L4TwoHeadArtifact
    matched_alpha0: GemmaL3L4TwoHeadArtifact
    challenger_alpha0_5: GemmaL3L4TwoHeadArtifact


def _load_materialization(
    *,
    materialization_report_path: Path | str,
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    matched_alpha0_candidate_path: Path | str,
    challenger_alpha0_5_candidate_path: Path | str,
    accepted_x4_report_path: Path | str,
    accepted_x4_candidate_path: Path | str,
    expected_accepted_x4_candidate_file_sha256: str,
) -> _LoadedMaterialization:
    report = _read_materialization_report(
        materialization_report_path,
        expected_report_sha256=expected_materialization_report_sha256,
        expected_file_sha256=(
            expected_materialization_report_file_sha256
        ),
    )
    materialization, strict_report = (
        load_gemma_h4_damping_materialization(
            materialization_report_path,
            expected_report_sha256=(
                expected_materialization_report_sha256
            ),
            expected_report_file_sha256=(
                expected_materialization_report_file_sha256
            ),
        )
    )
    if _canonical_json_bytes(strict_report) != _canonical_json_bytes(report):
        raise RuntimeError("strict materialization load changed report payload")
    files = _mapping(report.get("files"), label="materialization files")
    for supplied, key in (
        (matched_alpha0_candidate_path, "matched_alpha0"),
        (challenger_alpha0_5_candidate_path, "challenger_alpha0_5"),
    ):
        record = _mapping(files.get(key), label=f"{key} tensor file")
        embedded = Path(str(record.get("tensor_file")))
        if embedded.resolve() != Path(supplied).resolve():
            raise ValueError(
                f"{key} path differs from the authenticated materialization"
            )
    _accepted_report, accepted, provenance = _accepted_x4_artifact(
        report_path=accepted_x4_report_path,
        candidate_path=accepted_x4_candidate_path,
        expected_candidate_file_sha256=(
            expected_accepted_x4_candidate_file_sha256
        ),
    )
    recollection = _materialization_recollection(report)
    frozen_provenance = _mapping(
        recollection.get("accepted_x4_provenance"),
        label="materialization accepted X4 provenance",
    )
    if _canonical_json_bytes(provenance) != _canonical_json_bytes(
        frozen_provenance
    ):
        raise ValueError(
            "accepted X4 provenance differs from materialization"
        )
    accepted_metadata = _mapping(
        _mapping(
            report.get("artifacts"),
            label="materialization artifacts",
        ).get("accepted_x4_only"),
        label="accepted X4 metadata",
    )
    accepted.validate_integrity()
    if (
        accepted.artifact_sha256
        != accepted_metadata.get("artifact_sha256")
        or accepted.execution_sha256
        != accepted_metadata.get("execution_sha256")
        or accepted.runtime_binding_sha256
        != accepted_metadata.get("runtime_binding_sha256")
        or accepted.artifact_sha256
        != materialization.alpha0_artifact.parent_artifact_sha256
    ):
        raise ValueError("accepted X4 materialization binding differs")
    return _LoadedMaterialization(
        report=report,
        accepted_x4=accepted,
        matched_alpha0=materialization.alpha0_artifact,
        challenger_alpha0_5=materialization.alpha0_5_artifact,
    )


class _ArtifactLike(Protocol):
    artifact_sha256: str
    execution_sha256: str
    runtime_binding_sha256: str
    parent_artifact_sha256: str
    bridge_binding_sha256: str
    live_model_sha256: str
    adapter_execution_sha256: str

    def validate_integrity(self) -> None: ...

    def head(self, site: str) -> object | None: ...


class _AdapterLike(Protocol):
    def model_fingerprint(self) -> str: ...

    def execution_fingerprint(self) -> str: ...

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        capture_sites: tuple[str, ...],
    ) -> object: ...


class _BridgeLike(Protocol):
    bridge_binding_sha256: str

    def validate_integrity(self) -> None: ...

    def execute(
        self,
        adapter: _AdapterLike,
        model_inputs: Mapping[str, Tensor],
        *,
        x4_head: object | None,
        h4_head: object | None,
    ) -> object: ...


def _head_sha256(head: object | None, *, label: str) -> str:
    if head is None:
        raise ValueError(f"{label} head is missing")
    return _require_sha256(
        getattr(head, "artifact_sha256", None),
        label=f"{label} head",
    )


def _validate_live_artifacts(
    *,
    adapter: _AdapterLike,
    bridge: _BridgeLike,
    accepted_x4: _ArtifactLike,
    matched_alpha0: _ArtifactLike,
    challenger_alpha0_5: _ArtifactLike,
) -> dict[str, tuple[object, object | None]]:
    bridge.validate_integrity()
    artifacts = (
        accepted_x4,
        matched_alpha0,
        challenger_alpha0_5,
    )
    for artifact in artifacts:
        artifact.validate_integrity()
    if len({artifact.artifact_sha256 for artifact in artifacts}) != 3:
        raise ValueError("selection artifacts must be distinct")
    model_sha256 = adapter.model_fingerprint()
    execution_sha256 = adapter.execution_fingerprint()
    if any(
        artifact.bridge_binding_sha256 != bridge.bridge_binding_sha256
        or artifact.live_model_sha256 != model_sha256
        or artifact.adapter_execution_sha256 != execution_sha256
        for artifact in artifacts
    ):
        raise ValueError(
            "selection artifacts differ from the live factorized runtime"
        )
    if (
        matched_alpha0.parent_artifact_sha256
        != accepted_x4.artifact_sha256
        or challenger_alpha0_5.parent_artifact_sha256
        != accepted_x4.artifact_sha256
    ):
        raise ValueError("materialized arms do not descend from accepted X4")
    accepted_x4_head = accepted_x4.head(_X4_SITE)
    alpha0_x4_head = matched_alpha0.head(_X4_SITE)
    challenger_x4_head = challenger_alpha0_5.head(_X4_SITE)
    accepted_x4_sha256 = _head_sha256(
        accepted_x4_head,
        label="accepted X4",
    )
    if (
        _head_sha256(alpha0_x4_head, label="alpha0 X4")
        != accepted_x4_sha256
        or _head_sha256(challenger_x4_head, label="challenger X4")
        != accepted_x4_sha256
        or accepted_x4.head(_H4_SITE) is not None
    ):
        raise ValueError("selection arms do not share exact accepted X4")
    alpha0_h4 = matched_alpha0.head(_H4_SITE)
    challenger_h4 = challenger_alpha0_5.head(_H4_SITE)
    _head_sha256(alpha0_h4, label="matched alpha0 H4")
    _head_sha256(challenger_h4, label="challenger H4")
    if (
        getattr(alpha0_h4, "conditioning", None) != "l3_source_modes"
        or getattr(challenger_h4, "conditioning", None)
        != "l3_source_modes_plus_independent_realized_h4_modes_v1"
    ):
        raise ValueError("selection H4 arm semantics differ")
    return {
        ACCEPTED_X4_ONLY_ARM: (accepted_x4_head, None),
        MATCHED_ALPHA0_ARM: (alpha0_x4_head, alpha0_h4),
        CHALLENGER_ALPHA0_5_ARM: (
            challenger_x4_head,
            challenger_h4,
        ),
    }


def _gather_logits(logits: Tensor, indices: Tensor) -> Tensor:
    if (
        not isinstance(logits, Tensor)
        or logits.is_floating_point() is False
        or logits.ndim != 3
        or logits.shape[0] != 1
    ):
        raise ValueError("live logits must have shape [1, sequence, vocab]")
    return (
        logits[0]
        .index_select(0, indices.to(logits.device))
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )


@dataclass(frozen=True, slots=True)
class GemmaH4DampingLiveCollection:
    """Scalar observations plus a scalar/hash-only forward audit."""

    arms: Mapping[str, GemmaH4DampingFiniteNLLArmInput]
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        if set(self.arms) != set(DAMPING_FINITE_NLL_ARM_IDS):
            raise ValueError("live collection arm IDs differ")
        if any(
            arm.examples
            or len(arm.observations) != 16
            or any(
                not isinstance(
                    observation,
                    GemmaH4DampingFiniteNLLObservation,
                )
                for observation in arm.observations
            )
            for arm in self.arms.values()
        ):
            raise ValueError(
                "live collection must retain only scalar observations"
            )
        _assert_scalar_hash_only(self.audit, path="collection audit")


def collect_gemma_h4_damping_live_arms(
    *,
    panel: GemmaProgressivePanel,
    adapter: _AdapterLike,
    bridge: _BridgeLike,
    accepted_x4_artifact: _ArtifactLike,
    matched_alpha0_artifact: _ArtifactLike,
    challenger_alpha0_5_artifact: _ArtifactLike,
) -> GemmaH4DampingLiveCollection:
    """Stream one native pass and three arm passes into scalar observations."""

    if not isinstance(panel, GemmaProgressivePanel):
        raise TypeError("panel must be a strict GemmaProgressivePanel")
    if (
        panel.role != "calibration_a_selection"
        or len(panel.examples) != 16
        or tuple(example.family_id for example in panel.examples)
        != FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE
    ):
        raise ValueError("live selection panel geometry differs")
    heads = _validate_live_artifacts(
        adapter=adapter,
        bridge=bridge,
        accepted_x4=accepted_x4_artifact,
        matched_alpha0=matched_alpha0_artifact,
        challenger_alpha0_5=challenger_alpha0_5_artifact,
    )
    artifacts: Mapping[str, _ArtifactLike] = {
        ACCEPTED_X4_ONLY_ARM: accepted_x4_artifact,
        MATCHED_ALPHA0_ARM: matched_alpha0_artifact,
        CHALLENGER_ALPHA0_5_ARM: challenger_alpha0_5_artifact,
    }
    source_model_sha256 = adapter.model_fingerprint()
    source_execution_sha256 = adapter.execution_fingerprint()
    observations_by_arm: dict[
        str,
        list[GemmaH4DampingFiniteNLLObservation],
    ] = {
        arm_id: [] for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    }
    result_sha256s_by_arm: dict[str, list[str]] = {
        arm_id: [] for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    }
    boundary_receipts: list[str] = []
    model_input_sha256s: list[str] = []

    for example in panel.examples:
        example.validate_integrity()
        model_inputs = example.batch.model_inputs
        with torch.no_grad():
            native = adapter.forward(
                model_inputs,
                capture_sites=(),
            )
        native_sequence = getattr(native, "sequence", None)
        native_valid = getattr(native_sequence, "query_valid_mask", None)
        native_positions = getattr(native_sequence, "logical_positions", None)
        if (
            not isinstance(native_valid, Tensor)
            or native_valid.dtype != torch.bool
            or not isinstance(native_positions, Tensor)
            or native_positions.dtype not in (torch.int32, torch.int64)
            or native_valid.shape != model_inputs["input_ids"].shape
            or native_positions.shape != native_valid.shape
        ):
            raise ValueError(
                "direct native authority grid differs"
            )
        valid_mask = native_valid
        logical_positions = native_positions
        indices, targets = derive_gemma3_l3_l4_supervised_boundary(
            model_inputs["input_ids"],
            valid_mask,
        )
        expected_targets = torch.full_like(example.batch.targets, -100)
        expected_targets[0, indices.to(expected_targets.device)] = targets.to(
            expected_targets.device
        )
        if (
            not torch.equal(
                example.batch.valid_positions.to(valid_mask.device),
                valid_mask,
            )
            or not torch.equal(example.batch.targets, expected_targets)
        ):
            raise ValueError(
                "selection calibration targets differ from causal boundary"
            )
        source_logits = _gather_logits(
            getattr(native, "logits", None),
            indices,
        )
        # The full native run is no longer needed.  Only this prompt's
        # supervised source rows remain while its three arms stream.
        del native, native_sequence, native_valid, native_positions
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS:
            requested_x4, requested_h4 = heads[arm_id]
            with torch.no_grad():
                execution = bridge.execute(
                    adapter,
                    model_inputs,
                    x4_head=requested_x4,
                    h4_head=requested_h4,
                )
            validate_execution = getattr(
                execution,
                "validate_integrity",
                None,
            )
            if not callable(validate_execution):
                raise TypeError(
                    f"{arm_id} execution lacks integrity validation"
                )
            validate_execution()
            arm_prefix = getattr(execution, "prefix", None)
            arm_valid_mask = getattr(
                arm_prefix,
                "valid_target_mask",
                None,
            )
            arm_logical_positions = getattr(
                arm_prefix,
                "logical_positions",
                None,
            )
            expected_x4_sha256 = _head_sha256(
                requested_x4,
                label=f"{arm_id} requested X4",
            )
            expected_h4_sha256 = (
                None
                if requested_h4 is None
                else _head_sha256(
                    requested_h4,
                    label=f"{arm_id} requested H4",
                )
            )
            result_sha256 = _require_sha256(
                getattr(execution, "artifact_sha256", None),
                label=f"{arm_id} execution result",
            )
            if (
                getattr(execution, "model_forward_count", None) != 1
                or getattr(execution, "model_inputs_sha256", None)
                != example.model_inputs_sha256
                or getattr(execution, "bridge_binding_sha256", None)
                != bridge.bridge_binding_sha256
                or getattr(execution, "x4_head_sha256", None)
                != expected_x4_sha256
                or getattr(execution, "h4_head_sha256", None)
                != expected_h4_sha256
            ):
                raise ValueError(f"{arm_id} execution identity differs")
            if (
                not isinstance(arm_valid_mask, Tensor)
                or not isinstance(arm_logical_positions, Tensor)
                or not torch.equal(arm_valid_mask, valid_mask)
                or not torch.equal(arm_logical_positions, logical_positions)
            ):
                raise ValueError(f"{arm_id} execution grid differs")
            candidate_logits = _gather_logits(
                getattr(execution, "logits", None),
                indices,
            )
            transient_example = ShadowFidelityExample(
                example_id=example.example_id,
                family_id=example.family_id,
                source_logits=source_logits,
                candidate_logits=candidate_logits,
                targets=targets,
            )
            observation = (
                measure_gemma_h4_damping_finite_nll_observation(
                    transient_example
                )
            )
            observations_by_arm[arm_id].append(observation)
            result_sha256s_by_arm[arm_id].append(result_sha256)
            # The observation owns only scalars and hashes.  Drop every
            # candidate output reference before executing the next arm.
            del (
                transient_example,
                candidate_logits,
                execution,
                validate_execution,
                arm_prefix,
                arm_valid_mask,
                arm_logical_positions,
                observation,
            )
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            != example.model_inputs_sha256
        ):
            raise RuntimeError(
                "selection model inputs changed during matched forwards"
            )
        boundary_receipts.append(
            _sha256(
                _BOUNDARY_DOMAIN,
                {
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "model_inputs_sha256": example.model_inputs_sha256,
                    "boundary_indices_sha256": _tensor_sha256(indices),
                    "targets_sha256": _tensor_sha256(targets),
                    "valid_target_mask_sha256": _tensor_sha256(valid_mask),
                    "logical_positions_sha256": _tensor_sha256(
                        logical_positions
                    ),
                },
            )
        )
        model_input_sha256s.append(example.model_inputs_sha256)
        # No prompt's source rows cross the next prompt boundary.
        del source_logits, indices, targets, valid_mask, logical_positions

    if (
        adapter.model_fingerprint() != source_model_sha256
        or adapter.execution_fingerprint() != source_execution_sha256
    ):
        raise RuntimeError("live factorized source changed during selection")
    arms = {
        arm_id: GemmaH4DampingFiniteNLLArmInput(
            arm_id=arm_id,  # type: ignore[arg-type]
            semantic=DAMPING_FINITE_NLL_ARM_SEMANTICS[arm_id],  # type: ignore[index]
            execution_receipt_sha256=artifacts[arm_id].execution_sha256,
            observations=tuple(observations_by_arm[arm_id]),
        )
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    }
    audit: dict[str, object] = {
        "execution_mode": "matched_independent_prefill_forwards",
        "example_count": len(panel.examples),
        "model_forward_count_per_example": 4,
        "total_model_forward_count": 4 * len(panel.examples),
        "native_source_forward_count_per_example": 1,
        "candidate_forward_count_per_arm_per_example": 1,
        "native_source_reused_within_prompt_across_arm_metrics": True,
        "native_source_retained_across_prompts": False,
        "candidate_execution_released_before_next_arm": True,
        "finite_nll_measurement_mode": (
            "immediate_scalar_observation_per_arm_per_prompt"
        ),
        "arm_inputs_retain_shadow_examples": False,
        "arm_inputs_retain_logits": False,
        "native_source_semantics": (
            "direct_factorized_adapter_forward_without_interventions"
        ),
        "source_model_sha256": source_model_sha256,
        "source_execution_sha256": source_execution_sha256,
        "model_input_sha256s": tuple(sorted(model_input_sha256s)),
        "boundary_receipt_sha256s": tuple(sorted(boundary_receipts)),
        "arm_execution_sha256s": {
            arm_id: artifacts[arm_id].execution_sha256
            for arm_id in DAMPING_FINITE_NLL_ARM_IDS
        },
        "arm_result_sha256s": {
            arm_id: tuple(result_sha256s_by_arm[arm_id])
            for arm_id in DAMPING_FINITE_NLL_ARM_IDS
        },
        "raw_logits_retained": False,
        "tokens_retained": False,
    }
    audit["collection_sha256"] = _sha256(_COLLECTION_DOMAIN, audit)
    return GemmaH4DampingLiveCollection(arms=arms, audit=audit)


def _adapt_fresh_panel(
    *,
    artifact: Gemma3L3L4H4DampingSelectionPanelArtifact,
    opened: object,
    tokenizer: object,
    max_length: int,
    device: torch.device,
) -> GemmaProgressivePanel:
    prompts = tuple(getattr(opened, "prompts"))
    family_ids = tuple(getattr(opened, "family_ids"))
    source_file_sha256 = str(getattr(opened, "source_file_sha256"))
    role_input = Gemma3L3L4ProgressiveARolePrompts(
        corpus_id=GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
        profile="pilot",
        role="calibration_a_selection",
        prompts=prompts,
        family_ids=family_ids,
        source_file_sha256=source_file_sha256,
    )
    if role_input.ordered_prompt_sha256s != artifact.ordered_prompt_sha256s:
        raise ValueError("adapted selection prompt identities differ")
    view = Gemma3L3L4ProgressiveARolePreclaimView(
        role="calibration_a_selection",
        manifest_sha256=artifact.manifest_sha256,
        role_input_file_sha256=source_file_sha256,
        example_count=len(artifact.ordered_prompt_sha256s),
        family_ids=artifact.family_ids,
        ordered_prompt_sha256s=artifact.ordered_prompt_sha256s,
        ordered_family_ids=artifact.ordered_family_ids,
    )
    panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=role_input,
        view=view,
        max_length=max_length,
        device=device,
        forbidden_manifest_sha256s=(
            artifact.expanded_fit_lineage
            .forbidden_assessment_manifest_sha256s
        ),
    )
    if (
        panel.manifest_sha256 != artifact.manifest_sha256
        or tuple(example.example_id for example in panel.examples)
        != artifact.ordered_prompt_sha256s
        or tuple(example.family_id for example in panel.examples)
        != artifact.ordered_family_ids
    ):
        raise RuntimeError("strict selection panel adaptation drifted")
    return panel


def _source_code_sha256s() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "gemma3_l3_l4_h4_damping_selection_campaign.py",
        "gemma3_l3_l4_h4_damping_selection_panel.py",
        "gemma3_l3_l4_h4_damping_selection_runtime.py",
        "gemma3_l3_l4_h4_damping_materialization.py",
        "gemma3_l3_l4_two_head_lowerer.py",
        "gemma3_l3_l4_graph_organized_svd_shadow_runtime.py",
    )
    return {name: _file_sha256(package / name) for name in names}


def _publish_report(path: Path | str, report: Mapping[str, object]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite selection report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    stage = Path(stage_name)
    try:
        with stage.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, destination)
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        stage.unlink(missing_ok=True)


def run_gemma_h4_damping_selection_campaign(
    *,
    new_selection_input_path: Path | str = (
        _DEFAULT_PRIVATE_SELECTION_INPUT
    ),
    new_selection_panel_path: Path | str = _DEFAULT_SELECTION_PANEL,
    expected_new_selection_panel_artifact_sha256: str,
    expected_new_selection_panel_file_sha256: str,
    materialization_report_path: Path | str = (
        _DEFAULT_MATERIALIZATION_REPORT
    ),
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    matched_alpha0_candidate_path: Path | str = _DEFAULT_ALPHA0_CANDIDATE,
    challenger_alpha0_5_candidate_path: Path | str = (
        _DEFAULT_ALPHA0_5_CANDIDATE
    ),
    accepted_x4_report_path: Path | str = _DEFAULT_ACCEPTED_REPORT,
    accepted_x4_candidate_path: Path | str = _DEFAULT_ACCEPTED_CANDIDATE,
    expected_accepted_x4_candidate_file_sha256: str,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Open the new panel once and execute the frozen three-arm comparison."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite selection report")
    panel_path = Path(new_selection_panel_path)
    if _file_sha256(panel_path) != _require_sha256(
        expected_new_selection_panel_file_sha256,
        label="new selection panel file",
    ):
        raise ValueError("new selection panel file hash differs")
    panel_artifact = (
        load_gemma3_l3_l4_h4_damping_selection_panel_artifact(
            panel_path,
            expected_artifact_sha256=(
                expected_new_selection_panel_artifact_sha256
            ),
        )
    )
    loaded = _load_materialization(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=(
            expected_materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            expected_materialization_report_file_sha256
        ),
        matched_alpha0_candidate_path=matched_alpha0_candidate_path,
        challenger_alpha0_5_candidate_path=(
            challenger_alpha0_5_candidate_path
        ),
        accepted_x4_report_path=accepted_x4_report_path,
        accepted_x4_candidate_path=accepted_x4_candidate_path,
        expected_accepted_x4_candidate_file_sha256=(
            expected_accepted_x4_candidate_file_sha256
        ),
    )
    recollection = _materialization_recollection(loaded.report)
    lineage = panel_artifact.expanded_fit_lineage
    if (
        lineage.expanded_corpus_artifact_sha256
        != recollection["corpus_artifact_sha256"]
        or lineage.fit_manifest_sha256
        != recollection["fit_manifest_sha256"]
        or lineage.fit_binding_sha256 != recollection["fit_binding_sha256"]
    ):
        raise ValueError(
            "fresh selection panel and materialization lineage differ"
        )

    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    protocol.validate_integrity()
    metadata = protocol.metadata()
    tokenizer_contract = dict(
        _mapping(metadata["tokenizer"], label="frozen tokenizer")
    )
    if (
        gemma3_l3_l4_progressive_a_tokenizer_contract_sha256(
            tokenizer_contract
        )
        != lineage.tokenizer_contract_sha256
    ):
        raise ValueError("fresh panel tokenizer contract differs")
    tokenizer, live_tokenizer_contract = (
        _load_and_validate_frozen_local_tokenizer(protocol=protocol)
    )
    if _canonical_json_bytes(
        live_tokenizer_contract
    ) != _canonical_json_bytes(tokenizer_contract):
        raise ValueError("live tokenizer contract differs")

    model_metadata = _mapping(metadata["model"], label="frozen model")
    graph_binding = _mapping(
        metadata["graph_candidate"],
        label="frozen graph candidate",
    )
    basis_binding = _mapping(
        metadata["prompt_blind_basis"],
        label="frozen basis",
    )
    for name in (
        "raw_model_sha256",
        "progressive_runtime_binding_sha256",
        "graph_candidate_file_sha256",
        "basis_file_sha256",
        "base_artifact_file_sha256",
        "refit_artifact_file_sha256",
    ):
        _require_sha256(recollection.get(name), label=name)
    materialized_files = _mapping(
        loaded.report.get("files"),
        label="materialization files",
    )
    alpha0_file = _mapping(
        materialized_files.get("matched_alpha0"),
        label="matched alpha0 file",
    )
    challenger_file = _mapping(
        materialized_files.get("challenger_alpha0_5"),
        label="challenger file",
    )
    accepted_provenance = _mapping(
        recollection.get("accepted_x4_provenance"),
        label="accepted X4 provenance",
    )
    immutable_paths = {
        "new_selection_panel": panel_path,
        "materialization_report": Path(materialization_report_path),
        "matched_alpha0_candidate": Path(matched_alpha0_candidate_path),
        "challenger_alpha0_5_candidate": Path(
            challenger_alpha0_5_candidate_path
        ),
        "accepted_x4_report": Path(accepted_x4_report_path),
        "accepted_x4_candidate": Path(accepted_x4_candidate_path),
        "graph_candidate": Path(graph_candidate_path),
        "basis_package": Path(basis_package_path),
        "base_artifact": Path(base_artifact_path),
        "refit_artifact": Path(refit_artifact_path),
    }
    immutable_expected = {
        "new_selection_panel": expected_new_selection_panel_file_sha256,
        "materialization_report": (
            expected_materialization_report_file_sha256
        ),
        "matched_alpha0_candidate": str(
            alpha0_file["tensor_file_sha256"]
        ),
        "challenger_alpha0_5_candidate": str(
            challenger_file["tensor_file_sha256"]
        ),
        "accepted_x4_report": str(
            accepted_provenance["report_file_sha256"]
        ),
        "accepted_x4_candidate": (
            expected_accepted_x4_candidate_file_sha256
        ),
        "graph_candidate": str(graph_binding["tensor_file_sha256"]),
        "basis_package": str(basis_binding["tensor_file_sha256"]),
        "base_artifact": str(recollection["base_artifact_file_sha256"]),
        "refit_artifact": str(
            recollection["refit_artifact_file_sha256"]
        ),
    }
    immutable_before = {
        name: _file_sha256(path)
        for name, path in immutable_paths.items()
    }
    if immutable_before != immutable_expected:
        raise ValueError(
            "selection immutable input file binding differs"
        )
    if (
        model_metadata["source_model_sha256"]
        != recollection["raw_model_sha256"]
    ):
        raise ValueError("materialization raw model binding differs")
    code_before = _source_code_sha256s()
    device = resolve_torch_device("cpu")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != model_metadata["source_model_sha256"]:
        raise ValueError("live raw Gemma differs from frozen source")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256 = adapter.model_fingerprint()
        factorized_execution_sha256 = adapter.execution_fingerprint()
        if (
            factorized_model_sha256
            != graph_binding["factorized_live_execution_sha256"]
            or factorized_execution_sha256
            != graph_binding["factorized_refit_execution_sha256"]
            or factorized_model_sha256
            != recollection["factorized_model_sha256"]
            or factorized_execution_sha256
            != recollection["factorized_execution_sha256"]
        ):
            raise ValueError(
                "direct native source is not the authenticated factorized "
                "model"
            )
        graph_path = Path(graph_candidate_path)
        basis_path = Path(basis_package_path)
        graph_candidate = load_gemma3_graph_organized_svd_candidate(
            graph_path,
            expected_file_sha256=str(graph_binding["tensor_file_sha256"]),
        )
        basis = load_gemma3_l3_l4_basis_package(
            basis_path,
            expected_file_sha256=str(basis_binding["tensor_file_sha256"]),
            expected_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
        )
        runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            graph_candidate,
            basis,
            expected_candidate_artifact_sha256=str(
                graph_binding["logical_artifact_sha256"]
            ),
            expected_basis_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
            expected_plan_artifact_sha256=str(
                graph_binding["deployment_plan_sha256"]
            ),
            expected_live_model_sha256=str(
                graph_binding["factorized_live_execution_sha256"]
            ),
            expected_adapter_execution_sha256=str(
                graph_binding["factorized_refit_execution_sha256"]
            ),
            analysis_device="cpu",
        )
        if (
            runtime.runtime_binding_sha256
            != recollection["progressive_runtime_binding_sha256"]
        ):
            raise ValueError(
                "materialization progressive runtime binding differs"
            )
        bridge = runtime.export_one_pass_bridge()

        # All public immutable dependencies and the live model are
        # authenticated before this irreversible cross-process claim.  The
        # claim is never removed, including when any later operation fails.
        if (
            {
                name: _file_sha256(path)
                for name, path in immutable_paths.items()
            }
            != immutable_before
            or _source_code_sha256s() != code_before
            or adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
        ):
            raise RuntimeError(
                "public immutable or model state changed before claim"
            )
        claim_payload = _selection_claim_payload(
            panel_artifact_sha256=panel_artifact.artifact_sha256,
            panel_file_sha256=immutable_before["new_selection_panel"],
            private_input_file_sha256=(
                panel_artifact.selection_role_input_file_sha256
            ),
            materialization_report_sha256=str(
                loaded.report["report_sha256"]
            ),
            materialization_report_file_sha256=immutable_before[
                "materialization_report"
            ],
            artifacts={
                ACCEPTED_X4_ONLY_ARM: loaded.accepted_x4,
                MATCHED_ALPHA0_ARM: loaded.matched_alpha0,
                CHALLENGER_ALPHA0_5_ARM: loaded.challenger_alpha0_5,
            },
        )
        claim_path, claim_file_sha256 = _create_selection_claim(claim_payload)
        expected_private_input_sha256 = _require_sha256(
            panel_artifact.selection_role_input_file_sha256,
            label="new selection input file",
        )
        observed_private_input_sha256 = _file_sha256(
            new_selection_input_path
        )
        if observed_private_input_sha256 != expected_private_input_sha256:
            raise ValueError(
                "new selection input changed after its durable claim"
            )
        immutable_paths["new_selection_input"] = Path(
            new_selection_input_path
        )
        immutable_paths["selection_claim"] = claim_path
        immutable_expected["new_selection_input"] = (
            expected_private_input_sha256
        )
        immutable_expected["selection_claim"] = claim_file_sha256
        immutable_before["new_selection_input"] = (
            observed_private_input_sha256
        )
        immutable_before["selection_claim"] = claim_file_sha256

        # This is the first prompt-text read.  A failure here or anywhere
        # below still leaves the durable claim in place.
        source = Gemma3L3L4H4DampingSelectionPanelSource(
            artifact=panel_artifact,
            selection_input_path=new_selection_input_path,
        )
        opened = source.open_once()
        selection_input_file_sha256 = _require_sha256(
            getattr(opened, "source_file_sha256", None),
            label="new selection input file",
        )
        panel = _adapt_fresh_panel(
            artifact=panel_artifact,
            opened=opened,
            tokenizer=tokenizer,
            max_length=int(tokenizer_contract["max_length"]),
            device=torch.device(str(tokenizer_contract["device"])),
        )
        if not source.consumed or not source.opened:
            raise RuntimeError("fresh selection input was not consumed once")
        collection = collect_gemma_h4_damping_live_arms(
            panel=panel,
            adapter=adapter,
            bridge=bridge,
            accepted_x4_artifact=loaded.accepted_x4,
            matched_alpha0_artifact=loaded.matched_alpha0,
            challenger_alpha0_5_artifact=loaded.challenger_alpha0_5,
        )
        finite_nll = evaluate_gemma_h4_damping_finite_nll(
            collection.arms,
            expected_family_by_example=panel_artifact.family_by_example,
        )
        code_after = _source_code_sha256s()
        if code_after != code_before:
            raise RuntimeError("selection source changed during execution")
        immutable_after = {
            name: _file_sha256(path)
            for name, path in immutable_paths.items()
        }
        if (
            immutable_after != immutable_before
            or selection_input_file_sha256
            != immutable_before["new_selection_input"]
        ):
            raise RuntimeError(
                "selection immutable input changed during execution"
            )
        if (
            adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
        ):
            raise RuntimeError(
                "selection model or frozen runtime artifacts changed"
            )

        report: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "durable_claim": {
                "path": str(claim_path),
                "claim_sha256": claim_payload["claim_sha256"],
                "claim_file_sha256": claim_file_sha256,
                "private_input_file_sha256": immutable_before[
                    "new_selection_input"
                ],
                "consume_before_prompt_read": True,
                "cross_process": True,
                "retained_after_failure": True,
                "alternate_output_rerun_allowed": False,
                "identity_keyed": True,
                "private_input_path_affects_identity": False,
            },
            "panel": {
                "panel_id": panel_artifact.panel_id,
                "panel_artifact_sha256": panel_artifact.artifact_sha256,
                "panel_file_sha256": immutable_before[
                    "new_selection_panel"
                ],
                "selection_input_file_sha256": (
                    selection_input_file_sha256
                ),
                "manifest_sha256": panel_artifact.manifest_sha256,
                "membership_receipt_sha256": (
                    panel_artifact.membership_receipt_sha256
                ),
                "runtime_membership_receipt_sha256": (
                    panel.membership_receipt_sha256
                ),
                "family_ids": panel_artifact.family_ids,
                "example_count": len(panel.examples),
                "input_consumed_once": True,
            },
            "materialization": {
                "report_sha256": loaded.report["report_sha256"],
                "report_file_sha256": immutable_before[
                    "materialization_report"
                ],
                "accepted_x4_artifact_sha256": (
                    loaded.accepted_x4.artifact_sha256
                ),
                "matched_alpha0_artifact_sha256": (
                    loaded.matched_alpha0.artifact_sha256
                ),
                "challenger_alpha0_5_artifact_sha256": (
                    loaded.challenger_alpha0_5.artifact_sha256
                ),
            },
            "runtime": {
                "raw_model_sha256": model_metadata["source_model_sha256"],
                "factorized_model_sha256": factorized_model_sha256,
                "factorized_execution_sha256": (
                    factorized_execution_sha256
                ),
                "progressive_runtime_binding_sha256": (
                    runtime.runtime_binding_sha256
                ),
                "bridge_binding_sha256": bridge.bridge_binding_sha256,
                "graph_candidate_file_sha256": immutable_before[
                    "graph_candidate"
                ],
                "basis_file_sha256": immutable_before["basis_package"],
                "base_artifact_file_sha256": immutable_before[
                    "base_artifact"
                ],
                "refit_artifact_file_sha256": immutable_before[
                    "refit_artifact"
                ],
                "source_code_sha256s": code_before,
                "source_authority": (
                    "direct_factorized_adapter_forward_no_interventions"
                ),
            },
            "execution": collection.audit,
            "finite_nll": finite_nll,
            "result": {
                "qualified": finite_nll["qualification"]["qualified"],
                "paired_gate_passed": finite_nll["qualification"][
                    "paired_gate_passed"
                ],
                "absolute_gate_passed": finite_nll["qualification"][
                    "challenger_absolute_gate_passed"
                ],
                "selection_decision_is_one_shot": True,
            },
            "safety": {
                "development_only": True,
                "new_selection_input_opened": True,
                "new_selection_input_consumed_once": True,
                "durable_cross_process_claim_created": True,
                "claim_deleted_after_failure": False,
                "alternate_output_rerun_capability_present": False,
                "old_selection_input_capability_present": False,
                "guard_input_capability_present": False,
                "calibration_b_input_capability_present": False,
                "fit_input_capability_present": False,
                "raw_prompt_text_in_report": False,
                "token_ids_in_report": False,
                "logits_in_report": False,
                "activation_rows_in_report": False,
                "gradient_rows_in_report": False,
                "coefficient_tensors_in_report": False,
                "model_weights_in_report": False,
                "compression_claim": False,
                "latency_claim": False,
            },
        }
        _assert_scalar_hash_only(report)
        report["report_sha256"] = _sha256(_REPORT_DOMAIN, report)
        _publish_report(destination, report)
        return report
    finally:
        switcher.close()


def build_parser() -> argparse.ArgumentParser:
    """Return the run-only CLI; no historical panel capability is present."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the one-shot fresh H4 damping finite-NLL selection panel."
        )
    )
    parser.add_argument(
        "--new-selection-input",
        type=Path,
        default=_DEFAULT_PRIVATE_SELECTION_INPUT,
    )
    parser.add_argument(
        "--new-selection-panel",
        type=Path,
        default=_DEFAULT_SELECTION_PANEL,
    )
    parser.add_argument("--new-selection-panel-artifact-sha256", required=True)
    parser.add_argument("--new-selection-panel-file-sha256", required=True)
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=_DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument("--materialization-report-sha256", required=True)
    parser.add_argument(
        "--materialization-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--matched-alpha0-candidate",
        type=Path,
        default=_DEFAULT_ALPHA0_CANDIDATE,
    )
    parser.add_argument(
        "--challenger-alpha0-5-candidate",
        type=Path,
        default=_DEFAULT_ALPHA0_5_CANDIDATE,
    )
    parser.add_argument(
        "--accepted-x4-report",
        type=Path,
        default=_DEFAULT_ACCEPTED_REPORT,
    )
    parser.add_argument(
        "--accepted-x4-candidate",
        type=Path,
        default=_DEFAULT_ACCEPTED_CANDIDATE,
    )
    parser.add_argument("--accepted-x4-candidate-sha256", required=True)
    parser.add_argument(
        "--graph-candidate",
        type=Path,
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def build_preparation_parser() -> argparse.ArgumentParser:
    """Return the separate prompt-free-lineage preparation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare one private fresh selection input and prompt-free panel."
        )
    )
    parser.add_argument("--expanded-corpus-artifact", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--materialization-report-sha256", required=True)
    parser.add_argument(
        "--materialization-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--new-selection-input-output",
        type=Path,
        default=_DEFAULT_PRIVATE_SELECTION_INPUT,
    )
    parser.add_argument(
        "--new-selection-panel-output",
        type=Path,
        default=_DEFAULT_SELECTION_PANEL,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma_h4_damping_selection_campaign(
        new_selection_input_path=args.new_selection_input,
        new_selection_panel_path=args.new_selection_panel,
        expected_new_selection_panel_artifact_sha256=(
            args.new_selection_panel_artifact_sha256
        ),
        expected_new_selection_panel_file_sha256=(
            args.new_selection_panel_file_sha256
        ),
        materialization_report_path=args.materialization_report,
        expected_materialization_report_sha256=(
            args.materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            args.materialization_report_file_sha256
        ),
        matched_alpha0_candidate_path=args.matched_alpha0_candidate,
        challenger_alpha0_5_candidate_path=(
            args.challenger_alpha0_5_candidate
        ),
        accepted_x4_report_path=args.accepted_x4_report,
        accepted_x4_candidate_path=args.accepted_x4_candidate,
        expected_accepted_x4_candidate_file_sha256=(
            args.accepted_x4_candidate_sha256
        ),
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def preparation_main(argv: Sequence[str] | None = None) -> int:
    args = build_preparation_parser().parse_args(argv)
    report = prepare_gemma_h4_damping_selection_campaign(
        expanded_corpus_artifact_path=args.expanded_corpus_artifact,
        materialization_report_path=args.materialization_report,
        expected_materialization_report_sha256=(
            args.materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            args.materialization_report_file_sha256
        ),
        private_selection_input_output=(
            args.new_selection_input_output
        ),
        selection_panel_output=args.new_selection_panel_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
