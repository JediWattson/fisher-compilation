"""Prepare the new family-disjoint generator-innovation fit panel.

The fixed generator basis and causal innovation feature must exist before any
prompt in this module is published.  Preparation therefore authenticates the
exact frozen plan, the expanded Calibration-A corpus, the prior occupancy
selection panel, and the prompt-blind Calibration-B identities before it
writes either output.

The private output is a standard ``calibration_a_fit`` role input.  The public
output is a prompt-free receipt that binds the exact plan and ordered
prompt/family identities.  Both outputs are installed with exclusive links;
an interrupted or racing publication cannot overwrite an existing file.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from .gemma3_l3_l4_h4_damping_selection_panel import (
    _prompt_blind_forbidden_binding,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from .gemma3_l3_l4_iterative_generator_innovation_plan import (
    validate_gemma_iterative_generator_innovation_plan,
)
from .gemma3_l3_l4_iterative_occupancy_selection_panel import (
    Gemma3L3L4IterativeOccupancySelectionPanelArtifact,
    load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION,
    GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA,
    Gemma3L3L4ProgressiveACorpusArtifact,
    Gemma3L3L4ProgressiveARolePreclaimView,
    Gemma3L3L4ProgressiveARolePrompts,
    _load_progressive_a_artifact,
)
from .gemma3_l3_l4_progressive_worker import (
    GemmaProgressivePanel,
    gemma_progressive_panel_membership_receipt_sha256,
)


__all__ = [
    "DEFAULT_EXPANDED_FIT_CORPUS",
    "DEFAULT_GENERATOR_INNOVATION_PLAN",
    "DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT",
    "DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT",
    "DEFAULT_PRIOR_OCCUPANCY_PANEL",
    "FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256",
    "FROZEN_GENERATOR_INNOVATION_PLAN_SHA256",
    "GENERATOR_INNOVATION_FAMILIES",
    "GENERATOR_INNOVATION_FAMILY_SCHEDULE",
    "GENERATOR_INNOVATION_PANEL_ID",
    "GENERATOR_INNOVATION_PROMPTS",
    "Gemma3L3L4GeneratorInnovationPanelIntegrityError",
    "Gemma3L3L4GeneratorInnovationPanelReceipt",
    "load_gemma3_l3_l4_generator_innovation_panel_receipt",
    "load_gemma3_l3_l4_generator_innovation_role_input",
    "materialize_gemma3_l3_l4_generator_innovation_panel",
    "prepare_gemma3_l3_l4_generator_innovation_panel",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_GENERATOR_INNOVATION_PLAN = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-plan-v1.report.json"
)
DEFAULT_EXPANDED_FIT_CORPUS = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
DEFAULT_PRIOR_OCCUPANCY_PANEL = (
    _LOCAL_ROOT / "progressive-a-iterative-occupancy-selection-v1.panel.json"
)
DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-v1.private.json"
)
DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-v1.panel.json"
)

FROZEN_GENERATOR_INNOVATION_PLAN_SHA256 = (
    "a505dbcde88a7e3d55511dd5badd509c8f5fb52470b197446f33240d7b83d776"
)
FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256 = (
    "a1b3464e3e52696090a993d8e408f4cde79c22a9a14d1bd4ee51649c50c4a511"
)
_EXPANDED_FIT_CORPUS_ARTIFACT_SHA256 = (
    "e4804338dbc3e76a84bf0483526ac9bab4e5f8aeaa86a32283832fed25f4b766"
)
_EXPANDED_FIT_CORPUS_FILE_SHA256 = (
    "7cc40568413cb3dc5713354ded1f1f143e5c89e7c46df8089ed96199961614e2"
)
_PRIOR_OCCUPANCY_PANEL_ARTIFACT_SHA256 = (
    "1196c8680c985a322c2c6680293a7d826113fab6d178a9d0a218f0cf52ac42df"
)
_PRIOR_OCCUPANCY_PANEL_FILE_SHA256 = (
    "356214833d14031d3b043899d8b658a8469d8fb1bbc117055c4e168418942aaa"
)

GENERATOR_INNOVATION_PANEL_ID = (
    "gemma3-l3-l4-iterative-generator-innovation-fit-v1"
)
_ROLE = "calibration_a_fit"
_RECEIPT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.iterative_generator_innovation_panel.v1"
)
_MANIFEST_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-panel-manifest:v1\0"
)
_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-panel-receipt:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024

GENERATOR_INNOVATION_FAMILIES = (
    "generator-innovation-v1-graph-failure-propagation",
    "generator-innovation-v1-reversible-irreversible-transformation",
    "generator-innovation-v1-versioned-policy-conflict",
    "generator-innovation-v1-probabilistic-sensor-fusion",
    "generator-innovation-v1-spatial-occlusion-navigation",
    "generator-innovation-v1-queue-service-capacity",
    "generator-innovation-v1-recursive-program-trace",
    "generator-innovation-v1-adversarial-consistency-audit",
)
GENERATOR_INNOVATION_FAMILY_SCHEDULE = (
    GENERATOR_INNOVATION_FAMILIES + GENERATOR_INNOVATION_FAMILIES
)

# The two rounds deliberately change surface domain, vocabulary, and requested
# representation while preserving the semantic capability named by the family.
GENERATOR_INNOVATION_PROMPTS = (
    (
        "A data pipeline forms a directed graph: ingest feeds normalize and "
        "archive; normalize feeds score; score and archive both feed publish. "
        "The normalize service fails, while archive remains healthy. Trace "
        "which outputs are definitely unavailable, which can still be "
        "reached, and where a fallback edge would prevent the failure from "
        "propagating. State the dependency rule used at each node."
    ),
    (
        "A restoration lab photographs a manuscript, rotates the image, "
        "converts it to grayscale, crops away a margin, and later renames the "
        "file. Classify each operation as exactly reversible, reversible only "
        "with retained side information, or irreversible. Then give the "
        "minimal provenance record needed to reconstruct the closest possible "
        "earlier state."
    ),
    (
        "A company travel request was submitted under policy version 4, "
        "approved after version 5 took effect, and reimbursed after a version "
        "5 exception was amended. Version 4 governs submission eligibility; "
        "version 5 governs reimbursement, but approved requests retain their "
        "old meal cap. Resolve the applicable rules without mixing versions "
        "and identify the remaining conflict."
    ),
    (
        "A coastal station estimates whether a storm is approaching. Radar "
        "reports strong motion with a 10% false-alarm rate, a pressure sensor "
        "reports a sharp drop but shares a weather-model bias with the radar, "
        "and an independent buoy reports calm water. Explain how to combine "
        "the evidence without counting the shared bias twice, and say which "
        "reading should most change the posterior."
    ),
    (
        "A warehouse robot must travel from loading bay A to shelf D. A tall "
        "rack blocks direct sight of beacon C, a glass partition blocks the "
        "robot but not the beacon signal, and a low pallet blocks the lidar "
        "beam only from the south aisle. Construct a collision-free route and "
        "separate physical reachability from line-of-sight observability at "
        "each turn."
    ),
    (
        "A clinic has one check-in desk serving 12 patients per hour, two "
        "nurses serving 5 patients per hour each, and one physician serving 8 "
        "patients per hour. Patients must visit all three stages and arrive "
        "at 9 per hour. Identify the bottleneck, predict where the queue grows, "
        "and calculate the smallest capacity change that makes the flow stable."
    ),
    (
        "Trace the call stack for f(5) where f(n) returns 1 when n is at most "
        "1 and otherwise returns f(n-1) plus f(n-2). List calls in the order "
        "they are entered, mark when each frame returns, and show how repeated "
        "subproblems contribute to the final value without skipping recursive "
        "branches."
    ),
    (
        "A vendor claims every exported record was encrypted, no failed export "
        "was retried, and the audit log is complete. The log shows one retry "
        "after a timeout, a checksum list includes an export absent from the "
        "log, and the vendor's summary counts that export as encrypted. Build "
        "a consistency audit that identifies mutually incompatible claims, "
        "possible benign explanations, and the evidence an adversary could "
        "have selectively removed."
    ),
    (
        "A town water network has pumps P1 and P2 feeding junction J; J feeds "
        "hospitals H1 and H2, while P2 also has a direct emergency line to H2. "
        "A power fault disables P1 and then causes J's pressure valve to close. "
        "Propagate the two failures through the graph in causal order, list "
        "which hospital supply paths survive, and identify a single new edge "
        "that removes the shared point of failure."
    ),
    (
        "An audio archive is copied losslessly, resampled from 96 kHz to 48 "
        "kHz, normalized using a recorded gain, mixed from stereo to mono, and "
        "wrapped in a new container. For every step, decide whether the prior "
        "signal can be recovered exactly, approximately, or not at all. "
        "Explain how saved metadata changes those judgments and where "
        "information is fundamentally destroyed."
    ),
    (
        "An API client was created under access rules dated March. A June rule "
        "removes write access from its class, a July migration grants a "
        "temporary exception to clients created before May, and an August "
        "security patch says all temporary exceptions expire immediately. At "
        "an August request, determine which clauses supersede which and expose "
        "any ambiguity between effective date and creation-date scope."
    ),
    (
        "A rover localizes itself from GPS, wheel odometry, and a landmark "
        "camera. GPS and odometry both drift east because their calibration "
        "used the same faulty reference point; the camera independently places "
        "the rover west of both estimates but is sometimes occluded. Describe "
        "a probabilistic fusion that accounts for correlated error, changing "
        "camera reliability, and the observation that would best distinguish "
        "shared bias from camera failure."
    ),
    (
        "A hiker sees a radio tower east of a ridge but cannot see the lake "
        "behind the ridge. The trail passes through a tunnel that is walkable "
        "but blocks GPS, then reaches a lookout where the lake and tower align "
        "in view. Sketch the sequence of navigation states, distinguishing "
        "what is reachable, what is visible, and what can be inferred while "
        "each landmark is occluded."
    ),
    (
        "A router receives 1,200 packets per second. Parsing handles 1,500, "
        "inspection handles 900, and two parallel forwarding lanes each handle "
        "550 packets per second. Buffers hold 3,000 packets before inspection. "
        "Locate the limiting service stage, estimate how quickly the buffer "
        "fills, and compare adding a second inspector with increasing only the "
        "forwarding capacity."
    ),
    (
        "A tree procedure visit(node) first records the node's label, then "
        "recursively visits its left child, then its right child. The root A "
        "has children B and C; B has right child D; C has children E and F. "
        "Produce an entry-and-return trace with stack depth at every event, "
        "then explain how the trace changes if the procedure visits the right "
        "child first."
    ),
    (
        "Three replicated ledgers should contain the same ten transactions. "
        "Ledger A lacks transaction 7, ledger B contains transaction 7 twice, "
        "and ledger C agrees with an independently signed digest but reports a "
        "timestamp copied from A. An operator insists that any two matching "
        "fields prove consistency. Design an adversarial audit that separates "
        "independent agreement from copied evidence and identifies the minimum "
        "set of records that cannot all be true."
    ),
)


class Gemma3L3L4GeneratorInnovationPanelIntegrityError(RuntimeError):
    """The plan, lineage, private input, or receipt failed authentication."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file():
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            f"{label} must be a regular file"
        )
    encoded = path.read_bytes()
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            f"{label} has an invalid size"
        )
    return encoded


def _decode_mapping(encoded: bytes, *, label: str) -> dict[str, object]:
    try:
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(raw, dict):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            f"{label} must contain one JSON object"
        )
    return raw


def _role_payload() -> dict[str, object]:
    return {
        "schema": GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA,
        "format_version": GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION,
        "corpus_id": GENERATOR_INNOVATION_PANEL_ID,
        "profile": "pilot",
        "role": _ROLE,
        "prompts": GENERATOR_INNOVATION_PROMPTS,
        "family_ids": GENERATOR_INNOVATION_FAMILY_SCHEDULE,
    }


def _prompt_sha256s() -> tuple[str, ...]:
    return tuple(
        gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
        for prompt in GENERATOR_INNOVATION_PROMPTS
    )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GeneratorInnovationPanelReceipt:
    """Prompt-free commitment to the fixed plan and fresh fit membership."""

    plan_sha256: str
    plan_file_sha256: str
    expanded_fit_corpus_artifact_sha256: str
    expanded_fit_corpus_file_sha256: str
    prior_occupancy_panel_artifact_sha256: str
    prior_occupancy_panel_file_sha256: str
    forbidden_assessment_manifest_sha256s: tuple[str, ...]
    role_input_file_sha256: str
    ordered_prompt_sha256s: tuple[str, ...]
    ordered_family_ids: tuple[str, ...]
    manifest_sha256: str = field(init=False)
    membership_receipt_sha256: str = field(init=False)
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "plan_sha256",
            "plan_file_sha256",
            "expanded_fit_corpus_artifact_sha256",
            "expanded_fit_corpus_file_sha256",
            "prior_occupancy_panel_artifact_sha256",
            "prior_occupancy_panel_file_sha256",
            "role_input_file_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            self.plan_sha256 != FROZEN_GENERATOR_INNOVATION_PLAN_SHA256
            or self.plan_file_sha256
            != FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
            or self.expanded_fit_corpus_artifact_sha256
            != _EXPANDED_FIT_CORPUS_ARTIFACT_SHA256
            or self.expanded_fit_corpus_file_sha256
            != _EXPANDED_FIT_CORPUS_FILE_SHA256
            or self.prior_occupancy_panel_artifact_sha256
            != _PRIOR_OCCUPANCY_PANEL_ARTIFACT_SHA256
            or self.prior_occupancy_panel_file_sha256
            != _PRIOR_OCCUPANCY_PANEL_FILE_SHA256
        ):
            raise ValueError("panel lineage differs from the frozen rung")
        forbidden = tuple(sorted(set(self.forbidden_assessment_manifest_sha256s)))
        if (
            forbidden != self.forbidden_assessment_manifest_sha256s
            or len(forbidden) != 1
        ):
            raise ValueError(
                "forbidden assessment manifests must contain one sorted identity"
            )
        for value in forbidden:
            _require_sha256(value, label="forbidden assessment manifest")
        expected_prompts = _prompt_sha256s()
        if (
            self.ordered_prompt_sha256s != expected_prompts
            or len(set(expected_prompts)) != 16
        ):
            raise ValueError("panel prompt identities differ from the frozen set")
        if self.ordered_family_ids != GENERATOR_INNOVATION_FAMILY_SCHEDULE:
            raise ValueError("panel family schedule differs from the frozen set")
        if Counter(self.ordered_family_ids) != Counter(
            {family: 2 for family in GENERATOR_INNOVATION_FAMILIES}
        ):
            raise ValueError("panel must contain two examples per family")
        manifest = _domain_sha256(
            _MANIFEST_DOMAIN,
            {
                "panel_id": GENERATOR_INNOVATION_PANEL_ID,
                "role": _ROLE,
                "plan_sha256": self.plan_sha256,
                "plan_file_sha256": self.plan_file_sha256,
                "expanded_fit_corpus_artifact_sha256": (
                    self.expanded_fit_corpus_artifact_sha256
                ),
                "prior_occupancy_panel_artifact_sha256": (
                    self.prior_occupancy_panel_artifact_sha256
                ),
                "forbidden_assessment_manifest_sha256s": forbidden,
                "role_input_file_sha256": self.role_input_file_sha256,
                "ordered_members": tuple(
                    zip(
                        self.ordered_prompt_sha256s,
                        self.ordered_family_ids,
                        strict=True,
                    )
                ),
            },
        )
        membership = gemma_progressive_panel_membership_receipt_sha256(
            role=_ROLE,
            manifest_sha256=manifest,
            family_by_example=dict(
                zip(
                    self.ordered_prompt_sha256s,
                    self.ordered_family_ids,
                    strict=True,
                )
            ),
        )
        object.__setattr__(self, "manifest_sha256", manifest)
        object.__setattr__(
            self,
            "membership_receipt_sha256",
            membership,
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _domain_sha256(_RECEIPT_DOMAIN, self._payload()),
        )

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(sorted(GENERATOR_INNOVATION_FAMILIES))

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "format_version": 1,
            "panel_id": GENERATOR_INNOVATION_PANEL_ID,
            "role": _ROLE,
            "plan": {
                "plan_sha256": self.plan_sha256,
                "plan_file_sha256": self.plan_file_sha256,
            },
            "lineage": {
                "expanded_fit_corpus_artifact_sha256": (
                    self.expanded_fit_corpus_artifact_sha256
                ),
                "expanded_fit_corpus_file_sha256": (
                    self.expanded_fit_corpus_file_sha256
                ),
                "prior_occupancy_panel_artifact_sha256": (
                    self.prior_occupancy_panel_artifact_sha256
                ),
                "prior_occupancy_panel_file_sha256": (
                    self.prior_occupancy_panel_file_sha256
                ),
                "forbidden_assessment_manifest_sha256s": (
                    self.forbidden_assessment_manifest_sha256s
                ),
            },
            "panel": {
                "role_input_file_sha256": self.role_input_file_sha256,
                "example_count": 16,
                "family_count": 8,
                "examples_per_family": 2,
                "family_ids": self.family_ids,
                "ordered_members": tuple(
                    {
                        "prompt_sha256": prompt_sha256,
                        "family_id": family_id,
                    }
                    for prompt_sha256, family_id in zip(
                        self.ordered_prompt_sha256s,
                        self.ordered_family_ids,
                        strict=True,
                    )
                ),
            },
            "disjointness": {
                "expanded_fit_prompt_overlap_count": 0,
                "expanded_fit_family_overlap_count": 0,
                "prior_occupancy_prompt_overlap_count": 0,
                "prior_occupancy_family_overlap_count": 0,
                "calibration_b_prompt_overlap_count": 0,
                "calibration_b_family_overlap_count": 0,
                "all_identity_checks_passed": True,
            },
            "policy": {
                "plan_frozen_before_prompt_publication": True,
                "role": "new_family_disjoint_derivative_fit_only",
                "nested_outer_split": "leave_one_family_out",
                "selection_or_guard_authorized": False,
                "calibration_b_authorized": False,
            },
            "safety": {
                "prompt_text_in_receipt": False,
                "token_ids_in_receipt": False,
                "activation_rows_in_receipt": False,
                "gradient_rows_in_receipt": False,
                "prior_prompt_payload_opened": False,
                "calibration_b_payload_opened": False,
                "prompt_blind_hashes_only": True,
            },
            "manifest_sha256": self.manifest_sha256,
            "membership_receipt_sha256": (
                self.membership_receipt_sha256
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> Gemma3L3L4GeneratorInnovationPanelReceipt:
        try:
            plan = raw["plan"]
            lineage = raw["lineage"]
            panel = raw["panel"]
            if (
                not isinstance(plan, Mapping)
                or not isinstance(lineage, Mapping)
                or not isinstance(panel, Mapping)
            ):
                raise TypeError("receipt sections must be mappings")
            members = panel["ordered_members"]
            if not isinstance(members, list):
                raise TypeError("receipt ordered members must be a list")
            receipt = cls(
                plan_sha256=str(plan["plan_sha256"]),
                plan_file_sha256=str(plan["plan_file_sha256"]),
                expanded_fit_corpus_artifact_sha256=str(
                    lineage["expanded_fit_corpus_artifact_sha256"]
                ),
                expanded_fit_corpus_file_sha256=str(
                    lineage["expanded_fit_corpus_file_sha256"]
                ),
                prior_occupancy_panel_artifact_sha256=str(
                    lineage["prior_occupancy_panel_artifact_sha256"]
                ),
                prior_occupancy_panel_file_sha256=str(
                    lineage["prior_occupancy_panel_file_sha256"]
                ),
                forbidden_assessment_manifest_sha256s=tuple(
                    lineage["forbidden_assessment_manifest_sha256s"]
                ),
                role_input_file_sha256=str(
                    panel["role_input_file_sha256"]
                ),
                ordered_prompt_sha256s=tuple(
                    str(member["prompt_sha256"])
                    for member in members
                ),
                ordered_family_ids=tuple(
                    str(member["family_id"])
                    for member in members
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
                "generator innovation receipt is malformed"
            ) from error
        expected = receipt.to_dict()
        if dict(raw) != json.loads(_canonical_bytes(expected).decode("utf-8")):
            raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
                "generator innovation receipt metadata or hash differs"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class _AuthenticatedPreparationContext:
    plan_sha256: str
    plan_file_sha256: str
    expanded: Gemma3L3L4ProgressiveACorpusArtifact
    expanded_file_sha256: str
    prior: Gemma3L3L4IterativeOccupancySelectionPanelArtifact
    prior_file_sha256: str
    forbidden_manifest_sha256s: tuple[str, ...]


def _authenticate_context(
    *,
    plan_path: Path | str,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
    expanded_fit_corpus_path: Path | str,
    expected_expanded_fit_corpus_artifact_sha256: str,
    expected_expanded_fit_corpus_file_sha256: str,
    prior_occupancy_panel_path: Path | str,
    expected_prior_occupancy_panel_artifact_sha256: str,
    expected_prior_occupancy_panel_file_sha256: str,
) -> _AuthenticatedPreparationContext:
    """Authenticate all prompt-free inputs before private bytes are created."""

    if (
        expected_plan_sha256
        != FROZEN_GENERATOR_INNOVATION_PLAN_SHA256
        or expected_plan_file_sha256
        != FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
    ):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "expected plan hashes differ from the preregistered plan"
        )
    if (
        expected_expanded_fit_corpus_artifact_sha256
        != _EXPANDED_FIT_CORPUS_ARTIFACT_SHA256
        or expected_expanded_fit_corpus_file_sha256
        != _EXPANDED_FIT_CORPUS_FILE_SHA256
        or expected_prior_occupancy_panel_artifact_sha256
        != _PRIOR_OCCUPANCY_PANEL_ARTIFACT_SHA256
        or expected_prior_occupancy_panel_file_sha256
        != _PRIOR_OCCUPANCY_PANEL_FILE_SHA256
    ):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "expected development lineage differs from the frozen rung"
        )

    plan_encoded = _read_bytes(Path(plan_path), label="generator plan")
    plan_file_sha256 = _file_sha256(plan_encoded)
    if plan_file_sha256 != expected_plan_file_sha256:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator plan file SHA-256 differs"
        )
    plan = _decode_mapping(plan_encoded, label="generator plan")
    try:
        validate_gemma_iterative_generator_innovation_plan(plan)
    except (TypeError, ValueError) as error:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator plan failed standalone validation"
        ) from error
    if plan.get("plan_sha256") != expected_plan_sha256:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator plan logical SHA-256 differs"
        )
    decision = plan.get("decision")
    audit = plan.get("audit")
    if (
        not isinstance(decision, Mapping)
        or decision.get("next_step")
        != (
            "prepare_then_collect_preregistered_new_family_disjoint_"
            "generator_innovation_panel"
        )
        or decision.get("new_family_disjoint_panel_opened") is not False
        or not isinstance(audit, Mapping)
        or audit.get("basis_fixed_before_new_panel") is not True
        or audit.get("feature_fixed_before_new_panel") is not True
        or audit.get("new_panel_prompts_seen") is not False
    ):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator plan does not authorize panel preparation"
        )

    expanded_encoded = _read_bytes(
        Path(expanded_fit_corpus_path),
        label="expanded fit corpus",
    )
    expanded_file_sha256 = _file_sha256(expanded_encoded)
    if expanded_file_sha256 != expected_expanded_fit_corpus_file_sha256:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "expanded fit corpus file SHA-256 differs"
        )
    expanded = _load_progressive_a_artifact(
        expanded_fit_corpus_path,
        expected_artifact_sha256=(
            expected_expanded_fit_corpus_artifact_sha256
        ),
    )

    prior_encoded = _read_bytes(
        Path(prior_occupancy_panel_path),
        label="prior occupancy panel",
    )
    prior_file_sha256 = _file_sha256(prior_encoded)
    if prior_file_sha256 != expected_prior_occupancy_panel_file_sha256:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "prior occupancy panel file SHA-256 differs"
        )
    prior = load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
        prior_occupancy_panel_path,
        expected_artifact_sha256=(
            expected_prior_occupancy_panel_artifact_sha256
        ),
    )
    if (
        prior.expanded_fit_lineage.expanded_corpus_artifact_sha256
        != expanded.artifact_sha256
        or prior.expanded_fit_lineage.tokenizer_contract_sha256
        != expanded.tokenizer_contract_sha256
    ):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "prior occupancy panel belongs to another expanded corpus"
        )

    forbidden_manifest, forbidden_prompts, forbidden_families = (
        _prompt_blind_forbidden_binding()
    )
    forbidden_manifests = (forbidden_manifest,)
    if (
        expanded.forbidden_assessment_manifest_sha256s
        != forbidden_manifests
        or prior.expanded_fit_lineage
        .forbidden_assessment_manifest_sha256s
        != forbidden_manifests
    ):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "development lineage differs from prompt-blind Calibration-B"
        )

    prompt_ids = set(_prompt_sha256s())
    family_ids = set(GENERATOR_INNOVATION_FAMILIES)
    expanded_prompts = {
        prompt_sha256
        for view in expanded.role_views
        for prompt_sha256 in view.ordered_prompt_sha256s
    }
    expanded_families = {
        family_id
        for view in expanded.role_views
        for family_id in view.family_ids
    }
    prior_prompts = (
        set(prior.prior_panel.ordered_prompt_sha256s)
        | set(prior.ordered_prompt_sha256s)
    )
    prior_families = (
        set(prior.prior_panel.ordered_family_ids)
        | set(prior.ordered_family_ids)
    )
    if (
        prompt_ids & expanded_prompts
        or family_ids & expanded_families
        or prompt_ids & prior_prompts
        or family_ids & prior_families
        or prompt_ids & set(forbidden_prompts)
        or family_ids & set(forbidden_families)
    ):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "new generator panel overlaps an occupied prompt or family"
        )

    return _AuthenticatedPreparationContext(
        plan_sha256=expected_plan_sha256,
        plan_file_sha256=plan_file_sha256,
        expanded=expanded,
        expanded_file_sha256=expanded_file_sha256,
        prior=prior,
        prior_file_sha256=prior_file_sha256,
        forbidden_manifest_sha256s=forbidden_manifests,
    )


def _build_receipt(
    *,
    context: _AuthenticatedPreparationContext,
    role_input_file_sha256: str,
) -> Gemma3L3L4GeneratorInnovationPanelReceipt:
    return Gemma3L3L4GeneratorInnovationPanelReceipt(
        plan_sha256=context.plan_sha256,
        plan_file_sha256=context.plan_file_sha256,
        expanded_fit_corpus_artifact_sha256=(
            context.expanded.artifact_sha256
        ),
        expanded_fit_corpus_file_sha256=context.expanded_file_sha256,
        prior_occupancy_panel_artifact_sha256=(
            context.prior.artifact_sha256
        ),
        prior_occupancy_panel_file_sha256=context.prior_file_sha256,
        forbidden_assessment_manifest_sha256s=(
            context.forbidden_manifest_sha256s
        ),
        role_input_file_sha256=role_input_file_sha256,
        ordered_prompt_sha256s=_prompt_sha256s(),
        ordered_family_ids=GENERATOR_INNOVATION_FAMILY_SCHEDULE,
    )


def _stage_bytes(destination: Path, encoded: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_pair_once(
    *,
    private_path: Path,
    private_encoded: bytes,
    receipt_path: Path,
    receipt_encoded: bytes,
) -> None:
    if private_path.resolve() == receipt_path.resolve():
        raise ValueError("private role and prompt-free receipt paths must differ")
    if private_path.exists() or receipt_path.exists():
        raise FileExistsError(
            "refusing to overwrite generator innovation panel outputs"
        )
    private_temporary: Path | None = None
    receipt_temporary: Path | None = None
    private_installed = False
    receipt_installed = False
    try:
        private_temporary = _stage_bytes(private_path, private_encoded)
        receipt_temporary = _stage_bytes(receipt_path, receipt_encoded)
        try:
            os.link(private_temporary, private_path)
            private_installed = True
            os.link(receipt_temporary, receipt_path)
            receipt_installed = True
        except FileExistsError as error:
            if private_installed and not receipt_installed:
                private_path.unlink(missing_ok=True)
            raise FileExistsError(
                "refusing to overwrite generator innovation panel outputs"
            ) from error
        except BaseException:
            if private_installed and not receipt_installed:
                private_path.unlink(missing_ok=True)
            raise
    finally:
        if private_temporary is not None:
            private_temporary.unlink(missing_ok=True)
        if receipt_temporary is not None:
            receipt_temporary.unlink(missing_ok=True)


def prepare_gemma3_l3_l4_generator_innovation_panel(
    *,
    plan_path: Path | str = DEFAULT_GENERATOR_INNOVATION_PLAN,
    expected_plan_sha256: str = FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    expected_plan_file_sha256: str = (
        FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
    ),
    expanded_fit_corpus_path: Path | str = DEFAULT_EXPANDED_FIT_CORPUS,
    expected_expanded_fit_corpus_artifact_sha256: str = (
        _EXPANDED_FIT_CORPUS_ARTIFACT_SHA256
    ),
    expected_expanded_fit_corpus_file_sha256: str = (
        _EXPANDED_FIT_CORPUS_FILE_SHA256
    ),
    prior_occupancy_panel_path: Path | str = DEFAULT_PRIOR_OCCUPANCY_PANEL,
    expected_prior_occupancy_panel_artifact_sha256: str = (
        _PRIOR_OCCUPANCY_PANEL_ARTIFACT_SHA256
    ),
    expected_prior_occupancy_panel_file_sha256: str = (
        _PRIOR_OCCUPANCY_PANEL_FILE_SHA256
    ),
    private_output: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT
    ),
    receipt_output: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT
    ),
) -> dict[str, object]:
    """Authenticate the fixed rung, then publish private/public outputs once."""

    private_path = Path(private_output)
    receipt_path = Path(receipt_output)
    if private_path.exists() or receipt_path.exists():
        raise FileExistsError(
            "refusing to overwrite generator innovation panel outputs"
        )
    context = _authenticate_context(
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_plan_file_sha256=expected_plan_file_sha256,
        expanded_fit_corpus_path=expanded_fit_corpus_path,
        expected_expanded_fit_corpus_artifact_sha256=(
            expected_expanded_fit_corpus_artifact_sha256
        ),
        expected_expanded_fit_corpus_file_sha256=(
            expected_expanded_fit_corpus_file_sha256
        ),
        prior_occupancy_panel_path=prior_occupancy_panel_path,
        expected_prior_occupancy_panel_artifact_sha256=(
            expected_prior_occupancy_panel_artifact_sha256
        ),
        expected_prior_occupancy_panel_file_sha256=(
            expected_prior_occupancy_panel_file_sha256
        ),
    )

    # No private prompt bytes are serialized until every prompt-free source and
    # all three disjointness boundaries have passed above.
    private_encoded = _canonical_bytes(_role_payload())
    receipt = _build_receipt(
        context=context,
        role_input_file_sha256=_file_sha256(private_encoded),
    )
    receipt_encoded = _canonical_bytes(receipt.to_dict())
    _publish_pair_once(
        private_path=private_path,
        private_encoded=private_encoded,
        receipt_path=receipt_path,
        receipt_encoded=receipt_encoded,
    )
    return receipt.to_dict()


def load_gemma3_l3_l4_generator_innovation_panel_receipt(
    path: Path | str,
    *,
    plan_path: Path | str = DEFAULT_GENERATOR_INNOVATION_PLAN,
    expanded_fit_corpus_path: Path | str = DEFAULT_EXPANDED_FIT_CORPUS,
    prior_occupancy_panel_path: Path | str = DEFAULT_PRIOR_OCCUPANCY_PANEL,
) -> Gemma3L3L4GeneratorInnovationPanelReceipt:
    """Load a receipt and reauthenticate its exact plan and prior identities."""

    encoded = _read_bytes(Path(path), label="generator innovation receipt")
    raw = _decode_mapping(encoded, label="generator innovation receipt")
    if _canonical_bytes(raw) != encoded:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator innovation receipt is not canonical JSON"
        )
    receipt = Gemma3L3L4GeneratorInnovationPanelReceipt.from_dict(raw)
    context = _authenticate_context(
        plan_path=plan_path,
        expected_plan_sha256=receipt.plan_sha256,
        expected_plan_file_sha256=receipt.plan_file_sha256,
        expanded_fit_corpus_path=expanded_fit_corpus_path,
        expected_expanded_fit_corpus_artifact_sha256=(
            receipt.expanded_fit_corpus_artifact_sha256
        ),
        expected_expanded_fit_corpus_file_sha256=(
            receipt.expanded_fit_corpus_file_sha256
        ),
        prior_occupancy_panel_path=prior_occupancy_panel_path,
        expected_prior_occupancy_panel_artifact_sha256=(
            receipt.prior_occupancy_panel_artifact_sha256
        ),
        expected_prior_occupancy_panel_file_sha256=(
            receipt.prior_occupancy_panel_file_sha256
        ),
    )
    rebuilt = _build_receipt(
        context=context,
        role_input_file_sha256=receipt.role_input_file_sha256,
    )
    if rebuilt != receipt:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator innovation receipt differs from authenticated sources"
        )
    return receipt


def load_gemma3_l3_l4_generator_innovation_role_input(
    path: Path | str,
    *,
    receipt: Gemma3L3L4GeneratorInnovationPanelReceipt,
) -> Gemma3L3L4ProgressiveARolePrompts:
    """Open only the exact private fit role bound by a validated receipt."""

    if not isinstance(receipt, Gemma3L3L4GeneratorInnovationPanelReceipt):
        raise TypeError("receipt must be an authenticated panel receipt")
    encoded = _read_bytes(Path(path), label="generator innovation role input")
    if (
        _file_sha256(encoded) != receipt.role_input_file_sha256
        or encoded != _canonical_bytes(_role_payload())
    ):
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator innovation role input differs from its receipt"
        )
    role = Gemma3L3L4ProgressiveARolePrompts(
        corpus_id=GENERATOR_INNOVATION_PANEL_ID,
        profile="pilot",
        role=_ROLE,
        prompts=GENERATOR_INNOVATION_PROMPTS,
        family_ids=GENERATOR_INNOVATION_FAMILY_SCHEDULE,
        source_file_sha256=receipt.role_input_file_sha256,
    )
    if role.ordered_prompt_sha256s != receipt.ordered_prompt_sha256s:
        raise Gemma3L3L4GeneratorInnovationPanelIntegrityError(
            "generator innovation prompt membership drifted"
        )
    return role


def materialize_gemma3_l3_l4_generator_innovation_panel(
    *,
    tokenizer: object,
    receipt: Gemma3L3L4GeneratorInnovationPanelReceipt,
    role_input_path: Path | str,
    max_length: int,
    device: torch.device,
) -> GemmaProgressivePanel:
    """Use the shared strict tokenizer path for the authenticated fit role."""

    role_input = load_gemma3_l3_l4_generator_innovation_role_input(
        role_input_path,
        receipt=receipt,
    )
    view = Gemma3L3L4ProgressiveARolePreclaimView(
        role=_ROLE,
        manifest_sha256=receipt.manifest_sha256,
        role_input_file_sha256=receipt.role_input_file_sha256,
        example_count=16,
        family_ids=receipt.family_ids,
        ordered_prompt_sha256s=receipt.ordered_prompt_sha256s,
        ordered_family_ids=receipt.ordered_family_ids,
    )
    panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=role_input,
        view=view,
        max_length=max_length,
        device=device,
        forbidden_manifest_sha256s=(
            receipt.forbidden_assessment_manifest_sha256s
        ),
    )
    if panel.membership_receipt_sha256 != receipt.membership_receipt_sha256:
        raise RuntimeError("materialized generator innovation membership drifted")
    return panel
