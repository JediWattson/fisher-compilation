"""Fresh Iteration-5 selection membership and one-use opening boundary.

The public artifact in this module is prompt-free.  It commits one fixed
sixteen-example, eight-family ``calibration_a_selection`` panel against:

* the expanded development-corpus lineage;
* the earlier H4 damping selection panel; and
* the frozen prompt-blind assessment identities.

Raw prompts live only in the local role-input file.  That file can be opened
through :class:`Gemma3L3L4IterativeOccupancySelectionPanelSource` once, and
only after the exact public artifact has an authenticated durable claim.
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

import torch

from .gemma3_l3_l4_h4_damping_selection_panel import (
    Gemma3L3L4H4DampingExpandedFitLineage,
    Gemma3L3L4H4DampingSelectionPanelArtifact,
    _prompt_blind_forbidden_binding,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveARolePreclaimView,
    Gemma3L3L4ProgressiveARolePrompts,
)
from .gemma3_l3_l4_progressive_worker import (
    GemmaProgressivePanel,
    gemma_progressive_panel_membership_receipt_sha256,
)


__all__ = [
    "GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_PANEL_ID",
    "GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE",
    "ITERATIVE_OCCUPANCY_SELECTION_FAMILIES",
    "ITERATIVE_OCCUPANCY_SELECTION_FAMILY_SCHEDULE",
    "ITERATIVE_OCCUPANCY_SELECTION_PROMPTS",
    "Gemma3L3L4IterativeOccupancySelectionClaim",
    "Gemma3L3L4IterativeOccupancySelectionClosedError",
    "Gemma3L3L4IterativeOccupancySelectionIntegrityError",
    "Gemma3L3L4IterativeOccupancySelectionPanelArtifact",
    "Gemma3L3L4IterativeOccupancySelectionPanelSource",
    "Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding",
    "Gemma3L3L4IterativeOccupancySelectionRoleInput",
    "claim_gemma3_l3_l4_iterative_occupancy_selection_panel",
    "freeze_gemma3_l3_l4_iterative_occupancy_selection_panel",
    "load_gemma3_l3_l4_iterative_occupancy_selection_claim",
    "load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact",
    "load_gemma3_l3_l4_iterative_occupancy_selection_role_input",
    "materialize_gemma3_l3_l4_iterative_occupancy_selection_panel",
    "write_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact",
    "write_gemma3_l3_l4_iterative_occupancy_selection_role_input",
]


GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_PANEL_ID = (
    "gemma3-l3-l4-iterative-occupancy-selection-v1"
)
GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE = (
    "calibration_a_selection"
)
_ROLE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_role"
)
_PANEL_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_panel"
)
_PRIOR_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_occupancy_prior_panel"
)
_CLAIM_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_claim"
)
_MANIFEST_DOMAIN = b"fisher-graph:iterative-occupancy-selection-manifest:v1\0"
_PRIOR_DOMAIN = b"fisher-graph:iterative-occupancy-prior-panel:v1\0"
_ARTIFACT_DOMAIN = b"fisher-graph:iterative-occupancy-selection-artifact:v1\0"
_CLAIM_DOMAIN = b"fisher-graph:iterative-occupancy-selection-claim:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024

ITERATIVE_OCCUPANCY_SELECTION_FAMILIES = (
    "iterative-confirm-v1-chronology-causation",
    "iterative-confirm-v1-conservation-accounting",
    "iterative-confirm-v1-cross-condition-consistency",
    "iterative-confirm-v1-hypothetical-intervention",
    "iterative-confirm-v1-multilevel-composition",
    "iterative-confirm-v1-perspective-translation",
    "iterative-confirm-v1-probability-revision",
    "iterative-confirm-v1-resource-tradeoffs",
)
ITERATIVE_OCCUPANCY_SELECTION_FAMILY_SCHEDULE = (
    ITERATIVE_OCCUPANCY_SELECTION_FAMILIES
    + ITERATIVE_OCCUPANCY_SELECTION_FAMILIES
)

# Two surface-disjoint rounds probe the same eight capabilities.  These were
# frozen before the Iteration-5 occupancy candidate is fit or evaluated.
ITERATIVE_OCCUPANCY_SELECTION_PROMPTS = (
    (
        "A museum's motion alarm sounded at 9:18. The loading door opened at "
        "9:12, a delivery cart crossed the east hall at 9:15, and the east "
        "sensor detected motion at 9:17. The west sensor stayed quiet. Give "
        "the most defensible causal chain and identify what the west-sensor "
        "evidence rules out."
    ),
    (
        "A library begins with 240 books split among three rooms. Room A "
        "sends 18 books to B, B sends 11 to C, and C retires 7 damaged books "
        "from the collection. Explain which total is conserved during the "
        "transfers and compute the final collection total."
    ),
    (
        "Four talks J, K, L, and M occupy four consecutive slots. J must be "
        "before L, K cannot be adjacent to M, and M must be earlier than J. "
        "Determine whether the order M, K, J, L is allowed, citing every "
        "constraint rather than judging from one rule alone."
    ),
    (
        "A greenhouse trial produced 34 seedlings after using enriched soil "
        "and twelve hours of light. A matched earlier trial used ordinary "
        "soil, the same light, and produced 26. If the new trial had used "
        "ordinary soil while all else stayed fixed, what comparison supports "
        "the best estimate, and what uncertainty remains?"
    ),
    (
        "A company has two divisions. North's teams report profits of 8 and "
        "a loss of 3 million dollars; South's teams report profits of 4, 2, "
        "and 1 million. Compute each division's result and then the company "
        "result, preserving the team-to-division hierarchy."
    ),
    (
        "Lee faces north while Ana faces east. A sculpture is directly to "
        "Lee's right and directly behind Ana. Describe the sculpture's "
        "compass direction from each person's perspective and reconcile the "
        "two descriptions."
    ),
    (
        "A box is equally likely to be red or blue. A detector says red 80% "
        "of the time for red boxes and 20% of the time for blue boxes. It "
        "says red. Without needing decimal precision, explain how the "
        "evidence changes the relative plausibility of the two colors."
    ),
    (
        "A town has $90,000 for three repairs. The bridge costs $50,000, the "
        "clinic roof $35,000, and the playground $20,000. The bridge is "
        "mandatory, and the roof must be funded before the playground. List "
        "the feasible complete choices and the unused money for each."
    ),
    (
        "A build failed after source generation, compilation, unit tests, and "
        "packaging. Source generation and compilation completed, unit tests "
        "reported a missing fixture, and packaging never began. Identify the "
        "earliest supported cause of the failed build and distinguish it "
        "from downstream consequences."
    ),
    (
        "Three connected tanks contain 45, 30, and 25 liters. Ten liters move "
        "from the first tank to the second, then 12 move from the second to "
        "the third, and 4 liters spill. Show how internal transfers affect "
        "the system total and give the final total."
    ),
    (
        "Mina, Owen, Priya, and Rui sit in a row. Mina cannot sit at an end, "
        "Owen must sit left of Priya, and Rui cannot sit beside Owen. Test "
        "the arrangement Owen, Mina, Priya, Rui against all requirements and "
        "give one valid arrangement if it fails."
    ),
    (
        "Jordan received a scholarship after submitting early and earning a "
        "high score. Among otherwise similar applicants, late high scorers "
        "were not funded, while early lower scorers sometimes were. If "
        "Jordan had submitted late with the same score, what outcome is best "
        "supported and which comparison matters most?"
    ),
    (
        "A bill contains three titles. Title I has sections costing 5 and 7 "
        "million dollars. Title II has one section saving 4 million. Title "
        "III has sections costing 3, 2, and 1 million. Compute each title's "
        "net effect and then the bill's total effect."
    ),
    (
        "On a map, the harbor is east of the station and the library is "
        "north of the harbor. A traveler turns the map so north points left. "
        "Describe the library's location relative to the station both in "
        "compass terms and on the turned page."
    ),
    (
        "Rain is initially considered twice as likely as no rain. A reliable "
        "forecast is three times more likely to issue a warning on rainy "
        "days than on dry days, and it issues a warning. Explain the updated "
        "odds qualitatively and calculate their ratio."
    ),
    (
        "A clinic can schedule 12 staff-hours. Intake requires 5, vaccination "
        "requires 4, and records cleanup requires 3. Intake is required; "
        "vaccination may be scheduled only if at least 2 hours remain for "
        "records. Compare all feasible combinations and identify those that "
        "use every available hour."
    ),
)


class Gemma3L3L4IterativeOccupancySelectionIntegrityError(RuntimeError):
    """A role input, public artifact, or durable claim failed validation."""


class Gemma3L3L4IterativeOccupancySelectionClosedError(RuntimeError):
    """The claim-gated local prompt source has already been consumed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _file_sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_mapping(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    if not path.is_file():
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            f"{label} must be a regular file"
        )
    encoded = path.read_bytes()
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            f"{label} has an invalid size"
        )
    try:
        raw = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            f"{label} is not canonical ASCII JSON"
        ) from error
    if not isinstance(raw, dict) or _canonical_bytes(raw) != encoded:
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            f"{label} is not a canonical JSON object"
        )
    return encoded, raw


def _write_exclusive(path: Path, encoded: bytes, *, durable: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)
        if durable:
            handle.flush()
            os.fsync(handle.fileno())
    if durable:
        descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return _file_sha256(encoded)


def _role_payload() -> dict[str, object]:
    return {
        "schema": _ROLE_SCHEMA,
        "format_version": 1,
        "panel_id": GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_PANEL_ID,
        "role": GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE,
        "prompts": ITERATIVE_OCCUPANCY_SELECTION_PROMPTS,
        "family_ids": ITERATIVE_OCCUPANCY_SELECTION_FAMILY_SCHEDULE,
    }


@dataclass(frozen=True, slots=True)
class Gemma3L3L4IterativeOccupancySelectionRoleInput:
    prompts: tuple[str, ...]
    family_ids: tuple[str, ...]
    source_file_sha256: str
    ordered_prompt_sha256s: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.source_file_sha256, label="role input file")
        if self.prompts != ITERATIVE_OCCUPANCY_SELECTION_PROMPTS:
            raise ValueError("role prompts differ from the frozen panel")
        if self.family_ids != ITERATIVE_OCCUPANCY_SELECTION_FAMILY_SCHEDULE:
            raise ValueError("role family schedule differs from the frozen panel")
        prompt_hashes = tuple(
            gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
            for prompt in self.prompts
        )
        if len(set(prompt_hashes)) != 16:
            raise ValueError("frozen role prompts must have unique identities")
        object.__setattr__(self, "ordered_prompt_sha256s", prompt_hashes)


def write_gemma3_l3_l4_iterative_occupancy_selection_role_input(
    path: Path | str,
) -> str:
    """Publish the exact private Iteration-5 prompt source once."""

    return _write_exclusive(
        Path(path),
        _canonical_bytes(_role_payload()),
        durable=False,
    )


def load_gemma3_l3_l4_iterative_occupancy_selection_role_input(
    path: Path | str,
) -> Gemma3L3L4IterativeOccupancySelectionRoleInput:
    encoded, raw = _read_mapping(Path(path), label="occupancy role input")
    if raw != json.loads(_canonical_bytes(_role_payload()).decode("ascii")):
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            "occupancy role input differs from the frozen protocol"
        )
    return Gemma3L3L4IterativeOccupancySelectionRoleInput(
        prompts=tuple(raw["prompts"]),  # type: ignore[arg-type]
        family_ids=tuple(raw["family_ids"]),  # type: ignore[arg-type]
        source_file_sha256=_file_sha256(encoded),
    )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding:
    expanded_fit_lineage_receipt_sha256: str
    artifact_sha256: str
    manifest_sha256: str
    membership_receipt_sha256: str
    role_input_file_sha256: str
    ordered_prompt_sha256s: tuple[str, ...]
    ordered_family_ids: tuple[str, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "expanded_fit_lineage_receipt_sha256",
            "artifact_sha256",
            "manifest_sha256",
            "membership_receipt_sha256",
            "role_input_file_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            len(self.ordered_prompt_sha256s) != 16
            or len(set(self.ordered_prompt_sha256s)) != 16
            or len(self.ordered_family_ids) != 16
        ):
            raise ValueError("prior selection binding must contain 16 members")
        for value in self.ordered_prompt_sha256s:
            _require_sha256(value, label="prior prompt")
        object.__setattr__(
            self,
            "receipt_sha256",
            _domain_sha256(_PRIOR_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _PRIOR_SCHEMA,
            "format_version": 1,
            "expanded_fit_lineage_receipt_sha256": (
                self.expanded_fit_lineage_receipt_sha256
            ),
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "membership_receipt_sha256": self.membership_receipt_sha256,
            "role_input_file_sha256": self.role_input_file_sha256,
            "ordered_prompt_sha256s": self.ordered_prompt_sha256s,
            "ordered_family_ids": self.ordered_family_ids,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding:
        try:
            binding = cls(
                expanded_fit_lineage_receipt_sha256=str(
                    raw["expanded_fit_lineage_receipt_sha256"]
                ),
                artifact_sha256=str(raw["artifact_sha256"]),
                manifest_sha256=str(raw["manifest_sha256"]),
                membership_receipt_sha256=str(
                    raw["membership_receipt_sha256"]
                ),
                role_input_file_sha256=str(raw["role_input_file_sha256"]),
                ordered_prompt_sha256s=tuple(
                    raw["ordered_prompt_sha256s"]  # type: ignore[arg-type]
                ),
                ordered_family_ids=tuple(
                    raw["ordered_family_ids"]  # type: ignore[arg-type]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
                "prior panel binding is invalid"
            ) from error
        if raw.get("receipt_sha256") != binding.receipt_sha256:
            raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
                "prior panel binding receipt differs"
            )
        return binding


def _prior_binding(
    prior: Gemma3L3L4H4DampingSelectionPanelArtifact,
) -> Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding:
    if not isinstance(prior, Gemma3L3L4H4DampingSelectionPanelArtifact):
        raise TypeError("prior panel must be an authenticated damping panel")
    return Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding(
        expanded_fit_lineage_receipt_sha256=(
            prior.expanded_fit_lineage.receipt_sha256
        ),
        artifact_sha256=prior.artifact_sha256,
        manifest_sha256=prior.manifest_sha256,
        membership_receipt_sha256=prior.membership_receipt_sha256,
        role_input_file_sha256=prior.selection_role_input_file_sha256,
        ordered_prompt_sha256s=prior.ordered_prompt_sha256s,
        ordered_family_ids=prior.ordered_family_ids,
    )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4IterativeOccupancySelectionPanelArtifact:
    expanded_fit_lineage: Gemma3L3L4H4DampingExpandedFitLineage
    prior_panel: Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding
    selection_plan_sha256: str
    role_input_file_sha256: str
    ordered_prompt_sha256s: tuple[str, ...]
    ordered_family_ids: tuple[str, ...]
    manifest_sha256: str = field(init=False)
    membership_receipt_sha256: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.expanded_fit_lineage,
            Gemma3L3L4H4DampingExpandedFitLineage,
        ):
            raise TypeError("expanded lineage must be authenticated")
        if not isinstance(
            self.prior_panel,
            Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding,
        ):
            raise TypeError("prior panel binding must be authenticated")
        _require_sha256(self.selection_plan_sha256, label="selection plan")
        _require_sha256(self.role_input_file_sha256, label="role input file")
        expected_hashes = tuple(
            gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
            for prompt in ITERATIVE_OCCUPANCY_SELECTION_PROMPTS
        )
        if self.ordered_prompt_sha256s != expected_hashes:
            raise ValueError("selection prompt hashes differ from frozen prompts")
        if self.ordered_family_ids != ITERATIVE_OCCUPANCY_SELECTION_FAMILY_SCHEDULE:
            raise ValueError("selection family schedule differs")
        lineage = self.expanded_fit_lineage
        if (
            self.prior_panel.expanded_fit_lineage_receipt_sha256
            != lineage.receipt_sha256
        ):
            raise ValueError("prior panel and expanded lineage differ")
        forbidden_manifest, forbidden_prompts, forbidden_families = (
            _prompt_blind_forbidden_binding()
        )
        if lineage.forbidden_assessment_manifest_sha256s != (
            forbidden_manifest,
        ):
            raise ValueError("prompt-blind assessment lineage differs")
        occupied_prompts = (
            set(lineage.occupied_development_prompt_sha256s)
            | set(self.prior_panel.ordered_prompt_sha256s)
            | set(forbidden_prompts)
        )
        occupied_families = (
            set(lineage.occupied_development_family_ids)
            | set(self.prior_panel.ordered_family_ids)
            | set(forbidden_families)
        )
        if set(self.ordered_prompt_sha256s) & occupied_prompts:
            raise ValueError("selection prompts overlap occupied identities")
        if set(self.ordered_family_ids) & occupied_families:
            raise ValueError("selection families overlap occupied identities")
        manifest = _domain_sha256(
            _MANIFEST_DOMAIN,
            {
                "panel_id": (
                    GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_PANEL_ID
                ),
                "role": GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE,
                "lineage_receipt_sha256": lineage.receipt_sha256,
                "prior_panel_receipt_sha256": self.prior_panel.receipt_sha256,
                "selection_plan_sha256": self.selection_plan_sha256,
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
        occupied_manifests = (
            set(lineage.occupied_development_manifest_sha256s)
            | set(lineage.forbidden_assessment_manifest_sha256s)
            | {self.prior_panel.manifest_sha256}
        )
        if manifest in occupied_manifests:
            raise ValueError("selection manifest overlaps an occupied identity")
        membership = gemma_progressive_panel_membership_receipt_sha256(
            role=GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE,
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
        object.__setattr__(self, "membership_receipt_sha256", membership)
        object.__setattr__(
            self,
            "artifact_sha256",
            _domain_sha256(_ARTIFACT_DOMAIN, self._payload()),
        )

    @property
    def family_ids(self) -> tuple[str, ...]:
        return ITERATIVE_OCCUPANCY_SELECTION_FAMILIES

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _PANEL_SCHEMA,
            "format_version": 1,
            "panel_id": GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_PANEL_ID,
            "role": GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE,
            "expanded_fit_lineage": self.expanded_fit_lineage.to_dict(),
            "prior_panel": self.prior_panel.to_dict(),
            "selection_plan_sha256": self.selection_plan_sha256,
            "selection": {
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
            "policy": {
                "opening": "claimed_one_shot_selection",
                "maximum_panel_open_count": 1,
                "candidate_plan_frozen_before_open": True,
                "adaptive_candidate_changes_authorized": False,
                "guard_authorized": False,
                "assessment_authorized": False,
            },
            "safety": {
                "prompt_text_in_artifact": False,
                "token_ids_in_artifact": False,
                "activation_rows_in_artifact": False,
                "gradient_rows_in_artifact": False,
                "prior_selection_payload_opened": False,
                "assessment_payload_opened": False,
                "prompt_blind_assessment_consulted": True,
            },
            "manifest_sha256": self.manifest_sha256,
            "membership_receipt_sha256": self.membership_receipt_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> Gemma3L3L4IterativeOccupancySelectionPanelArtifact:
        try:
            selection = raw["selection"]
            assert isinstance(selection, Mapping)
            members = selection["ordered_members"]
            assert isinstance(members, list)
            lineage = Gemma3L3L4H4DampingExpandedFitLineage.from_dict(
                raw["expanded_fit_lineage"]  # type: ignore[arg-type]
            )
            prior = (
                Gemma3L3L4IterativeOccupancySelectionPriorPanelBinding.from_dict(
                    raw["prior_panel"]  # type: ignore[arg-type]
                )
            )
            artifact = cls(
                expanded_fit_lineage=lineage,
                prior_panel=prior,
                selection_plan_sha256=str(raw["selection_plan_sha256"]),
                role_input_file_sha256=str(
                    selection["role_input_file_sha256"]
                ),
                ordered_prompt_sha256s=tuple(
                    str(member["prompt_sha256"])  # type: ignore[index]
                    for member in members
                ),
                ordered_family_ids=tuple(
                    str(member["family_id"])  # type: ignore[index]
                    for member in members
                ),
            )
        except (
            AssertionError,
            KeyError,
            TypeError,
            ValueError,
            Gemma3L3L4IterativeOccupancySelectionIntegrityError,
        ) as error:
            raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
                "occupancy selection artifact is invalid"
            ) from error
        expected = artifact.to_dict()
        if dict(raw) != json.loads(_canonical_bytes(expected).decode("ascii")):
            raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
                "occupancy selection artifact metadata or hash differs"
            )
        return artifact


def freeze_gemma3_l3_l4_iterative_occupancy_selection_panel(
    *,
    expanded_fit_lineage: Gemma3L3L4H4DampingExpandedFitLineage,
    prior_selection_panel: Gemma3L3L4H4DampingSelectionPanelArtifact,
    selection_plan_sha256: str,
    role_input_path: Path | str,
) -> Gemma3L3L4IterativeOccupancySelectionPanelArtifact:
    """Commit the exact fresh role after all three disjointness checks."""

    opened = load_gemma3_l3_l4_iterative_occupancy_selection_role_input(
        role_input_path
    )
    return Gemma3L3L4IterativeOccupancySelectionPanelArtifact(
        expanded_fit_lineage=expanded_fit_lineage,
        prior_panel=_prior_binding(prior_selection_panel),
        selection_plan_sha256=selection_plan_sha256,
        role_input_file_sha256=opened.source_file_sha256,
        ordered_prompt_sha256s=opened.ordered_prompt_sha256s,
        ordered_family_ids=opened.family_ids,
    )


def write_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
    path: Path | str,
    artifact: Gemma3L3L4IterativeOccupancySelectionPanelArtifact,
) -> str:
    if not isinstance(
        artifact,
        Gemma3L3L4IterativeOccupancySelectionPanelArtifact,
    ):
        raise TypeError("artifact must be an occupancy selection artifact")
    return _write_exclusive(
        Path(path),
        _canonical_bytes(artifact.to_dict()),
        durable=False,
    )


def load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
    path: Path | str,
    *,
    expected_artifact_sha256: str | None = None,
) -> Gemma3L3L4IterativeOccupancySelectionPanelArtifact:
    _encoded, raw = _read_mapping(Path(path), label="occupancy panel artifact")
    artifact = Gemma3L3L4IterativeOccupancySelectionPanelArtifact.from_dict(raw)
    if (
        expected_artifact_sha256 is not None
        and artifact.artifact_sha256
        != _require_sha256(
            expected_artifact_sha256,
            label="expected artifact",
        )
    ):
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            "occupancy panel artifact differs from expected identity"
        )
    return artifact


@dataclass(frozen=True, slots=True)
class Gemma3L3L4IterativeOccupancySelectionClaim:
    path: Path
    artifact_sha256: str
    manifest_sha256: str
    role_input_file_sha256: str
    claim_sha256: str
    claim_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("claim path must be a Path")
        for name in (
            "artifact_sha256",
            "manifest_sha256",
            "role_input_file_sha256",
            "claim_sha256",
            "claim_file_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)


def _claim_payload(
    artifact: Gemma3L3L4IterativeOccupancySelectionPanelArtifact,
) -> dict[str, object]:
    payload = {
        "schema": _CLAIM_SCHEMA,
        "format_version": 1,
        "state": "claimed_before_prompt_materialization",
        "artifact_sha256": artifact.artifact_sha256,
        "manifest_sha256": artifact.manifest_sha256,
        "membership_receipt_sha256": artifact.membership_receipt_sha256,
        "selection_plan_sha256": artifact.selection_plan_sha256,
        "role_input_file_sha256": artifact.role_input_file_sha256,
    }
    payload["claim_sha256"] = _domain_sha256(_CLAIM_DOMAIN, payload)
    return payload


def claim_gemma3_l3_l4_iterative_occupancy_selection_panel(
    path: Path | str,
    *,
    artifact: Gemma3L3L4IterativeOccupancySelectionPanelArtifact,
) -> Gemma3L3L4IterativeOccupancySelectionClaim:
    """Durably and exclusively claim the exact prompt-free commitment."""

    destination = Path(path)
    encoded = _canonical_bytes(_claim_payload(artifact))
    file_sha256 = _write_exclusive(destination, encoded, durable=True)
    return load_gemma3_l3_l4_iterative_occupancy_selection_claim(
        destination,
        artifact=artifact,
        expected_file_sha256=file_sha256,
    )


def load_gemma3_l3_l4_iterative_occupancy_selection_claim(
    path: Path | str,
    *,
    artifact: Gemma3L3L4IterativeOccupancySelectionPanelArtifact,
    expected_file_sha256: str | None = None,
) -> Gemma3L3L4IterativeOccupancySelectionClaim:
    source = Path(path)
    encoded, raw = _read_mapping(source, label="occupancy selection claim")
    expected = _claim_payload(artifact)
    if raw != json.loads(_canonical_bytes(expected).decode("ascii")):
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            "occupancy selection claim differs from the artifact"
        )
    file_sha256 = _file_sha256(encoded)
    if (
        expected_file_sha256 is not None
        and file_sha256
        != _require_sha256(expected_file_sha256, label="claim file")
    ):
        raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
            "occupancy selection claim file hash differs"
        )
    return Gemma3L3L4IterativeOccupancySelectionClaim(
        path=source,
        artifact_sha256=artifact.artifact_sha256,
        manifest_sha256=artifact.manifest_sha256,
        role_input_file_sha256=artifact.role_input_file_sha256,
        claim_sha256=str(raw["claim_sha256"]),
        claim_file_sha256=file_sha256,
    )


class Gemma3L3L4IterativeOccupancySelectionPanelSource:
    """Claim-gated, fail-closed one-use opener for the local prompt file."""

    __slots__ = ("_artifact", "_role_input_path", "_consumed", "_opened")

    def __init__(
        self,
        *,
        artifact: Gemma3L3L4IterativeOccupancySelectionPanelArtifact,
        role_input_path: Path | str,
    ) -> None:
        self._artifact = artifact
        self._role_input_path = Path(role_input_path)
        self._consumed = False
        self._opened = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def artifact(self) -> Gemma3L3L4IterativeOccupancySelectionPanelArtifact:
        return self._artifact

    def open_once(
        self,
        *,
        claim: Gemma3L3L4IterativeOccupancySelectionClaim,
    ) -> Gemma3L3L4IterativeOccupancySelectionRoleInput:
        if self._consumed:
            raise Gemma3L3L4IterativeOccupancySelectionClosedError(
                "occupancy selection source has already been consumed"
            )
        authenticated = (
            load_gemma3_l3_l4_iterative_occupancy_selection_claim(
                claim.path,
                artifact=self._artifact,
                expected_file_sha256=claim.claim_file_sha256,
            )
        )
        if authenticated != claim:
            raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
                "durable claim receipt differs"
            )
        self._consumed = True
        opened = (
            load_gemma3_l3_l4_iterative_occupancy_selection_role_input(
                self._role_input_path
            )
        )
        if (
            opened.source_file_sha256 != self._artifact.role_input_file_sha256
            or opened.ordered_prompt_sha256s
            != self._artifact.ordered_prompt_sha256s
            or opened.family_ids != self._artifact.ordered_family_ids
        ):
            raise Gemma3L3L4IterativeOccupancySelectionIntegrityError(
                "opened occupancy source differs from its commitment"
            )
        self._opened = True
        return opened


def materialize_gemma3_l3_l4_iterative_occupancy_selection_panel(
    *,
    source: Gemma3L3L4IterativeOccupancySelectionPanelSource,
    claim: Gemma3L3L4IterativeOccupancySelectionClaim,
    tokenizer: object,
    max_length: int,
    device: torch.device,
) -> GemmaProgressivePanel:
    """Open once after claim and reuse the strict progressive tokenizer path."""

    opened = source.open_once(claim=claim)
    artifact = source.artifact
    role_input = Gemma3L3L4ProgressiveARolePrompts(
        corpus_id=GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_PANEL_ID,
        profile="pilot",
        role=GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE,
        prompts=opened.prompts,
        family_ids=opened.family_ids,
        source_file_sha256=opened.source_file_sha256,
    )
    view = Gemma3L3L4ProgressiveARolePreclaimView(
        role=GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE,
        manifest_sha256=artifact.manifest_sha256,
        role_input_file_sha256=artifact.role_input_file_sha256,
        example_count=16,
        family_ids=ITERATIVE_OCCUPANCY_SELECTION_FAMILIES,
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
        panel.membership_receipt_sha256
        != artifact.membership_receipt_sha256
    ):
        raise RuntimeError("materialized occupancy membership drifted")
    return panel
