"""Restore the exact deployable generator catalog from base plus refit state.

The sequential-refit artifact is intentionally an overlay: it stores complete
fits only for layers 10 through 17 and binds layers 0 through 9 to the frozen
full-stack artifact.  This module turns those two analysis artifacts into one
small runtime catalog while preserving their exact lineage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path
import re

from .full_mlp_stack_generators import FullMLPStackGeneratorFit
from .gemma3_full_mlp_stack_artifact import (
    load_gemma3_full_mlp_stack_artifact,
)
from .gemma3_full_mlp_stack_refit_artifact import (
    load_gemma3_full_mlp_stack_refit_artifact,
)
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)


__all__ = [
    "Gemma3RefitRuntimeCatalog",
    "restore_gemma3_full_mlp_stack_refit_runtime",
]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_LAYER_COUNT = 18
_REFIT_START_LAYER = 10


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Gemma3RefitRuntimeCatalog:
    """Authenticated ordered replacement catalog and its frozen provenance."""

    replacements: tuple[Gemma3ModalGeneratorReplacement, ...]
    layer_lineage: tuple[Mapping[str, object], ...]
    generator_plan_sha256s: tuple[str, ...]
    deployed_fit_sha256s: tuple[str, ...]
    source_fit_sha256s: tuple[str, ...]
    source_model_sha256: str
    base_artifact_file_sha256: str
    base_scientific_payload_sha256: str
    refit_artifact_file_sha256: str
    refit_scientific_payload_sha256: str
    model_metadata: Mapping[str, object]
    analysis_split: Mapping[str, object]
    partition_metadata: Mapping[str, object]
    frozen_refit_metrics: Mapping[str, object]
    resource_accounting: Mapping[str, object]
    refit_start_layer: int = _REFIT_START_LAYER

    def __post_init__(self) -> None:
        expected = tuple(range(_EXPECTED_LAYER_COUNT))
        if (
            len(self.replacements) != _EXPECTED_LAYER_COUNT
            or tuple(value.layer_ordinal for value in self.replacements)
            != expected
            or len(self.layer_lineage) != _EXPECTED_LAYER_COUNT
            or tuple(row.get("layer_ordinal") for row in self.layer_lineage)
            != expected
            or len(self.generator_plan_sha256s) != _EXPECTED_LAYER_COUNT
            or len(self.deployed_fit_sha256s) != _EXPECTED_LAYER_COUNT
            or len(self.source_fit_sha256s) != _EXPECTED_LAYER_COUNT
            or self.refit_start_layer != _REFIT_START_LAYER
        ):
            raise ValueError("runtime catalog must cover exact ordered layers")
        for value, label in (
            (self.source_model_sha256, "source model"),
            (self.base_artifact_file_sha256, "base artifact file"),
            (
                self.base_scientific_payload_sha256,
                "base scientific payload",
            ),
            (self.refit_artifact_file_sha256, "refit artifact file"),
            (
                self.refit_scientific_payload_sha256,
                "refit scientific payload",
            ),
        ):
            _require_sha256(value, label=label)
        for label, values in (
            ("generator plan", self.generator_plan_sha256s),
            ("deployed fit", self.deployed_fit_sha256s),
            ("source fit", self.source_fit_sha256s),
        ):
            for value in values:
                _require_sha256(value, label=label)
        for ordinal, (replacement, row) in enumerate(
            zip(self.replacements, self.layer_lineage, strict=True)
        ):
            expected_kind = (
                "frozen_source"
                if ordinal < self.refit_start_layer
                else "sequential_refit"
            )
            if (
                replacement.generator_plan.artifact_sha256
                != self.generator_plan_sha256s[ordinal]
                or row.get("deployment_kind") != expected_kind
                or row.get("deployed_plan_sha256")
                != self.generator_plan_sha256s[ordinal]
                or row.get("deployed_fit_sha256")
                != self.deployed_fit_sha256s[ordinal]
                or row.get("source_fit_sha256")
                != self.source_fit_sha256s[ordinal]
            ):
                raise ValueError("runtime catalog lineage is inconsistent")
        for value, label in (
            (self.model_metadata, "model metadata"),
            (self.analysis_split, "analysis split"),
            (self.partition_metadata, "partition metadata"),
            (self.frozen_refit_metrics, "frozen refit metrics"),
            (self.resource_accounting, "resource accounting"),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{label} must be a mapping")
        content = self.analysis_split.get("content_sha256")
        if (
            self.model_metadata.get("adapter_model_fingerprint")
            != self.source_model_sha256
            or self.analysis_split.get("role")
            != "open_development_assessment"
            or not isinstance(content, tuple)
            or not content
            or len(content) != len(set(content))
            or any(
                not isinstance(value, str)
                or _SHA256_RE.fullmatch(value) is None
                for value in content
            )
            or self.analysis_split.get("example_count") != len(content)
        ):
            raise ValueError("runtime model or analysis split metadata is invalid")
        _require_sha256(
            self.analysis_split.get("serialized_sha256"),
            label="analysis split",
        )
        if (
            self.partition_metadata.get("assessment_prompt_count")
            != len(content)
            or type(
                self.partition_metadata.get("selection_prompt_count")
            )
            is not int
            or type(self.partition_metadata.get("expected_prompt_count"))
            is not int
        ):
            raise ValueError("runtime partition metadata is inconsistent")
        _require_sha256(
            self.partition_metadata.get("artifact_sha256"),
            label="partition artifact",
        )
        for name in (
            "nll_per_token",
            "delta_nll_per_token",
            "native_to_candidate_kl_per_token",
            "top1_agreement_to_native",
        ):
            value = self.frozen_refit_metrics.get(name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"frozen refit metric {name} must be numeric")


def _fit_replacement(
    state: object,
    *,
    expected_ordinal: int,
    restore_fit: Callable[
        [Mapping[str, object]], FullMLPStackGeneratorFit
    ],
) -> tuple[Gemma3ModalGeneratorReplacement, str, str, str]:
    raw = _require_mapping(state, label="serialized generator fit")
    fit = restore_fit(raw)
    fit.validate_integrity()
    if fit.superfragment.layer_ordinal != expected_ordinal:
        raise ValueError("serialized fit describes the wrong layer")
    replacement = Gemma3ModalGeneratorReplacement(
        layer_ordinal=expected_ordinal,
        removed_mode_indices=fit.superfragment.channel_indices,
        generator_plan=fit.executable_plan,
    )
    return (
        replacement,
        fit.artifact_sha256,
        fit.executable_plan.artifact_sha256,
        fit.superfragment.source_model_sha256,
    )


def restore_gemma3_full_mlp_stack_refit_runtime(
    base_artifact_path: Path | str,
    refit_artifact_path: Path | str,
    *,
    load_base: Callable[[Path | str], dict[str, object]] = (
        load_gemma3_full_mlp_stack_artifact
    ),
    load_refit: Callable[[Path | str], dict[str, object]] = (
        load_gemma3_full_mlp_stack_refit_artifact
    ),
    restore_fit: Callable[
        [Mapping[str, object]], FullMLPStackGeneratorFit
    ] = FullMLPStackGeneratorFit.from_state_dict,
    file_sha256: Callable[[Path], str] = _file_sha256,
) -> Gemma3RefitRuntimeCatalog:
    """Strictly combine unchanged source layers and sequential refit layers."""

    base_path = Path(base_artifact_path)
    refit_path = Path(refit_artifact_path)
    base_file_sha256 = _require_sha256(
        file_sha256(base_path),
        label="base artifact file",
    )
    refit_file_sha256 = _require_sha256(
        file_sha256(refit_path),
        label="refit artifact file",
    )
    base = load_base(base_path)
    refit = load_refit(refit_path)
    if not isinstance(base, dict) or not isinstance(refit, dict):
        raise TypeError("strict artifact loaders must return dictionaries")

    base_scientific = _require_sha256(
        base.get("scientific_payload_sha256"),
        label="base scientific payload",
    )
    refit_scientific = _require_sha256(
        refit.get("scientific_payload_sha256"),
        label="refit scientific payload",
    )
    frozen_sources = _require_mapping(
        refit.get("frozen_sources"),
        label="refit frozen sources",
    )
    frozen_full = _require_mapping(
        frozen_sources.get("full_stack"),
        label="refit frozen full stack",
    )
    if (
        frozen_full.get("artifact_file_sha256") != base_file_sha256
        or frozen_full.get("scientific_payload_sha256") != base_scientific
    ):
        raise ValueError("refit artifact does not bind the supplied base file")

    base_model = _require_mapping(base.get("model"), label="base model")
    refit_model = _require_mapping(refit.get("model"), label="refit model")
    source_model_sha256 = _require_sha256(
        base_model.get("adapter_model_fingerprint"),
        label="base source model",
    )
    if (
        refit_model.get("adapter_model_fingerprint") != source_model_sha256
        or refit_model.get("model_id") != base_model.get("model_id")
        or refit_model.get("resolved_commit")
        != base_model.get("resolved_commit")
    ):
        raise ValueError("base and refit model bindings differ")
    base_splits = _require_mapping(base.get("splits"), label="base splits")
    partition_metadata = dict(
        _require_mapping(
            base_splits.get("partition"),
            label="base partition metadata",
        )
    )

    source_layers = tuple(
        _require_mapping(value, label="source layer summary")
        for value in _require_sequence(
            refit.get("source_layer_summaries"),
            label="source layer summaries",
        )
    )
    if (
        len(source_layers) != _EXPECTED_LAYER_COUNT
        or tuple(row.get("layer_ordinal") for row in source_layers)
        != tuple(range(_EXPECTED_LAYER_COUNT))
    ):
        raise ValueError("source summaries do not cover exact ordered layers")
    splits = _require_mapping(refit.get("splits"), label="refit splits")
    analysis_split = dict(
        _require_mapping(
            splits.get("assessment"),
            label="refit assessment split",
        )
    )
    evaluation = _require_mapping(
        refit.get("evaluation"),
        label="refit evaluation",
    )
    conditions = _require_mapping(
        evaluation.get("conditions"),
        label="refit evaluation conditions",
    )
    frozen_refit_metrics = dict(
        _require_mapping(
            conditions.get("sequential_refit_full_stack"),
            label="sequential refit full-stack metrics",
        )
    )
    resource_accounting = dict(
        _require_mapping(
            refit.get("resource_accounting"),
            label="refit resource accounting",
        )
    )

    base_states_raw = base.pop("generator_fits", None)
    refit_states_raw = refit.pop("refit_generator_fits", None)
    base_states = list(
        _require_sequence(base_states_raw, label="base generator fits")
    )
    refit_states = list(
        _require_sequence(refit_states_raw, label="refit generator fits")
    )
    if (
        len(base_states) != _EXPECTED_LAYER_COUNT
        or len(refit_states)
        != _EXPECTED_LAYER_COUNT - _REFIT_START_LAYER
    ):
        raise ValueError("base/refit generator fit counts are invalid")
    refit_rows = tuple(
        _require_mapping(value, label="layer refit row")
        for value in _require_sequence(
            refit.get("layer_refits"),
            label="layer refits",
        )
    )
    if (
        len(refit_rows) != len(refit_states)
        or tuple(row.get("layer_ordinal") for row in refit_rows)
        != tuple(range(_REFIT_START_LAYER, _EXPECTED_LAYER_COUNT))
    ):
        raise ValueError("refit rows do not cover exact ordered suffix")

    replacements: list[Gemma3ModalGeneratorReplacement] = []
    lineage: list[Mapping[str, object]] = []
    plan_hashes: list[str] = []
    deployed_fit_hashes: list[str] = []
    source_fit_hashes: list[str] = []
    for ordinal in range(_EXPECTED_LAYER_COUNT):
        source_row = source_layers[ordinal]
        source_fit_sha256 = _require_sha256(
            source_row.get("source_fit_sha256"),
            label=f"layer {ordinal} source fit",
        )
        if source_row.get("source_model_sha256") != source_model_sha256:
            raise ValueError("source layer binds a different model")
        if ordinal < _REFIT_START_LAYER:
            state = base_states[ordinal]
            deployment_kind = "frozen_source"
            expected_fit_sha256 = source_fit_sha256
            refit_row: Mapping[str, object] | None = None
        else:
            state = refit_states[ordinal - _REFIT_START_LAYER]
            deployment_kind = "sequential_refit"
            refit_row = refit_rows[ordinal - _REFIT_START_LAYER]
            expected_fit_sha256 = _require_sha256(
                refit_row.get("refit_fit_sha256"),
                label=f"layer {ordinal} refit fit",
            )
            if refit_row.get("source_fit_sha256") != source_fit_sha256:
                raise ValueError("refit row source lineage differs")

        replacement, fit_sha256, plan_sha256, fit_model_sha256 = (
            _fit_replacement(
                state,
                expected_ordinal=ordinal,
                restore_fit=restore_fit,
            )
        )
        if (
            fit_sha256 != expected_fit_sha256
            or fit_model_sha256 != source_model_sha256
        ):
            raise ValueError("deployed generator fit lineage differs")
        if ordinal < _REFIT_START_LAYER:
            expected_plan = _require_sha256(
                source_row.get("dense_plan_sha256"),
                label=f"layer {ordinal} source dense plan",
            )
            if plan_sha256 != expected_plan:
                raise ValueError("unchanged source plan hash differs")
        replacements.append(replacement)
        plan_hashes.append(plan_sha256)
        deployed_fit_hashes.append(fit_sha256)
        source_fit_hashes.append(source_fit_sha256)
        lineage.append(
            {
                "layer_ordinal": ordinal,
                "layer_id": source_row.get("layer_id"),
                "deployment_kind": deployment_kind,
                "source_fit_sha256": source_fit_sha256,
                "deployed_fit_sha256": fit_sha256,
                "deployed_plan_sha256": plan_sha256,
            }
        )
        if ordinal < _REFIT_START_LAYER:
            base_states[ordinal] = None
        else:
            refit_states[ordinal - _REFIT_START_LAYER] = None
        del state, replacement
        gc.collect()

    return Gemma3RefitRuntimeCatalog(
        replacements=tuple(replacements),
        layer_lineage=tuple(lineage),
        generator_plan_sha256s=tuple(plan_hashes),
        deployed_fit_sha256s=tuple(deployed_fit_hashes),
        source_fit_sha256s=tuple(source_fit_hashes),
        source_model_sha256=source_model_sha256,
        base_artifact_file_sha256=base_file_sha256,
        base_scientific_payload_sha256=base_scientific,
        refit_artifact_file_sha256=refit_file_sha256,
        refit_scientific_payload_sha256=refit_scientific,
        model_metadata=dict(refit_model),
        analysis_split=analysis_split,
        partition_metadata=partition_metadata,
        frozen_refit_metrics=frozen_refit_metrics,
        resource_accounting=resource_accounting,
    )
