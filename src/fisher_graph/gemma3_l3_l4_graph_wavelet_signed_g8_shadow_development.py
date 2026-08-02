"""Source-authoritative A-fit smoke test for the signed-g8 executor.

The preregistered structural confirmation missed only its 7-of-8 uniformity
gate.  This runner therefore uses an already-consumed, explicitly exported
16-prompt Calibration-A fit panel for diagnosis.  It does not open
Calibration-B, validation, or test, and a pass cannot qualify deployment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path

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
from .gemma3_l3_l4_conditional_spectral_shadow_evaluation import (
    Gemma3L3L4ConditionalSpectralShadowExample,
    evaluate_gemma3_l3_l4_conditional_spectral_development_shadow,
)
from .gemma3_l3_l4_conditional_spectral_shadow_runtime import (
    Gemma3L3L4ConditionalSpectralShadowRuntime,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
    _tokenizer_backend_identity,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_candidate import (
    DEFAULT_FROZEN_ARTIFACT_SHA256,
    DEFAULT_FROZEN_REPORT_SHA256,
    DEFAULT_FROZEN_TENSOR_FILE_SHA256,
    DEFAULT_OUTPUT as DEFAULT_CANDIDATE_ARTIFACT,
    _file_sha256,
    _reserve_outputs,
    _stage_json,
    load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PANEL",
    "run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_PANEL = _LOCAL_ROOT / "dev-v9-a-fit-first16-export.json"
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "shadow-a-fit16-dev-v1.json"
)
DEFAULT_MAX_LENGTH = 128
_MINIMUM_MAX_LENGTH = 10
_EXPECTED_PANEL_FILE_SHA256 = (
    "00e1f7bf07c918e3092b7b4cab5bbc2f7d0cac4df05a737061ce7383d8078809"
)
_EXPECTED_RAW_MODEL_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_EXPECTED_FACTORIZED_MODEL_SHA256 = (
    "ead03074b87898c9e6c5b068b738420ab0dcf178f07603e885a71964b94ebb7a"
)
_EXPECTED_FACTORIZED_EXECUTION_SHA256 = (
    "911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9"
)
# The frozen qualification contract used a different post-tokenization call
# shape.  Pin the lazy backend reached by this runner's one-prompt, length-128
# ABI and require it to remain identical for all 16 authenticated A prompts.
_EXPECTED_A_FIT_TOKENIZER_POST_BYTES = 14_386_431
_EXPECTED_A_FIT_TOKENIZER_POST_SHA256 = (
    "3e30cf837beecfdaf19813c7c21fa95e50bdc490dd2079e5dc2993bc63e5933d"
)
_FACTORIZED_SCOPE = "factorized_refit"
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development"
)
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-signed-g8-shadow-development:v1\0"
)
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer_state": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_tensors": False,
    "contains_scalar_metrics": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}
_PANEL_FIELDS = {
    "schema",
    "format_version",
    "source_corpus_id",
    "source_role",
    "scientific_status",
    "selection_rule",
    "calibration_b_exported",
    "guard_exported",
    "validation_exported",
    "test_exported",
    "model_or_tokenizer_accessed",
    "prompts",
    "family_ids",
    "fit_positions",
    "source_prompt_indices",
    "prompt_sha256",
    "source_fit_prompt_index_sha256",
}
_SHA256_HEX = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_REPORT_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _load_panel(
    path: Path | str,
) -> tuple[tuple[Gemma3L3L4ConditionalSpectralShadowExample, ...], dict[str, object]]:
    source = Path(path)
    if _file_sha256(source) != _EXPECTED_PANEL_FILE_SHA256:
        raise ValueError("A-fit shadow panel file differs")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("A-fit shadow panel must be an object")
    prompts = raw.get("prompts")
    families = raw.get("family_ids")
    positions = raw.get("fit_positions")
    source_indices = raw.get("source_prompt_indices")
    prompt_hashes = raw.get("prompt_sha256")
    if (
        set(raw) != _PANEL_FIELDS
        or raw.get("schema")
        != "fisher_graph.local_v9_a_fit_development_export"
        or raw.get("format_version") != 1
        or raw.get("source_corpus_id") != "structured-strong-v9"
        or raw.get("source_role") != "calibration_a_fit_only"
        or raw.get("scientific_status") != "development_only"
        or raw.get("selection_rule")
        != "first_16_authenticated_fit_partition_positions"
        or raw.get("calibration_b_exported") is not False
        or raw.get("guard_exported") is not False
        or raw.get("validation_exported") is not False
        or raw.get("test_exported") is not False
        or raw.get("model_or_tokenizer_accessed") is not False
        or not isinstance(prompts, list)
        or not isinstance(families, list)
        or not isinstance(positions, list)
        or not isinstance(source_indices, list)
        or not isinstance(prompt_hashes, list)
        or not all(
            len(value) == 16
            for value in (
                prompts,
                families,
                positions,
                source_indices,
                prompt_hashes,
            )
        )
        or len(set(prompts)) != 16
        or positions != list(range(16))
        or any(
            type(index) is not int or index < 0
            for index in source_indices
        )
        or len(set(source_indices)) != 16
        or any(
            not isinstance(family, str)
            or not family
            or family != family.strip()
            for family in families
        )
        or len(Counter(families)) != 8
        or set(Counter(families).values()) != {2}
        or not isinstance(raw.get("source_fit_prompt_index_sha256"), str)
        or len(raw["source_fit_prompt_index_sha256"]) != 64
        or any(
            character not in _SHA256_HEX
            for character in raw["source_fit_prompt_index_sha256"]
        )
        or any(
            not isinstance(prompt, str)
            or not prompt
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != digest
            for prompt, digest in zip(prompts, prompt_hashes, strict=True)
        )
    ):
        raise ValueError("A-fit shadow panel protocol differs")
    examples = tuple(
        Gemma3L3L4ConditionalSpectralShadowExample(
            example_id=f"a_fit_{int(position):03d}",
            family_id=str(family),
            prompt=str(prompt),
        )
        for prompt, family, position in zip(
            prompts,
            families,
            positions,
            strict=True,
        )
    )
    return examples, {
        "file_sha256": _EXPECTED_PANEL_FILE_SHA256,
        "schema": raw["schema"],
        "source_corpus_id": raw["source_corpus_id"],
        "source_role": raw["source_role"],
        "source_fit_prompt_index_sha256": raw[
            "source_fit_prompt_index_sha256"
        ],
        "example_count": 16,
        "family_count": 8,
        "examples_per_family": 2,
        "prompt_sha256s": tuple(prompt_hashes),
        "source_prompt_indices": tuple(source_indices),
        "contains_prompt_text": False,
        "calibration_b_exported": False,
        "validation_exported": False,
        "test_exported": False,
    }


def _frozen_tokenizer_integrity_check(
    tokenizer: object,
    contract: Mapping[str, object],
) -> Callable[[str], None]:
    """Return a before/after guard for the tokenizer's lazy backend state."""

    initial = {
        "bytes": contract["backend_serialized_bytes"],
        "sha256": contract["backend_serialized_sha256"],
    }
    post = {
        "bytes": _EXPECTED_A_FIT_TOKENIZER_POST_BYTES,
        "sha256": _EXPECTED_A_FIT_TOKENIZER_POST_SHA256,
    }

    def check(stage: str) -> None:
        actual = _tokenizer_backend_identity(tokenizer)
        if stage == "before":
            if actual not in (initial, post):
                raise ValueError("tokenizer backend drifted before tokenization")
        elif stage == "after":
            if actual != post:
                raise ValueError("tokenizer backend drifted during tokenization")
        else:
            raise ValueError("tokenizer integrity stage is invalid")

    return check


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("shadow output must be JSON under .local-runs")
    return destination


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    reservation = _reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = _json_sha256(report)
        stage = _stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development(
    *,
    candidate_artifact_path: Path | str = DEFAULT_CANDIDATE_ARTIFACT,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    panel_path: Path | str = DEFAULT_PANEL,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, object]:
    """Run the frozen candidate on reusable A-fit prompts, metrics only."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite shadow report")
    if (
        type(max_length) is not int
        or not _MINIMUM_MAX_LENGTH <= max_length <= 256
    ):
        raise ValueError("max_length must lie in [10, 256]")
    examples, panel_receipt = _load_panel(panel_path)
    candidate = load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        candidate_artifact_path,
        expected_artifact_sha256=DEFAULT_FROZEN_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_FROZEN_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_FROZEN_REPORT_SHA256,
    )
    basis = load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    tokenizer, tokenizer_contract = _load_and_validate_frozen_local_tokenizer(
        protocol=protocol
    )
    tokenizer_integrity_check = _frozen_tokenizer_integrity_check(
        tokenizer,
        tokenizer_contract,
    )
    model_metadata = candidate.model
    if (
        model_metadata.get("source_model_sha256")
        != _EXPECTED_RAW_MODEL_SHA256
    ):
        raise ValueError("candidate raw model lineage differs")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    device = resolve_torch_device("cpu")
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("live raw Gemma differs from the frozen candidate")
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
            factorized_model_sha256 != _EXPECTED_FACTORIZED_MODEL_SHA256
            or factorized_execution_sha256
            != _EXPECTED_FACTORIZED_EXECUTION_SHA256
        ):
            raise ValueError("live factorized Gemma differs")
        runtime = (
            Gemma3L3L4ConditionalSpectralShadowRuntime.from_signed_g8_candidate(
                candidate,
                basis,
                expected_basis_payload_sha256=(
                    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
                ),
                expected_live_model_sha256=factorized_model_sha256,
                expected_adapter_execution_sha256=(
                    factorized_execution_sha256
                ),
                analysis_device="cpu",
            )
        )
        evaluation = (
            evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
                runtime=runtime,
                adapter=adapter,
                tokenizer=tokenizer,
                examples=examples,
                max_length=max_length,
                model_input_device=device,
                tokenizer_integrity_check=tokenizer_integrity_check,
            )
        )
    finally:
        switcher.close()
    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise RuntimeError("shadow runner did not restore the raw Gemma model")
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": "reusable_calibration_a_fit_diagnostic_smoke",
        "reason": (
            "run_after_fresh_structural_confirmation_passed_all_but_"
            "the_7_of_8_native_group_uniformity_gate"
        ),
        "candidate": {
            "artifact_sha256": candidate.artifact_sha256,
            "plan_artifact_sha256": candidate.plan.artifact_sha256,
            "method": candidate.method,
            "candidate_serving_authorized": False,
        },
        "panel": panel_receipt,
        "tokenization": {
            "max_length": max_length,
            "tokenization_batch_size": 1,
            "device": "cpu",
            "tokenizer_class": tokenizer_contract["tokenizer_class"],
            "tokenizer_configuration_sha256": tokenizer_contract[
                "configuration_sha256"
            ],
            "tokenizer_initial_backend_sha256": tokenizer_contract[
                "backend_serialized_sha256"
            ],
            "tokenizer_post_backend_sha256": (
                _EXPECTED_A_FIT_TOKENIZER_POST_SHA256
            ),
            "prompt_text_retained": False,
            "token_ids_retained": False,
            "backend_integrity_checked_before_and_after_each_prompt": True,
        },
        "model": {
            "raw_source_model_sha256": _EXPECTED_RAW_MODEL_SHA256,
            "factorized_live_model_sha256": (
                _EXPECTED_FACTORIZED_MODEL_SHA256
            ),
            "factorized_adapter_execution_sha256": (
                _EXPECTED_FACTORIZED_EXECUTION_SHA256
            ),
            "local_files_only": True,
            "device": "cpu",
            "dtype": "float32",
        },
        "evaluation": evaluation,
        "scientific_status": {
            "development_smoke_complete": True,
            "source_outputs_authoritative": True,
            "candidate_outputs_used_for_metrics_only": True,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "formal_qualification": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
        },
        "artifact": {
            "file": str(destination),
            "committable": False,
        },
        "safety": _SAFETY,
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run signed-g8 source-authoritative A-fit shadow smoke",
    )
    parser.add_argument("--candidate-artifact", default=DEFAULT_CANDIDATE_ARTIFACT)
    parser.add_argument("--basis-package", default=DEFAULT_BASIS_PACKAGE)
    parser.add_argument("--base-artifact", default=DEFAULT_FULL_MLP_STACK_ARTIFACT)
    parser.add_argument("--refit-artifact", default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development(
        candidate_artifact_path=arguments.candidate_artifact,
        basis_package_path=arguments.basis_package,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        panel_path=arguments.panel,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        max_length=arguments.max_length,
    )
    evaluation = report["evaluation"]  # type: ignore[assignment]
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "artifact": report["artifact"],
                "scientific_status": report["scientific_status"],
                "behavioral": evaluation["behavioral"],  # type: ignore[index]
                "affected_behavioral": evaluation[
                    "affected_behavioral"
                ],  # type: ignore[index]
                "target_modal": evaluation["target_modal"],  # type: ignore[index]
                "full_width_boundary": evaluation[
                    "full_width_boundary"
                ],  # type: ignore[index]
                "coverage": evaluation["coverage"],  # type: ignore[index]
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
