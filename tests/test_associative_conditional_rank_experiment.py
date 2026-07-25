import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

import fisher_graph.associative_conditional_rank_experiment as experiment


class AssociativeConditionalRankExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.output = (
            Path(cls.temporary_directory.name) / "conditional-rank.pt"
        )
        cls.report = experiment.run_associative_conditional_rank_experiment(
            basis_examples=32,
            router_examples=32,
            calibration_b_examples=32,
            output=cls.output,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def _resign(
        self,
        raw: dict[str, object],
        *,
        output: Path,
    ) -> None:
        payload = {
            key: value
            for key, value in raw.items()
            if key not in {
                "scientific_payload_sha256",
                "report_sha256",
            }
        }
        scientific_digest = experiment._scientific_payload_sha256(
            payload
        )
        report = experiment._build_report(
            payload,
            output=output,
            scientific_digest=scientific_digest,
        )
        report_digest = experiment._report_sha256(report)
        torch.save(
            {
                **payload,
                "scientific_payload_sha256": scientific_digest,
                "report_sha256": report_digest,
            },
            output,
        )
        output.with_suffix(".json").write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_round_trip_is_weights_only_and_fails_closed_before_validation(
        self,
    ) -> None:
        loaded = experiment.load_associative_conditional_rank_artifact(
            self.output
        )

        self.assertFalse(
            self.report["scientific_status"]["calibration_b_passed"]
        )
        self.assertFalse(
            loaded["scientific_status"]["validation_evaluated"]
        )
        self.assertFalse(loaded["scientific_status"]["test_evaluated"])
        self.assertFalse(
            loaded["scientific_status"][
                "conditional_representation_viable"
            ]
        )
        self.assertFalse(
            loaded["scientific_status"][
                "source_layer_compute_savings_supported"
            ]
        )
        raw = torch.load(
            self.output,
            map_location="cpu",
            weights_only=True,
        )
        self.assertFalse(raw["contains_model_weights"])
        self.assertNotIn("model_state_dict", raw)
        self.assertEqual(
            raw["analysis"]["validation"],
            {
                "evaluated": False,
                "reason": (
                    "calibration_b_gate_failed_validation_not_evaluated"
                ),
            },
        )

    def test_unsigned_tamper_is_rejected_by_digest(self) -> None:
        raw = torch.load(
            self.output,
            map_location="cpu",
            weights_only=True,
        )
        raw["scientific_status"]["compression_claim"] = True
        tampered = Path(self.temporary_directory.name) / "raw-tamper.pt"
        torch.save(raw, tampered)

        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            experiment.load_associative_conditional_rank_artifact(
                tampered
            )

    def test_resigned_router_forgery_is_rejected_by_fit_recomputation(
        self,
    ) -> None:
        raw = copy.deepcopy(
            torch.load(
                self.output,
                map_location="cpu",
                weights_only=True,
            )
        )
        raw["routing_plan"]["router"]["weight"][0, 0] += 0.25
        tampered = (
            Path(self.temporary_directory.name) / "router-forgery.pt"
        )
        self._resign(raw, output=tampered)

        with self.assertRaisesRegex(
            ValueError,
            "routing plan does not match fit evidence",
        ):
            experiment.load_associative_conditional_rank_artifact(
                tampered
            )

    def test_resigned_status_forgery_is_rejected_by_gate_recomputation(
        self,
    ) -> None:
        raw = copy.deepcopy(
            torch.load(
                self.output,
                map_location="cpu",
                weights_only=True,
            )
        )
        raw["scientific_status"]["calibration_b_passed"] = True
        raw["scientific_status"]["validation_evaluated"] = True
        tampered = (
            Path(self.temporary_directory.name) / "status-forgery.pt"
        )
        self._resign(raw, output=tampered)

        with self.assertRaisesRegex(
            ValueError,
            "scientific status is invalid",
        ):
            experiment.load_associative_conditional_rank_artifact(
                tampered
            )


if __name__ == "__main__":
    unittest.main()
