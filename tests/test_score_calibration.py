import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from faceguard_api.calibration import ScoreCalibration, unavailable_calibration_result

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_deepfake_scores.py"
PUBLIC_RESULT = (
    ROOT
    / "reports"
    / "deepfake_score_calibration"
    / "2026-08-08"
    / "deepfake_video_calibration.json"
)


def load_module():
    specification = importlib.util.spec_from_file_location(
        "score_calibration_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ScoreCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_module()

    def test_calibration_metrics_are_zero_for_perfect_binary_probabilities(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float)
        metrics = self.calibration.calibration_metrics(labels, probabilities)
        self.assertLess(metrics["brier"], 1e-12)
        self.assertLess(metrics["ece"], 1e-6)

    def test_report_selects_only_on_validation_and_contains_no_identifiers(self):
        validation_real = np.linspace(0.02, 0.30, 100)
        validation_fake = np.linspace(0.70, 0.99, 100)
        test_real = np.r_[np.linspace(0.03, 0.28, 49), 0.95]
        test_fake = np.linspace(0.68, 0.98, 50)
        report = self.calibration.build_calibration_report(
            np.r_[np.zeros(100), np.ones(100)],
            np.r_[validation_real, validation_fake],
            np.r_[np.zeros(50), np.ones(50)],
            np.r_[test_real, test_fake],
            model_fingerprint="a" * 64,
            calibration_version="test-v1",
        )

        self.assertIn(report["selected_method"], {"temperature", "platt", "isotonic"})
        self.assertEqual(report["selection_split"], "validation")
        self.assertFalse(report["official_test_used_for_selection"])
        self.assertFalse(report["privacy"]["contains_video_ids"])
        self.assertLessEqual(
            report["risk_bands"]["low_max_raw_score"],
            report["risk_bands"]["high_min_raw_score"],
        )
        serialized = json.dumps(report)
        self.assertNotIn("Celeb-real/", serialized)
        self.assertNotIn("Celeb-synthesis/", serialized)

    def test_unapproved_artifact_withholds_probability_but_keeps_risk_band(self):
        payload = {
            "schema_version": "1.0",
            "calibration_version": "test-v1",
            "scope": "deepfake_video_mean_16_frames",
            "model_fingerprint": "a" * 64,
            "selected_method": "temperature",
            "parameters": {"temperature": 2.0},
            "calibration_status": "research_only_unapproved",
            "display_approved": False,
            "warning": "연구용",
            "risk_bands": {
                "low_max_raw_score": 0.2,
                "high_min_raw_score": 0.8,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = ScoreCalibration.load(
                path,
                expected_model_fingerprint="a" * 64,
                expected_scope="deepfake_video_mean_16_frames",
            )

        assert calibration is not None
        result = calibration.apply(0.9)
        self.assertIsNone(result.calibrated_probability)
        self.assertEqual(result.risk_level, "high")
        self.assertEqual(result.calibration_version, "test-v1")

    def test_approved_artifact_returns_calibrated_probability(self):
        payload = {
            "schema_version": "1.0",
            "calibration_version": "test-v1",
            "scope": "deepfake_video_mean_16_frames",
            "model_fingerprint": "a" * 64,
            "selected_method": "platt",
            "parameters": {"slope": 0.5, "intercept": 0.0},
            "calibration_status": "validated",
            "display_approved": True,
            "warning": "검증 완료",
            "risk_bands": {
                "low_max_raw_score": 0.2,
                "high_min_raw_score": 0.8,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = ScoreCalibration.load(
                path,
                expected_model_fingerprint="a" * 64,
                expected_scope="deepfake_video_mean_16_frames",
            )

        assert calibration is not None
        result = calibration.apply(0.9)
        self.assertIsNotNone(result.calibrated_probability)
        self.assertGreater(result.calibrated_probability, 0.5)

    def test_artifact_model_mismatch_is_rejected(self):
        payload = {
            "schema_version": "1.0",
            "calibration_version": "test-v1",
            "scope": "deepfake_video_mean_16_frames",
            "model_fingerprint": "a" * 64,
            "selected_method": "temperature",
            "parameters": {"temperature": 1.0},
            "calibration_status": "validated",
            "display_approved": True,
            "warning": "검증 완료",
            "risk_bands": {
                "low_max_raw_score": 0.2,
                "high_min_raw_score": 0.8,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                ScoreCalibration.load(
                    path,
                    expected_model_fingerprint="b" * 64,
                    expected_scope="deepfake_video_mean_16_frames",
                )

    def test_missing_artifact_has_explicit_unavailable_status(self):
        result = unavailable_calibration_result()
        self.assertEqual(result.calibration_status, "not_available")
        self.assertIsNone(result.calibrated_probability)
        self.assertIn("확률", result.warning)

    def test_public_kaggle_result_is_private_safe_and_withholds_probability(self):
        payload = json.loads(PUBLIC_RESULT.read_text(encoding="utf-8"))
        calibration = ScoreCalibration.load(
            PUBLIC_RESULT,
            expected_model_fingerprint=payload["model_fingerprint"],
            expected_scope="deepfake_video_mean_16_frames",
        )

        assert calibration is not None
        result = calibration.apply(0.85)
        self.assertFalse(payload["display_approved"])
        self.assertFalse(payload["gate"]["overall_pass"])
        self.assertIsNone(result.calibrated_probability)
        self.assertFalse(payload["privacy"]["contains_video_ids"])
        self.assertFalse(payload["privacy"]["contains_frame_scores"])


if __name__ == "__main__":
    unittest.main()
