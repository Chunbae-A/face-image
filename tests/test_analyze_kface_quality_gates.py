from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "analyze_kface_quality_gates",
    SCRIPTS / "analyze_kface_quality_gates.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AnalyzeKFaceQualityGatesTests(unittest.TestCase):
    def test_face_pixel_side_and_brightness_mask(self) -> None:
        rule = MODULE.QualityGateRule(
            "test",
            minimum_detection_score=0.70,
            minimum_face_pixel_side=40.0,
            minimum_brightness=30.0,
            maximum_brightness=200.0,
        )
        quality = np.asarray(
            [
                [0.80, 0.10, 100.0, 120.0, 200.0, 100.0],
                [0.65, 0.10, 100.0, 120.0, 200.0, 100.0],
                [0.80, 0.02, 100.0, 120.0, 200.0, 100.0],
                [0.80, 0.10, 100.0, 10.0, 200.0, 100.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            rule.mask(quality), np.asarray([True, False, False, False])
        )

    def test_repeated_quality_gate_analysis_has_no_sensitive_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(20260817)
            for subject_index in range(8):
                subject = f"subject_{subject_index + 1:016x}"
                base = np.zeros(512, dtype=np.float32)
                base[subject_index] = 1.0
                medium = []
                low = []
                for _ in range(10):
                    medium_row = base + rng.normal(0, 0.004, size=512)
                    low_row = base + rng.normal(0, 0.008, size=512)
                    medium.append(medium_row / np.linalg.norm(medium_row))
                    low.append(low_row / np.linalg.norm(low_row))
                quality = np.tile(
                    np.asarray(
                        [[0.95, 0.10, 100.0, 128.0, 640.0, 640.0]],
                        dtype=np.float32,
                    ),
                    (10, 1),
                )
                np.savez_compressed(
                    root / f"{subject}__chunk_00000.npz",
                    image_indices=np.arange(10, dtype=np.int32),
                    low_embeddings=np.asarray(low, dtype=np.float32),
                    medium_embeddings=np.asarray(medium, dtype=np.float32),
                    low_quality=quality,
                    medium_quality=quality,
                )

            rules = (
                MODULE.QualityGateRule("baseline"),
                MODULE.QualityGateRule(
                    "strict",
                    minimum_detection_score=0.80,
                    minimum_face_pixel_side=40.0,
                    minimum_brightness=30.0,
                ),
            )
            result = MODULE.analyze_quality_gates(
                root,
                rules=rules,
                reference_count=2,
                seeds=(20260817, 20260818),
                target_far=0.02,
                calibration_far=0.01,
                bins=1_000,
                device="cpu",
                diagnostic_threshold=0.5,
            )

            self.assertEqual(result["input_subjects"], 8)
            self.assertEqual(result["eligible_subjects"], 8)
            self.assertEqual(set(result["aggregates"]), {"baseline", "strict"})
            self.assertIn("face_pixel_side", result["quality_diagnostics"]["low"])
            self.assertFalse(result["contains_subject_identifiers"])
            self.assertFalse(result["contains_embeddings"])
            self.assertFalse(result["individual_scores_persisted"])
            for seed in result["runs"].values():
                for rule in seed.values():
                    self.assertGreater(
                        rule["conditions"]["low"]["test"]["query_coverage"], 0
                    )


if __name__ == "__main__":
    unittest.main()
