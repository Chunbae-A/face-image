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
    "analyze_kface_enrollment_strategies",
    SCRIPTS / "analyze_kface_enrollment_strategies.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AnalyzeKFaceEnrollmentStrategiesTests(unittest.TestCase):
    def test_quality_weights_prefer_better_enrollment_image(self) -> None:
        quality = np.asarray(
            [
                [0.90, 0.10, 100.0, 127.5, 640.0, 640.0],
                [0.60, 0.01, 100.0, 10.0, 640.0, 640.0],
            ],
            dtype=np.float32,
        )
        weights = MODULE.enrollment_quality_weights(quality)

        self.assertGreater(weights[0], weights[1])
        self.assertTrue(np.all(weights > 0))

    def test_dual_prototypes_preserve_two_embedding_modes(self) -> None:
        rows = np.zeros((5, 512), dtype=np.float32)
        rows[:3, 0] = 1.0
        rows[3:, 1] = 1.0
        quality = np.tile(
            np.asarray([[0.9, 0.1, 100.0, 128.0, 640.0, 640.0]], dtype=np.float32),
            (5, 1),
        )
        strategy = MODULE.EnrollmentStrategy("dual", 2, False, "test")

        templates = MODULE.build_templates(rows, quality, strategy)

        self.assertEqual(templates.shape, (2, 512))
        self.assertGreater(np.max(templates @ rows[0]), 0.99)
        self.assertGreater(np.max(templates @ rows[-1]), 0.99)

    def test_full_repeated_benchmark_has_no_sensitive_outputs(self) -> None:
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

            strategies = (
                MODULE.EnrollmentStrategy("mean", 1, False, "mean"),
                MODULE.EnrollmentStrategy("weighted", 1, True, "weighted"),
                MODULE.EnrollmentStrategy("dual", 2, False, "dual"),
            )
            result = MODULE.analyze_enrollment_strategies(
                root,
                strategies=strategies,
                reference_count=5,
                seeds=(20260817, 20260818),
                calibration_fars=(0.02, 0.01),
                target_far=0.02,
                bins=1_000,
                device="cpu",
            )

            self.assertEqual(result["eligible_subjects"], 8)
            self.assertEqual(result["query_coverage"], 1.0)
            self.assertEqual(len(result["aggregates"]), 6)
            self.assertEqual(
                result["split_protocol"]["reference_query_image_overlap"], 0
            )
            self.assertTrue(
                result["split_protocol"][
                    "benchmark_candidate_ranking_uses_repeated_test_metrics"
                ]
            )
            self.assertFalse(result["contains_subject_identifiers"])
            self.assertFalse(result["contains_embeddings"])
            self.assertFalse(result["individual_scores_persisted"])
            for aggregate in result["aggregates"].values():
                self.assertEqual(aggregate["query_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
