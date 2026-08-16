from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_kface_full_embeddings",
    ROOT / "scripts" / "evaluate_kface_full_embeddings.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvaluateKFaceFullEmbeddingsTests(unittest.TestCase):
    def test_repeated_subject_disjoint_full_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(20260817)
            for subject_index in range(6):
                subject = f"subject_{subject_index + 1:016x}"
                base = np.zeros(512, dtype=np.float32)
                base[subject_index] = 1.0
                medium = []
                low = []
                for _ in range(8):
                    medium_row = base + rng.normal(0, 0.004, size=512)
                    low_row = base + rng.normal(0, 0.008, size=512)
                    medium.append(medium_row / np.linalg.norm(medium_row))
                    low.append(low_row / np.linalg.norm(low_row))
                quality = np.tile(
                    np.asarray([[0.95, 0.1, 100.0, 128.0, 640.0, 640.0]], dtype=np.float32),
                    (8, 1),
                )
                np.savez_compressed(
                    root / f"{subject}__chunk_00000.npz",
                    image_indices=np.arange(8, dtype=np.int32),
                    low_embeddings=np.asarray(low, dtype=np.float32),
                    medium_embeddings=np.asarray(medium, dtype=np.float32),
                    low_quality=quality,
                    medium_quality=quality,
                )

            result = MODULE.evaluate_full(
                root,
                references=(1, 2),
                seeds=(20260817, 20260818),
                target_far=0.02,
                calibration_far=0.01,
                minimum_detection_score=0.60,
                bins=1_000,
                device="cpu",
            )

            self.assertEqual(result["input_subjects"], 6)
            self.assertEqual(result["eligible_subjects"], 6)
            self.assertEqual(result["seeds"], [20260817, 20260818])
            self.assertFalse(result["contains_embeddings"])
            self.assertFalse(result["contains_subject_identifiers"])
            self.assertEqual(result["threshold_status"], "research_only_unapproved")
            for seed in result["runs"].values():
                for protocol in seed.values():
                    self.assertGreater(protocol["conditions"]["low"]["test"]["tar"], 0.9)
                    self.assertLessEqual(
                        protocol["conditions"]["low"]["test"]["far"], 0.02
                    )


if __name__ == "__main__":
    unittest.main()
