from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate_module = _load(
    "evaluate_kface_verification", ROOT / "scripts" / "evaluate_kface_verification.py"
)
plot_module = _load(
    "plot_kface_verification", ROOT / "scripts" / "plot_kface_verification.py"
)


def _write_resolution(root: Path, *, noise: float, subjects: int = 8) -> None:
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    accepted = 0
    for subject_index in range(subjects):
        rng = np.random.default_rng(1000 + subject_index)
        center = np.zeros(512, dtype=np.float32)
        center[subject_index] = 1.0
        rows = []
        for _ in range(8):
            row = center + rng.normal(0, noise, size=512).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row)
        np.savez_compressed(
            embeddings_dir / f"subject_{subject_index:016x}.npz",
            embeddings=np.stack(rows),
            quality=np.ones((8, 6), dtype=np.float32),
            selected_indices=np.arange(8, dtype=np.int32),
        )
        accepted += len(rows)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "subject_count": subjects,
                "selected_images": accepted,
                "accepted_images": accepted,
                "rejected_images": 0,
            }
        ),
        encoding="utf-8",
    )


class EvaluateKFaceVerificationTests(unittest.TestCase):
    def test_subject_disjoint_splits_and_private_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "low"
            medium = root / "medium"
            _write_resolution(low, noise=0.005)
            _write_resolution(medium, noise=0.003)

            result = evaluate_module.evaluate(
                low,
                medium,
                references=(3, 5),
                seed=20260815,
                target_far=0.001,
            )

            self.assertEqual(result["paired_subjects"], 8)
            self.assertEqual(result["validation_subjects"], 4)
            self.assertEqual(result["test_subjects"], 4)
            self.assertEqual(
                set(result["protocols"]), {"references_3", "references_5"}
            )
            for protocol in result["protocols"].values():
                self.assertEqual(protocol["enrollment_resolution"], "medium")
                self.assertEqual(protocol["research_gate"]["target_minimum_tar"], 0.90)
                self.assertEqual(protocol["research_gate"]["target_maximum_far"], 0.001)
                for condition in protocol["conditions"].values():
                    self.assertLessEqual(condition["validation"]["far"], 0.001)
                    self.assertGreater(condition["test"]["roc_auc"], 0.99)
                    self.assertGreater(condition["test"]["tar"], 0.99)
            encoded = json.dumps(result)
            self.assertNotIn("subject_000", encoded)
            self.assertFalse(result["contains_subject_identifiers"])
            self.assertFalse(result["contains_embeddings"])

            image_path = root / "verification.png"
            plot_module.plot(result, image_path)
            self.assertTrue(image_path.is_file())
            self.assertGreater(image_path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
