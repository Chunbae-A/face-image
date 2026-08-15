from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_kface_resolutions", ROOT / "scripts" / "compare_kface_resolutions.py"
)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def write_results(path: Path, vectors: dict[str, np.ndarray]) -> None:
    embeddings = path / "embeddings"
    embeddings.mkdir(parents=True)
    selected = sum(len(items) for items in vectors.values())
    (path / "summary.json").write_text(
        json.dumps(
            {
                "selected_images": selected,
                "accepted_images": selected,
                "rejected_images": 0,
            }
        ),
        encoding="utf-8",
    )
    for subject, values in vectors.items():
        quality = np.tile(
            np.asarray([[0.9, 0.2, 100.0, 120.0, 224.0, 224.0]], dtype=np.float32),
            (len(values), 1),
        )
        np.savez_compressed(
            embeddings / f"{subject}.npz", embeddings=values, quality=quality
        )


class CompareKFaceResolutionsTests(unittest.TestCase):
    def test_compares_only_aggregate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vector = np.zeros((2, 512), dtype=np.float32)
            vector[:, 0] = 1.0
            write_results(root / "low", {"subject_aaa": vector})
            write_results(root / "medium", {"subject_aaa": vector.copy()})

            result = compare.compare_results(root / "low", root / "medium")

            self.assertEqual(result["paired_subject_count"], 1)
            self.assertAlmostEqual(
                result["cross_resolution_same_subject_similarity"]["mean"], 1.0
            )
            self.assertAlmostEqual(result["low"]["face_acceptance_rate"], 1.0)
            serialized = json.dumps(result)
            self.assertNotIn("subject_aaa", serialized)
            self.assertFalse(result["contains_embeddings"])


if __name__ == "__main__":
    unittest.main()
