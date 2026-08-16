from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_kface_kaggle_dataset",
    ROOT / "scripts" / "prepare_kface_kaggle_dataset.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareKFaceKaggleDatasetTests(unittest.TestCase):
    def test_hardlinks_flattened_private_chunks_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "upload"
            for subject in ("subject_0000000000000001", "subject_0000000000000002"):
                chunks = source / "subjects" / subject / "chunks"
                chunks.mkdir(parents=True)
                for index in range(2):
                    (chunks / f"chunk_{index:05d}.npz").write_bytes(
                        f"{subject}-{index}".encode()
                    )

            result = MODULE.prepare(
                source,
                destination,
                dataset_id="hywznn/private-test",
                expected_subjects=2,
                expected_chunks=4,
            )

            self.assertEqual(result["subject_count"], 2)
            self.assertEqual(result["chunk_count"], 4)
            linked = destination / "subject_0000000000000001__chunk_00000.npz"
            original = (
                source
                / "subjects"
                / "subject_0000000000000001"
                / "chunks"
                / "chunk_00000.npz"
            )
            self.assertEqual(linked.stat().st_ino, original.stat().st_ino)
            metadata = json.loads(
                (destination / "dataset-metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["id"], "hywznn/private-test")
            self.assertNotIn("public", metadata)

            batched = root / "batched"
            batched_result = MODULE.prepare(
                source,
                batched,
                dataset_id="hywznn/private-test-batched",
                expected_subjects=2,
                expected_chunks=4,
                subjects_per_batch=1,
            )
            self.assertEqual(batched_result["upload_layout"], "batched_directories")
            self.assertTrue(
                (
                    batched
                    / "subjects_001_001"
                    / "subject_0000000000000001__chunk_00000.npz"
                ).is_file()
            )
            self.assertTrue(
                (
                    batched
                    / "subjects_002_002"
                    / "subject_0000000000000002__chunk_00001.npz"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
