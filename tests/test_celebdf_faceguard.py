import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


celebdf = load_module("celebdf_faceguard", ROOT / "scripts" / "celebdf_faceguard.py")
runner = load_module("run_celebdf_arcface", ROOT / "scripts" / "run_celebdf_arcface.py")


class InventoryTests(unittest.TestCase):
    def test_inventory_and_safe_extract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("Celeb-real/id0_0000.mp4", b"video-zero")
                archive.writestr("prefix/Celeb-real/id1_0002.mp4", b"video-one")
                archive.writestr("Celeb-synthesis/id0_id1_0000.mp4", b"fake")
            rows = celebdf.inventory_zip(archive_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row.subject_id for row in rows], ["id0", "id1"])
            output = root / "out"
            summary = celebdf.extract_rows(archive_path, rows, output)
            self.assertEqual(summary["extracted"], 2)
            self.assertEqual(
                (output / "Celeb-real" / "id0_0000.mp4").read_bytes(),
                b"video-zero",
            )
            self.assertEqual(celebdf.extract_rows(archive_path, rows, output)["skipped"], 2)

    def test_unsafe_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                celebdf._safe_target(Path(directory), "../escape.mp4")


class FrameSamplingTests(unittest.TestCase):
    def test_even_indices_are_unique_and_inside_video(self):
        indices = runner.sample_frame_indices(100, 10)
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertGreaterEqual(indices[0], 0)
        self.assertLess(indices[-1], 100)

    def test_short_video_uses_every_frame(self):
        self.assertEqual(runner.sample_frame_indices(3, 10), [0, 1, 2])


def synthetic_records(subject_count: int = 8, videos_per_subject: int = 8):
    records = []
    for subject_index in range(subject_count):
        embedding = np.zeros(subject_count, dtype=np.float32)
        embedding[subject_index] = 1.0
        for video_index in range(videos_per_subject):
            video_id = f"id{subject_index}_{video_index:04d}"
            records.append(
                celebdf.VideoEmbedding(
                    subject_id=f"id{subject_index}",
                    video_id=video_id,
                    relative_path=f"Celeb-real/{video_id}.mp4",
                    embedding=embedding.copy(),
                    sampled_frames=10,
                    valid_frames=10,
                    mean_detection_score=0.99,
                    mean_face_area_ratio=0.2,
                    decode_seconds=0.1,
                    inference_seconds=0.2,
                )
            )
    return records


class EmbeddingEvaluationTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        records = synthetic_records(subject_count=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.npz"
            celebdf.save_video_embeddings(records, path)
            loaded = celebdf.load_video_embeddings(path)
        self.assertEqual(len(loaded), len(records))
        self.assertEqual(loaded[0].video_id, records[0].video_id)
        self.assertTrue(np.allclose(loaded[0].embedding, records[0].embedding))

    def test_reference_protocol_has_video_disjoint_query_counts(self):
        grouped = celebdf.group_eligible_records(synthetic_records())
        validation, test = celebdf.split_subjects(grouped, seed=7)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 6)
        pairs = celebdf.build_pair_scores(grouped, test, reference_count=5)
        # Six test identities, three post-registration query videos each.
        self.assertEqual(int(pairs.labels.sum()), 18)
        self.assertEqual(int((pairs.labels == 0).sum()), 90)

    def test_perfect_synthetic_embeddings_score_perfectly(self):
        report = celebdf.evaluate_embeddings(
            synthetic_records(),
            seed=7,
            bootstrap_repeats=20,
        )
        self.assertEqual(report["eligible_subject_count"], 8)
        for protocol in report["protocols"].values():
            self.assertAlmostEqual(protocol["test_roc_auc"], 1.0)
            self.assertAlmostEqual(protocol["test_eer"], 0.0)
            self.assertEqual(protocol["operating_points"]["far_0.001"]["test"]["tar"], 1.0)
            self.assertEqual(protocol["operating_points"]["far_0.001"]["test"]["far"], 0.0)


if __name__ == "__main__":
    unittest.main()
