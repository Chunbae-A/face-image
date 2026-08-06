import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


celebdf = load_module("celebdf_faceguard", ROOT / "scripts" / "celebdf_faceguard.py")
audit = load_module("audit_celebdf_baseline", ROOT / "scripts" / "audit_celebdf_baseline.py")


def synthetic_records(frames: int):
    records = []
    for subject_index in range(8):
        embedding = np.zeros(8, dtype=np.float32)
        embedding[subject_index] = 1.0
        for video_index in range(8):
            video_id = f"id{subject_index}_{video_index:04d}"
            records.append(
                celebdf.VideoEmbedding(
                    subject_id=f"id{subject_index}",
                    video_id=video_id,
                    relative_path=f"Celeb-real/{video_id}.mp4",
                    embedding=embedding.copy(),
                    sampled_frames=frames,
                    valid_frames=frames,
                    mean_detection_score=0.99,
                    mean_face_area_ratio=0.2,
                    decode_seconds=0.01 * frames,
                    inference_seconds=0.02 * frames,
                )
            )
    return records


class BaselineAuditTests(unittest.TestCase):
    def test_audit_is_sanitized_and_covers_all_combinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embeddings = {}
            reports = {}
            rejects = {}
            for frames in (1, 5, 10):
                run_root = root / f"frames_{frames}"
                run_root.mkdir()
                embeddings[frames] = run_root / "embeddings.npz"
                reports[frames] = run_root / "run.json"
                rejects[frames] = run_root / "rejects.csv"
                celebdf.save_video_embeddings(synthetic_records(frames), embeddings[frames])
                reports[frames].write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "frames_per_video": frames,
                            "minimum_valid_frames": min(3, frames),
                            "selected_video_count": 64,
                            "successful_video_count_total": 64,
                            "rejected_this_run": 1,
                            "elapsed_seconds": float(frames),
                            "manifest_sha256": "manifest-hash",
                            "model_name": "synthetic",
                            "model_hashes": {"model.onnx": "same-hash"},
                            "device": "cpu",
                        }
                    ),
                    encoding="utf-8",
                )
                with rejects[frames].open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["video_id", "reason"])
                    writer.writeheader()
                    writer.writerow({"video_id": f"private_{frames}", "reason": "test_reject"})

            output = root / "sanitized"
            report = audit.run_audit(
                embeddings=embeddings,
                run_reports=reports,
                rejects=rejects,
                output_dir=output,
                seeds=(7, 8),
                reference_counts=(1, 3, 5),
                bootstrap_repeats=10,
            )

            self.assertEqual(len(report["metrics"]), 18)
            self.assertEqual(len(report["leakage_checks"]), 6)
            self.assertTrue(
                all(item["validation_test_identity_overlap"] == 0 for item in report["leakage_checks"])
            )
            self.assertEqual(report["decisions"]["frames"]["recommendation"], "use_5_frames")
            self.assertEqual(
                report["decisions"]["registration"]["recommendation"],
                "use_3_references",
            )
            self.assertEqual(
                report["decisions"]["registration"]["reference_frames_per_video"],
                5,
            )
            self.assertEqual(
                report["input_runs"][0]["reject_reason_counts"],
                {"test_reject": 1},
            )
            self.assertEqual(report["input_runs"][0]["quality"]["success_rate"], 1.0)
            serialized = (output / "celebdf_baseline_audit.json").read_text(encoding="utf-8")
            self.assertNotIn("private_1", serialized)
            self.assertNotIn('"id0"', serialized)
            self.assertTrue((output / "celebdf_baseline_audit_metrics.csv").exists())
            self.assertTrue((output / "celebdf_baseline_audit_summary.csv").exists())

    def test_duplicate_frame_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.mapping_from_specs([(1, Path("a")), (1, Path("b"))])

    def test_duplicate_video_id_is_rejected_by_leakage_check(self):
        records = synthetic_records(5)
        records.append(records[0])
        with self.assertRaisesRegex(ValueError, "duplicate video_id"):
            audit.leakage_summary(
                records,
                seed=7,
                minimum_valid_frames=3,
                max_reference_count=5,
            )

    def test_run_report_frame_mismatch_and_paths_are_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "run.json"
            report_path.write_text(
                json.dumps(
                    {
                        "frames_per_video": 5,
                        "selected_video_count": 64,
                        "video_root": "/content/drive/MyDrive/private-data",
                        "output": "/content/private-embeddings.npz",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frame mismatch"):
                audit.sanitized_run_report(report_path, 10)
            sanitized = audit.sanitized_run_report(report_path, 5)
            self.assertNotIn("video_root", sanitized)
            self.assertNotIn("output", sanitized)
            self.assertEqual(sanitized["selected_video_count"], 64)

    def test_bootstrap_repeats_must_be_positive(self):
        self.assertEqual(audit.positive_int("1"), 1)
        with self.assertRaises(argparse.ArgumentTypeError):
            audit.positive_int("0")


if __name__ == "__main__":
    unittest.main()
