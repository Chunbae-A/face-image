import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

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
runner = load_module("run_celebdf_arcface", ROOT / "scripts" / "run_celebdf_arcface.py")
audit = load_module(
    "audit_celebdf_robustness",
    ROOT / "scripts" / "audit_celebdf_robustness.py",
)


def synthetic_records(condition: str, videos_per_subject: int = 9):
    records = []
    for subject_index in range(8):
        base = np.zeros(9, dtype=np.float32)
        base[subject_index] = 1.0
        for video_index in range(videos_per_subject):
            embedding = base.copy()
            if condition != "clean":
                embedding[-1] = 0.04 * (video_index + 1)
                embedding = celebdf.l2_normalize(embedding)
            video_id = f"id{subject_index}_{video_index:04d}"
            records.append(
                celebdf.VideoEmbedding(
                    subject_id=f"id{subject_index}",
                    video_id=video_id,
                    relative_path=f"Celeb-real/{video_id}.mp4",
                    embedding=embedding,
                    sampled_frames=5,
                    valid_frames=5,
                    mean_detection_score=0.95,
                    mean_face_area_ratio=0.2,
                    decode_seconds=0.02,
                    inference_seconds=0.04,
                    transform_seconds=0.01 if condition != "clean" else 0.0,
                )
            )
    return records


class InputConditionTests(unittest.TestCase):
    def test_conditions_are_deterministic_and_preserve_frame_contract(self):
        checker = np.where(np.indices((32, 32)).sum(axis=0) % 2, 224, 32).astype(
            np.uint8
        )
        frame = np.stack([checker, np.roll(checker, 1, axis=0), checker], axis=-1)
        clean = runner.apply_input_condition(frame, "clean")
        self.assertTrue(np.array_equal(clean, frame))
        for condition in runner.INPUT_CONDITIONS[1:]:
            first = runner.apply_input_condition(frame, condition)
            second = runner.apply_input_condition(frame, condition)
            self.assertEqual(first.shape, frame.shape)
            self.assertEqual(first.dtype, np.uint8)
            self.assertTrue(np.array_equal(first, second), condition)
            self.assertFalse(np.array_equal(first, frame), condition)

    def test_unknown_condition_is_rejected(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            runner.apply_input_condition(frame, "unknown")

    def test_running_condition_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            row = celebdf.ArchiveVideo(
                archive_member="Celeb-real/id0_0000.mp4",
                relative_path="Celeb-real/id0_0000.mp4",
                subject_id="id0",
                video_id="id0_0000",
                uncompressed_bytes=1,
                crc32=1,
            )
            celebdf.write_manifest([row], manifest)
            output = root / "embeddings.npz"
            celebdf.save_video_embeddings(
                synthetic_records("jpeg_q30", videos_per_subject=1)[:1],
                output,
            )
            run_report = root / "run.json"
            run_report.write_text(
                json.dumps({"status": "running", "input_condition": "jpeg_q30"}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                accept_noncommercial_model_license=True,
                manifest=manifest,
                mode="full",
                smoke_subjects=2,
                smoke_videos_per_subject=1,
                output=output,
                run_report=run_report,
                input_condition="jpeg_q30",
                model_name="buffalo_l",
                model_root=root / "models",
                det_size=640,
                video_root=root / "videos",
                rejects=root / "rejects.csv",
                frames_per_video=5,
                minimum_valid_frames=3,
                fail_fast=False,
                checkpoint_every=25,
                progress_every=25,
            )
            inventory = {
                "insightface_version": "test",
                "onnxruntime_version": "test",
                "onnxruntime_available_providers": ["CPUExecutionProvider"],
                "onnxruntime_selected_providers": ["CPUExecutionProvider"],
                "device": "cpu",
                "model_name": "buffalo_l",
                "model_root": str(root / "models"),
                "model_hashes": {"model.onnx": "hash"},
            }
            with mock.patch.object(
                runner,
                "initialize_face_app",
                return_value=(object(), inventory),
            ):
                report = runner.run_pipeline(args)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["input_condition"], "jpeg_q30")
            self.assertEqual(report["successful_video_count_total"], 1)


class RobustnessAuditTests(unittest.TestCase):
    def test_degraded_protocol_keeps_clean_registration_embeddings(self):
        clean = synthetic_records("clean")
        degraded = synthetic_records("low_light_gamma2")
        mixed, _ = audit.build_common_protocol_records(
            {"clean": clean, "low_light_gamma2": degraded},
            seed=7,
        )
        grouped = audit._ordered_protocol_groups(
            mixed["low_light_gamma2"],
            minimum_videos=8,
            minimum_valid_frames=3,
        )
        clean_by_video = {record.video_id: record for record in clean}
        degraded_by_video = {record.video_id: record for record in degraded}
        for records in grouped.values():
            for record in records[:5]:
                self.assertTrue(
                    np.array_equal(record.embedding, clean_by_video[record.video_id].embedding)
                )
            for record in records[5:]:
                self.assertTrue(
                    np.array_equal(
                        record.embedding,
                        degraded_by_video[record.video_id].embedding,
                    )
                )

    def test_audit_uses_common_pool_and_writes_sanitized_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embeddings = {}
            reports = {}
            rejects = {}
            for condition in ("clean", "low_light_gamma2"):
                condition_root = root / condition
                condition_root.mkdir()
                embeddings[condition] = condition_root / "embeddings.npz"
                reports[condition] = condition_root / "run.json"
                rejects[condition] = condition_root / "rejects.csv"
                celebdf.save_video_embeddings(
                    synthetic_records(condition),
                    embeddings[condition],
                )
                reports[condition].write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "selected_video_count": 72,
                            "successful_video_count_total": 72,
                            "rejected_this_run": 1,
                            "frames_per_video": 5,
                            "minimum_valid_frames": 3,
                            "input_condition": condition,
                            "elapsed_seconds": 1.0,
                            "manifest_sha256": "manifest-hash",
                            "model_name": "synthetic",
                            "model_hashes": {"model.onnx": "same-hash"},
                            "device": "cpu",
                        }
                    ),
                    encoding="utf-8",
                )
                with rejects[condition].open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["video_id", "reason"],
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"video_id": f"private_{condition}", "reason": "test_reject"}
                    )

            degraded = celebdf.load_video_embeddings(embeddings["low_light_gamma2"])
            celebdf.save_video_embeddings(degraded[:-1], embeddings["low_light_gamma2"])
            reports["low_light_gamma2"].write_text(
                reports["low_light_gamma2"].read_text(encoding="utf-8").replace(
                    '"successful_video_count_total": 72',
                    '"successful_video_count_total": 71',
                ),
                encoding="utf-8",
            )

            output = root / "sanitized"
            report = audit.run_audit(
                embeddings=embeddings,
                run_reports=reports,
                rejects=rejects,
                output_dir=output,
                seeds=(7, 8),
                bootstrap_repeats=10,
            )

            self.assertEqual(report["conditions"], ["clean", "low_light_gamma2"])
            self.assertEqual(len(report["metrics"]), 4)
            self.assertEqual(len(report["leakage_checks"]), 2)
            self.assertTrue(
                all(
                    item["validation_test_identity_overlap"] == 0
                    and item["registration_query_video_overlap"] == 0
                    for item in report["leakage_checks"]
                )
            )
            self.assertIn("single_global_threshold_approved", report["decisions"])
            serialized = (output / "celebdf_robustness_audit.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("private_low_light_gamma2", serialized)
            self.assertNotIn('"id0"', serialized)
            self.assertTrue((output / "celebdf_robustness_metrics.csv").exists())
            self.assertTrue((output / "celebdf_robustness_summary.csv").exists())
            self.assertNotIn(
                b"\r\n",
                (output / "celebdf_robustness_metrics.csv").read_bytes(),
            )

    def test_duplicate_condition_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.mapping_from_specs(
                [("clean", Path("a")), ("clean", Path("b"))]
            )

    def test_run_report_with_wrong_frame_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "selected_video_count": 590,
                        "successful_video_count_total": 590,
                        "frames_per_video": 10,
                        "minimum_valid_frames": 3,
                        "input_condition": "clean",
                        "manifest_sha256": "manifest",
                        "model_hashes": {"model.onnx": "hash"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                audit.sanitized_run_report(path, "clean")

    def test_bootstrap_repeats_must_be_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            audit.positive_int("0")


if __name__ == "__main__":
    unittest.main()
