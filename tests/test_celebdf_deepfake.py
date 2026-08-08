import importlib.util
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
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


deepfake = load_module(
    "celebdf_deepfake_test_module",
    ROOT / "scripts" / "celebdf_deepfake.py",
)
sys.modules["celebdf_deepfake"] = deepfake
runner = load_module(
    "run_celebdf_deepfake_test_module",
    ROOT / "scripts" / "run_celebdf_deepfake.py",
)


class InventoryTests(unittest.TestCase):
    def test_official_labels_are_converted_and_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("Celeb-real/id0_0000.mp4", b"real-test")
                archive.writestr("Celeb-real/id0_0001.mp4", b"real-train")
                archive.writestr("Celeb-real/id1_0000.mp4", b"real-train-two")
                archive.writestr("YouTube-real/00000.mp4", b"youtube-test")
                archive.writestr("YouTube-real/00001.mp4", b"youtube-train")
                archive.writestr("YouTube-real/00002.mp4", b"youtube-train-two")
                archive.writestr("Celeb-synthesis/id0_id1_0000.mp4", b"fake-test")
                archive.writestr("Celeb-synthesis/id0_id1_0001.mp4", b"fake-train")
                archive.writestr("Celeb-synthesis/id1_id0_0000.mp4", b"fake-train-two")
                archive.writestr(
                    "List_of_testing_videos.txt",
                    "1 Celeb-real/id0_0000.mp4\n"
                    "1 YouTube-real/00000.mp4\n"
                    "0 Celeb-synthesis/id0_id1_0000.mp4\n",
                )

            rows, _ = deepfake.inventory_zip(
                archive_path,
                require_expected_counts=False,
            )
            by_path = {row.relative_path: row for row in rows}
            self.assertEqual(by_path["Celeb-real/id0_0000.mp4"].label, 0)
            self.assertEqual(by_path["Celeb-synthesis/id0_id1_0000.mp4"].label, 1)
            self.assertTrue(by_path["Celeb-synthesis/id0_id1_0000.mp4"].official_test)
            self.assertEqual(by_path["Celeb-synthesis/id0_id1_0000.mp4"].split, "test")

            assigned = deepfake.assign_train_validation_split(
                rows,
                validation_fraction=0.5,
                seed=7,
            )
            test_paths = {row.relative_path for row in assigned if row.split == "test"}
            self.assertEqual(
                test_paths,
                {
                    "Celeb-real/id0_0000.mp4",
                    "YouTube-real/00000.mp4",
                    "Celeb-synthesis/id0_id1_0000.mp4",
                },
            )
            audit = deepfake.leakage_audit(assigned)
            self.assertEqual(audit["train_validation_group_overlap"], 0)
            self.assertEqual(audit["official_test_outside_test_split"], 0)

    def test_official_label_path_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("Celeb-real/id0_0000.mp4", b"real")
                archive.writestr(
                    "List_of_testing_videos.txt",
                    "0 Celeb-real/id0_0000.mp4\n",
                )
            with self.assertRaisesRegex(ValueError, "label/path mismatch"):
                deepfake.inventory_zip(archive_path, require_expected_counts=False)

    def test_auto_extracted_directory_matches_zip_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "dataset.zip"
            members = {
                "Celeb-real/id0_0000.mp4": b"real",
                "YouTube-real/00000.mp4": b"youtube",
                "Celeb-synthesis/id0_id1_0000.mp4": b"fake",
                "List_of_testing_videos.txt": (
                    b"1 YouTube-real/00000.mp4\n"
                    b"0 Celeb-synthesis/id0_id1_0000.mp4\n"
                ),
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, payload in members.items():
                    archive.writestr(name, payload)
            zip_rows, zip_test = deepfake.inventory_zip(
                archive_path,
                require_expected_counts=False,
            )
            extracted = root / "extracted"
            for name, payload in members.items():
                target = extracted / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            directory_rows, directory_test = deepfake.inventory_directory(
                extracted,
                require_expected_counts=False,
            )
            self.assertEqual(zip_test, directory_test)
            self.assertEqual(
                [(row.relative_path, row.label, row.official_test) for row in zip_rows],
                [(row.relative_path, row.label, row.official_test) for row in directory_rows],
            )

    def test_extract_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                deepfake._safe_target(Path(directory), "../outside.mp4")


def make_score_records():
    rows = []
    # Deliberately make mean aggregation perfect while top-k creates a false
    # positive for the second real video.
    values = {
        "validation": {
            "real-a": [0.05, 0.10, 0.15, 0.10],
            "real-b": [0.05, 0.10, 0.15, 0.80],
            "fake-a": [0.80, 0.90, 0.85, 0.90],
            "fake-b": [0.70, 0.80, 0.75, 0.80],
        },
        "test": {
            "real-c": [0.10, 0.10, 0.15, 0.15],
            "real-d": [0.05, 0.10, 0.15, 0.10],
            "fake-c": [0.75, 0.80, 0.85, 0.80],
            "fake-d": [0.70, 0.75, 0.80, 0.85],
        },
    }
    for split, videos in values.items():
        for video_id, scores in videos.items():
            label = int(video_id.startswith("fake"))
            for frame_index, score in enumerate(scores):
                rows.append(
                    deepfake.ScoreRecord(
                        split=split,
                        video_id=video_id,
                        label=label,
                        frame_index=frame_index,
                        score=score,
                        latency_ms=2.0,
                    )
                )
    return rows


class MetricTests(unittest.TestCase):
    def test_perfect_scores_have_perfect_auc_and_ap(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = deepfake.threshold_at_fpr(labels, scores, 0.01)
        metrics = deepfake.classification_metrics(
            labels,
            scores,
            threshold=threshold,
        )
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["fpr"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        precision, recall = deepfake.precision_recall_curve(labels, scores)
        self.assertEqual(float(recall[0]), 0.0)
        self.assertEqual(float(recall[-1]), 1.0)
        self.assertTrue(np.all(np.isfinite(precision)))

    def test_video_aggregation_and_validation_only_selection(self):
        report = deepfake.evaluate_score_records(make_score_records())
        self.assertFalse(report["official_test_used_for_selection"])
        self.assertEqual(report["selection_split"], "validation")
        self.assertEqual(report["selected_aggregation"], "mean")
        self.assertAlmostEqual(report["test_video"]["roc_auc"], 1.0)
        self.assertAlmostEqual(report["test_video"]["fpr"], 0.0)
        self.assertTrue(report["research_gate"]["overall_pass"])
        self.assertIn("test_video_curves", report)

    def test_aggregation_rejects_inconsistent_video_label(self):
        rows = [
            deepfake.ScoreRecord("test", "same-video", 0, 0, 0.1),
            deepfake.ScoreRecord("test", "same-video", 1, 1, 0.9),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent labels"):
            deepfake.aggregate_video_scores(rows, method="mean")

    def test_operating_point_at_recall_uses_lowest_available_fpr(self):
        labels = np.asarray([0, 0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.7, 0.9])
        half_recall = deepfake.operating_point_at_recall(labels, scores, 0.5)
        full_recall = deepfake.operating_point_at_recall(labels, scores, 1.0)
        self.assertAlmostEqual(half_recall["recall"], 0.5)
        self.assertAlmostEqual(half_recall["fpr"], 0.0)
        self.assertAlmostEqual(full_recall["recall"], 1.0)
        self.assertAlmostEqual(full_recall["fpr"], 1 / 3)


class RunnerUtilityTests(unittest.TestCase):
    def test_frame_indices_are_unique_and_avoid_video_edges(self):
        indices = runner.sample_frame_indices(100, 8)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertGreater(indices[0], 0)
        self.assertLess(indices[-1], 99)

    def test_frame_subset_keeps_evenly_spaced_rows_per_video(self):
        rows = [
            runner.CropRecord(
                split="validation",
                video_id="video-a",
                label=1,
                frame_index=index,
                relative_crop_path=f"a/{index}.jpg",
                detection_score=0.9,
                face_area_ratio=0.2,
            )
            for index in range(32)
        ]
        selected = runner.select_frame_subset(rows, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0].frame_index, 0)
        self.assertEqual(selected[-1].frame_index, 31)

    def test_evaluation_degradations_preserve_image_size(self):
        from PIL import Image

        image = Image.new("RGB", (224, 224), (128, 96, 64))
        for condition in runner.EVALUATION_CONDITIONS:
            transformed = runner.apply_evaluation_condition(image, condition)
            self.assertEqual(transformed.size, image.size)
            self.assertEqual(transformed.mode, "RGB")

    def test_onnx_export_uses_legacy_exporter_without_onnxscript(self):
        source = (ROOT / "scripts" / "run_celebdf_deepfake.py").read_text(
            encoding="utf-8"
        )
        export_call = source.split("def export_onnx", 1)[1].split(
            "def smoke_onnx", 1
        )[0]
        self.assertIn("torch.onnx.export(", export_call)
        self.assertIn("dynamo=False", export_call)

    def test_fair_comparison_normalization_is_shared(self):
        expected = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.assertEqual(
            runner.normalization_spec("efficientnet_b4", "half"),
            expected,
        )
        self.assertEqual(runner.normalization_spec("xception", "half"), expected)

    def test_xception_uses_pinned_timm_model_interface(self):
        calls = []
        fake_model = types.SimpleNamespace(
            pretrained_cfg={
                "url": "https://weights.invalid/xception.pth",
                "license": "apache-2.0",
            }
        )

        def create_model(*args, **kwargs):
            calls.append((args, kwargs))
            return fake_model

        fake_timm = types.SimpleNamespace(__version__="test", create_model=create_model)
        with mock.patch.dict(sys.modules, {"timm": fake_timm}):
            model, inventory = runner.build_model(
                architecture="xception",
                pretrained=True,
            )
        self.assertIs(model, fake_model)
        self.assertEqual(calls[0][0], ("legacy_xception.tf_in1k",))
        self.assertEqual(calls[0][1]["num_classes"], 1)
        self.assertTrue(calls[0][1]["exportable"])
        self.assertEqual(inventory["architecture_id"], "xception")

    def test_parser_accepts_validation_only_xception_comparison(self):
        parser = runner.build_parser()
        train_args = parser.parse_args(
            [
                "train",
                "--crop-manifest", "manifest.csv",
                "--crop-root", "crops",
                "--checkpoint", "model.pt",
                "--train-report", "train.json",
                "--architecture", "xception",
                "--normalization", "half",
                "--input-size", "256",
            ]
        )
        evaluate_args = parser.parse_args(
            [
                "evaluate",
                "--crop-manifest", "manifest.csv",
                "--crop-root", "crops",
                "--checkpoint", "model.pt",
                "--private-scores", "scores.csv",
                "--metrics", "metrics.json",
                "--validation-only",
                "--frame-counts", "16",
                "--aggregation-methods", "mean",
            ]
        )
        self.assertEqual(train_args.architecture, "xception")
        self.assertEqual(train_args.normalization, "half")
        self.assertTrue(evaluate_args.validation_only)
        self.assertEqual(evaluate_args.aggregation_methods, ["mean"])

    def test_onnx_smoke_requires_export_metadata_instead_of_cli_preprocessing(self):
        parser = runner.build_parser()
        smoke_args = parser.parse_args(
            [
                "smoke-onnx",
                "--model", "model.onnx",
                "--crop-manifest", "manifest.csv",
                "--crop-root", "crops",
                "--report", "smoke.json",
                "--export-report", "export.json",
            ]
        )
        self.assertEqual(smoke_args.export_report, Path("export.json"))
        self.assertFalse(hasattr(smoke_args, "input_size"))
        self.assertFalse(hasattr(smoke_args, "architecture"))
        self.assertFalse(hasattr(smoke_args, "normalization"))

        source = (ROOT / "scripts" / "run_celebdf_deepfake.py").read_text(
            encoding="utf-8"
        )
        smoke_call = source.split("def smoke_onnx", 1)[1].split(
            "def build_parser", 1
        )[0]
        self.assertIn("_sha256(args.model)", smoke_call)
        self.assertIn('export_report.get("onnx_sha256")', smoke_call)
        self.assertIn('export_report.get("architecture_id"', smoke_call)
        self.assertIn('export_report.get("normalization"', smoke_call)


if __name__ == "__main__":
    unittest.main()
