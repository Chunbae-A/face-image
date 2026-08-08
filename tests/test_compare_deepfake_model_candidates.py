import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_deepfake_model_candidates.py"
CONFIG_PATH = ROOT / "configs" / "deepfake" / "effb4_xception_comparison.json"
SPEC = importlib.util.spec_from_file_location("compare_deepfake_candidates_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CONFIG = MODULE._load_json(CONFIG_PATH)


def candidate(
    candidate_id: str,
    *,
    fpr: float,
    robust_auc: float,
    latency: float,
    recall: float = 0.95,
):
    condition = {
        "video": {"roc_auc": robust_auc, "fpr": fpr, "recall": recall},
        "latency": {"p50_ms": latency / 2, "p95_ms": latency},
    }
    return {
        "evaluation_scope": "validation_only",
        "selection_split": "validation",
        "official_test_used_for_selection": False,
        "official_test_inference_performed": False,
        "architecture_id": candidate_id,
        "model": candidate_id,
        "crop_manifest_sha256": "same-manifest",
        "seed": CONFIG["seed"],
        "input_size": CONFIG["input_size"],
        "normalization": CONFIG["normalization"],
        "train_frames_per_video": CONFIG["train_frames_per_video"],
        "target_fpr": CONFIG["target_validation_fpr"],
        "selected_frames_per_video": CONFIG["evaluation_frame_counts"][0],
        "aggregation_candidates": CONFIG["aggregation_methods"],
        "training_protocol": {
            "optimizer": CONFIG["optimizer"],
            "learning_rate": CONFIG["learning_rate"],
            "weight_decay": CONFIG["weight_decay"],
            "epochs": CONFIG["epochs"],
            "early_stopping_patience": CONFIG["early_stopping_patience"],
            "batch_size": CONFIG["batch_size"],
            "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
            "augmentation": CONFIG["augmentation"],
            "dataloader_seed": CONFIG["seed"],
            "sampler_seed": CONFIG["seed"],
        },
        "checkpoint_sha256": f"{candidate_id}-checkpoint",
        "validation_operating_point_at_recall_0_95": {
            "fpr": fpr,
            "recall": recall,
        },
        "validation_video": {"roc_auc": 0.99, "fpr": 0.01},
        "validation_video_latency": {"p50_ms": latency / 2, "p95_ms": latency},
        "condition_validation": {
            name: copy.deepcopy(condition) for name in CONFIG["conditions"]
        },
    }


class CandidateComparisonTests(unittest.TestCase):
    def test_selects_lowest_validation_fpr_without_official_test(self):
        report = MODULE.build_comparison(
            [
                ("efficientnet_b4", candidate("efficientnet_b4", fpr=0.02, robust_auc=0.9, latency=10)),
                ("xception", candidate("xception", fpr=0.01, robust_auc=0.8, latency=20)),
            ],
            CONFIG,
        )
        self.assertEqual(report["selected_candidate"], "xception")
        self.assertFalse(report["official_test_used_for_selection"])
        self.assertFalse(report["official_test_inference_performed_before_freeze"])
        self.assertEqual(len(report["selection_fingerprint_sha256"]), 64)

    def test_robustness_auc_breaks_equal_fpr_tie(self):
        report = MODULE.build_comparison(
            [
                ("efficientnet_b4", candidate("efficientnet_b4", fpr=0.01, robust_auc=0.85, latency=10)),
                ("xception", candidate("xception", fpr=0.01, robust_auc=0.92, latency=20)),
            ],
            CONFIG,
        )
        self.assertEqual(report["selected_candidate"], "xception")

    def test_rejects_different_manifest(self):
        xception = candidate("xception", fpr=0.01, robust_auc=0.9, latency=10)
        xception["crop_manifest_sha256"] = "different"
        with self.assertRaisesRegex(MODULE.ComparisonError, "crop_manifest_sha256"):
            MODULE.build_comparison(
                [
                    ("efficientnet_b4", candidate("efficientnet_b4", fpr=0.01, robust_auc=0.9, latency=10)),
                    ("xception", xception),
                ],
                CONFIG,
            )

    def test_rejects_candidate_that_ran_official_test(self):
        xception = candidate("xception", fpr=0.01, robust_auc=0.9, latency=10)
        xception["official_test_inference_performed"] = True
        with self.assertRaisesRegex(MODULE.ComparisonError, "official_test_inference_performed"):
            MODULE.build_comparison(
                [
                    ("efficientnet_b4", candidate("efficientnet_b4", fpr=0.01, robust_auc=0.9, latency=10)),
                    ("xception", xception),
                ],
                CONFIG,
            )

    def test_candidate_below_target_recall_is_not_selected(self):
        report = MODULE.build_comparison(
            [
                (
                    "efficientnet_b4",
                    candidate("efficientnet_b4", fpr=0.02, robust_auc=0.9, latency=10),
                ),
                (
                    "xception",
                    candidate(
                        "xception",
                        fpr=0.0,
                        robust_auc=0.99,
                        latency=1,
                        recall=0.94,
                    ),
                ),
            ],
            CONFIG,
        )
        self.assertEqual(report["selected_candidate"], "efficientnet_b4")
        by_id = {row["candidate_id"]: row for row in report["candidates"]}
        self.assertFalse(by_id["xception"]["eligible_at_target_recall"])

    def test_missing_fairness_value_is_rejected(self):
        xception = candidate("xception", fpr=0.01, robust_auc=0.9, latency=10)
        del xception["training_protocol"]
        with self.assertRaisesRegex(MODULE.ComparisonError, "training_protocol"):
            MODULE.build_comparison(
                [
                    ("efficientnet_b4", candidate("efficientnet_b4", fpr=0.01, robust_auc=0.9, latency=10)),
                    ("xception", xception),
                ],
                CONFIG,
            )

    def test_fails_when_no_candidate_reaches_target_recall(self):
        with self.assertRaisesRegex(MODULE.ComparisonError, "target recall"):
            MODULE.build_comparison(
                [
                    (
                        "efficientnet_b4",
                        candidate(
                            "efficientnet_b4",
                            fpr=0.0,
                            robust_auc=0.9,
                            latency=10,
                            recall=0.94,
                        ),
                    ),
                    (
                        "xception",
                        candidate(
                            "xception",
                            fpr=0.0,
                            robust_auc=0.9,
                            latency=10,
                            recall=0.94,
                        ),
                    ),
                ],
                CONFIG,
            )


if __name__ == "__main__":
    unittest.main()
