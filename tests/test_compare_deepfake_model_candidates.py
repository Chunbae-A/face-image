import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_deepfake_model_candidates.py"
SPEC = importlib.util.spec_from_file_location("compare_deepfake_candidates_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(candidate_id: str, *, fpr: float, robust_auc: float, latency: float):
    condition = {
        "video": {"roc_auc": robust_auc, "fpr": fpr, "recall": 0.95},
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
        "seed": 20260808,
        "input_size": 256,
        "normalization": "half",
        "train_frames_per_video": 16,
        "target_fpr": 0.01,
        "selected_frames_per_video": 16,
        "aggregation_candidates": ["mean"],
        "checkpoint_sha256": f"{candidate_id}-checkpoint",
        "validation_operating_point_at_recall_0_95": {
            "fpr": fpr,
            "recall": 0.95,
        },
        "validation_video": {"roc_auc": 0.99, "fpr": 0.01},
        "validation_video_latency": {"p50_ms": latency / 2, "p95_ms": latency},
        "condition_validation": {
            "clean": copy.deepcopy(condition),
            "low_light_gamma2": copy.deepcopy(condition),
        },
    }


class CandidateComparisonTests(unittest.TestCase):
    def test_selects_lowest_validation_fpr_without_official_test(self):
        report = MODULE.build_comparison(
            [
                ("efficientnet_b4", candidate("efficientnet_b4", fpr=0.02, robust_auc=0.9, latency=10)),
                ("xception", candidate("xception", fpr=0.01, robust_auc=0.8, latency=20)),
            ]
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
            ]
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
                ]
            )

    def test_rejects_candidate_that_ran_official_test(self):
        xception = candidate("xception", fpr=0.01, robust_auc=0.9, latency=10)
        xception["official_test_inference_performed"] = True
        with self.assertRaisesRegex(MODULE.ComparisonError, "official_test_inference_performed"):
            MODULE.build_comparison(
                [
                    ("efficientnet_b4", candidate("efficientnet_b4", fpr=0.01, robust_auc=0.9, latency=10)),
                    ("xception", xception),
                ]
            )


if __name__ == "__main__":
    unittest.main()
