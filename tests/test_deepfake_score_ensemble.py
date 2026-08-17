import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "optimize_deepfake_score_ensemble.py"
CONFIG_PATH = ROOT / "configs" / "deepfake" / "jpeg_conditional_ensemble.json"
SPEC = importlib.util.spec_from_file_location("deepfake_score_ensemble_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ScoreRecord = MODULE.ScoreRecord


def records(model: str):
    rows = []
    conditions = CONFIG["conditions"]
    for condition in conditions:
        if condition == "jpeg_q30" and model == "efficientnet_b4":
            real = [0.10, 0.20, 0.65, 0.70]
            fake = [0.40, 0.60, 0.75, 0.80]
        elif condition == "jpeg_q30":
            real = [0.05, 0.10, 0.15, 0.20]
            fake = [0.75, 0.80, 0.85, 0.90]
        else:
            real = [0.05, 0.10, 0.15, 0.20]
            fake = [0.75, 0.80, 0.85, 0.90]
        for label, scores in ((0, real), (1, fake)):
            for index, score in enumerate(scores):
                rows.append(
                    ScoreRecord(
                        split="validation",
                        video_id=f"private-{label}-{index}",
                        label=label,
                        frame_index=0,
                        score=score,
                        latency_ms=2.0 if model == "efficientnet_b4" else 3.0,
                        condition=condition,
                    )
                )
    return rows


class DeepfakeScoreEnsembleTests(unittest.TestCase):
    def test_selects_condition_aware_policy_when_jpeg_improves(self):
        report = MODULE.build_ensemble_report(
            records("efficientnet_b4"),
            records("xception"),
            CONFIG,
        )
        self.assertTrue(report["ensemble_selected"])
        self.assertTrue(report["selected_policy"].startswith("conditional_"))
        self.assertFalse(report["official_test_used_for_selection"])
        self.assertFalse(report["official_test_inference_performed"])
        serialized = json.dumps(report)
        self.assertNotIn("private-0-0", serialized)
        self.assertFalse(report["privacy"]["contains_frame_scores"])

    def test_condition_aware_route_leaves_other_conditions_unchanged(self):
        aligned = MODULE.align_score_records(
            "efficientnet_b4",
            records("efficientnet_b4"),
            "xception",
            records("xception"),
            expected_conditions=CONFIG["conditions"],
        )
        fused = MODULE.fuse_aligned_records(
            aligned,
            primary_weight=0.5,
            specialist_conditions=["jpeg_q30"],
        )
        for (primary, _specialist), output in zip(aligned, fused):
            if primary.condition == "jpeg_q30":
                self.assertEqual(output.latency_ms, 5.0)
            else:
                self.assertEqual(output.score, primary.score)
                self.assertEqual(output.latency_ms, primary.latency_ms)

    def test_rejects_official_test_scores(self):
        unsafe = records("xception")
        unsafe[0] = MODULE.replace(unsafe[0], split="test")
        with self.assertRaisesRegex(MODULE.EnsembleError, "official test is locked"):
            MODULE.build_ensemble_report(
                records("efficientnet_b4"),
                unsafe,
                CONFIG,
            )

    def test_rejects_mismatched_frame_keys(self):
        specialist = records("xception")[:-1]
        with self.assertRaisesRegex(MODULE.EnsembleError, "frame keys do not match"):
            MODULE.build_ensemble_report(
                records("efficientnet_b4"),
                specialist,
                CONFIG,
            )

    def test_rejects_label_mismatch(self):
        specialist = records("xception")
        specialist[0] = MODULE.replace(specialist[0], label=1)
        with self.assertRaisesRegex(MODULE.EnsembleError, "labels do not match"):
            MODULE.build_ensemble_report(
                records("efficientnet_b4"),
                specialist,
                CONFIG,
            )

    def test_keeps_primary_when_specialist_has_no_gain(self):
        report = MODULE.build_ensemble_report(
            records("xception"),
            records("xception"),
            CONFIG,
        )
        self.assertEqual(report["selected_policy"], "primary_only")
        self.assertFalse(report["ensemble_selected"])


if __name__ == "__main__":
    unittest.main()
