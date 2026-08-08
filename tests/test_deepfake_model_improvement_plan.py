import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_deepfake_model_improvement_plan.py"
PLAN = ROOT / "configs" / "deepfake" / "model_improvement_plan.json"
SPEC = importlib.util.spec_from_file_location("validate_deepfake_model_improvement_plan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DeepfakeModelImprovementPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = MODULE.load_plan(PLAN)

    def test_repository_plan_is_valid(self):
        self.assertEqual(MODULE.validate_plan(self.plan), [])

    def test_official_test_cannot_select_threshold(self):
        changed = copy.deepcopy(self.plan)
        changed["protocol"]["official_test_used_for_threshold_selection"] = True
        changed["common_evaluation"]["threshold_selected_on"] = "official_test"
        errors = MODULE.validate_plan(changed)
        self.assertTrue(any("official_test_used_for_threshold_selection" in item for item in errors))
        self.assertTrue(any("validation에서만" in item for item in errors))

    def test_test_errors_cannot_become_hard_negatives(self):
        changed = copy.deepcopy(self.plan)
        changed["protocol"]["hard_negative_source"] = "official_test_false_positives"
        changed["protocol"]["test_errors_recycled_into_training"] = True
        errors = MODULE.validate_plan(changed)
        self.assertTrue(any("hard negative" in item for item in errors))
        self.assertTrue(any("test_errors_recycled_into_training" in item for item in errors))

    def test_stage_dependency_must_appear_earlier(self):
        changed = copy.deepcopy(self.plan)
        changed["stages"][1]["depends_on"] = ["p4_ftcn"]
        errors = MODULE.validate_plan(changed)
        self.assertTrue(any("앞 단계에 없습니다" in item for item in errors))

    def test_private_model_artifact_cannot_be_committable(self):
        changed = copy.deepcopy(self.plan)
        changed["artifacts"]["committable"].append("onnx_model")
        errors = MODULE.validate_plan(changed)
        self.assertTrue(any("비공개 산출물" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
