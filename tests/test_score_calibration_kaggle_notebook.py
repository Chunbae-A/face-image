import ast
import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_score_calibration_kaggle_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "celebdf_score_calibration_kaggle.ipynb"


def load_module():
    specification = importlib.util.spec_from_file_location(
        "score_calibration_kaggle_builder_test", BUILDER
    )
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ScoreCalibrationKaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module()
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_committed_notebook_matches_builder(self):
        self.assertEqual(cls_notebook := self.notebook, self.builder.build_notebook())
        self.assertTrue(cls_notebook["metadata"]["kaggle"]["is_private"])
        self.assertEqual(cls_notebook["metadata"]["kaggle"]["accelerator"], "gpu")

    def test_code_cells_parse_and_have_no_execution_artifacts(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            python_source = "\n".join(
                line
                for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            if python_source.strip():
                ast.parse(python_source)

    def test_notebook_embeds_current_evaluation_and_calibration_scripts(self):
        bootstrap = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
            and "EMBEDDED_FILES_B64 =" in "".join(cell["source"])
        )
        mapping_text = bootstrap.split("EMBEDDED_FILES_B64 = ", 1)[1].split(
            "\nEMBEDDED_CODE_SHA256", 1
        )[0]
        embedded = ast.literal_eval(mapping_text)
        expected = {
            "scripts/celebdf_deepfake.py",
            "scripts/run_celebdf_deepfake.py",
            "scripts/calibrate_deepfake_scores.py",
        }
        self.assertEqual(set(embedded), expected)
        for relative_path in expected:
            self.assertEqual(
                base64.b64decode(embedded[relative_path]),
                (ROOT / relative_path).read_bytes(),
            )

    def test_private_scores_never_enter_kaggle_working_or_result_zip(self):
        source = "\n".join(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('WORK_ROOT = Path("/kaggle/temp/celebdf_score_calibration")', source)
        self.assertIn("PRIVATE_SCORES.unlink", source)
        self.assertIn('"frame_scores_private.csv" not in names', source)
        self.assertNotIn("archive.write(PRIVATE_SCORES", source)

    def test_notebook_reuses_existing_outputs_without_retraining(self):
        source = "\n".join(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("celebdf_deepfake_preprocess_private.tar", source)
        self.assertIn("celebdf_deepfake_private_model.zip", source)
        self.assertIn('"evaluate"', source)
        self.assertNotIn('"train"', source)
        self.assertIn('"--frame-counts", "16"', source)
        self.assertIn('"--conditions", "clean"', source)


if __name__ == "__main__":
    unittest.main()
