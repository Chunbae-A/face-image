import ast
import base64
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_jpeg_conditional_ensemble_kaggle_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "celebdf_jpeg_conditional_ensemble_kaggle.ipynb"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class JpegConditionalEnsembleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module("jpeg_ensemble_builder_test", BUILDER)
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_source = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_committed_notebook_matches_builder(self):
        self.assertEqual(self.notebook, self.builder.build_notebook())

    def test_notebook_is_private_parseable_and_has_no_outputs(self):
        self.assertTrue(self.notebook["metadata"]["kaggle"]["is_private"])
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

    def test_embeds_current_ensemble_code(self):
        expected = {
            "configs/deepfake/jpeg_conditional_ensemble.json",
            "scripts/celebdf_deepfake.py",
            "scripts/run_celebdf_deepfake.py",
            "scripts/optimize_deepfake_score_ensemble.py",
        }
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
        self.assertEqual(set(embedded), expected)
        for relative_path in expected:
            self.assertEqual(
                base64.b64decode(embedded[relative_path]),
                (ROOT / relative_path).read_bytes(),
            )

    def test_reuses_private_models_without_training_or_official_test(self):
        self.assertIn("effb4_xception_private_models.zip", self.code_source)
        self.assertIn(
            'PREPROCESS_NOTEBOOK_HANDLE = "hywznn/deepsogak-celebdf-preprocess"',
            self.code_source,
        )
        self.assertIn(
            'PRIVATE_MODELS_NOTEBOOK_HANDLE = "hywznn/deepsogak-effb4-xception-compare"',
            self.code_source,
        )
        self.assertIn("kagglehub.notebook_output_download", self.code_source)
        self.assertIn('"--validation-only"', self.code_source)
        self.assertNotIn('"scripts/run_celebdf_deepfake.py", "train"', self.code_source)
        self.assertIn('"official_test": "locked"', self.code_source)

    def test_private_artifacts_stay_in_temp_and_are_deleted(self):
        self.assertIn('WORK_ROOT = Path("/kaggle/temp/jpeg_conditional_ensemble")', self.code_source)
        self.assertIn("shutil.rmtree(WORK_ROOT)", self.code_source)
        self.assertIn("assert not PRIVATE_SCORE_ROOT.exists()", self.code_source)
        package_cell = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
            and 'with zipfile.ZipFile(RESULT_ZIP, "w"' in "".join(cell["source"])
        )
        self.assertIn('not any(name.endswith((".csv", ".jpg", ".mp4", ".pt", ".onnx", ".tar"))', package_cell)


if __name__ == "__main__":
    unittest.main()
