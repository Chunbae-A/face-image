import ast
import base64
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_celebdf_deepfake_kaggle_notebooks.py"
PREPROCESS_NOTEBOOK = ROOT / "notebooks" / "celebdf_deepfake_preprocess_kaggle.ipynb"
TRAIN_NOTEBOOK = ROOT / "notebooks" / "celebdf_deepfake_train_kaggle.ipynb"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DeepfakeKaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module("deepfake_kaggle_builder_test", BUILDER)
        cls.preprocess = json.loads(PREPROCESS_NOTEBOOK.read_text(encoding="utf-8"))
        cls.train = json.loads(TRAIN_NOTEBOOK.read_text(encoding="utf-8"))

    def test_committed_notebooks_match_builder(self):
        self.assertEqual(self.preprocess, self.builder.build_preprocess_notebook())
        self.assertEqual(self.train, self.builder.build_train_notebook())

    def test_all_code_cells_parse_and_have_no_outputs(self):
        for notebook in (self.preprocess, self.train):
            self.assertTrue(notebook["metadata"]["kaggle"]["is_private"])
            for cell in notebook["cells"]:
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

    def test_both_notebooks_embed_current_scripts(self):
        expected = {
            "scripts/celebdf_deepfake.py",
            "scripts/run_celebdf_deepfake.py",
        }
        for notebook in (self.preprocess, self.train):
            bootstrap = next(
                "".join(cell["source"])
                for cell in notebook["cells"]
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

    def test_preprocess_output_is_explicitly_private(self):
        source = "\n".join(
            "".join(cell["source"])
            for cell in self.preprocess["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("celebdf_deepfake_preprocess_private.tar", source)
        self.assertIn("CROP_MANIFEST.name", source)
        self.assertIn("warning", source)

    def test_train_sanitized_bundle_excludes_private_artifacts(self):
        bundle_cell = next(
            "".join(cell["source"])
            for cell in self.train["cells"]
            if cell["cell_type"] == "code"
            and "# 10. 비식별 결과와" in "".join(cell["source"])
        )
        self.assertIn("PRIVATE_SCORES.name not in names", bundle_cell)
        self.assertIn("CROP_MANIFEST.name not in names", bundle_cell)
        self.assertNotIn("archive.write(PRIVATE_SCORES", bundle_cell)
        self.assertIn("celebdf_deepfake_private_model.zip", bundle_cell)

    def test_train_uses_kaggle_torchvision_compatible_pillow(self):
        source = "\n".join(
            "".join(cell["source"])
            for cell in self.train["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('"Pillow==11.3.0"', source)
        self.assertNotIn('"Pillow==12.3.0"', source)


if __name__ == "__main__":
    unittest.main()
