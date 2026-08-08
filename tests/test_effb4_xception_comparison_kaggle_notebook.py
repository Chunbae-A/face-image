import ast
import base64
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_effb4_xception_comparison_kaggle_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "celebdf_effb4_xception_compare_kaggle.ipynb"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Effb4XceptionComparisonNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module("effb4_xception_builder_test", BUILDER)
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

    def test_embeds_current_config_and_comparison_scripts(self):
        expected = {
            "configs/deepfake/effb4_xception_comparison.json",
            "scripts/celebdf_deepfake.py",
            "scripts/run_celebdf_deepfake.py",
            "scripts/compare_deepfake_model_candidates.py",
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

    def test_pins_timm_without_replacing_kaggle_torch(self):
        self.assertIn('"timm==1.0.28"', self.code_source)
        install_cell = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code" and "timm==1.0.28" in "".join(cell["source"])
        )
        self.assertNotIn('"torch==', install_cell)
        self.assertNotIn('"torchvision==', install_cell)

    def test_both_models_use_same_controlled_settings(self):
        config = json.loads(
            (ROOT / "configs" / "deepfake" / "effb4_xception_comparison.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [candidate["id"] for candidate in config["candidates"]],
            ["efficientnet_b4", "xception"],
        )
        self.assertEqual(config["input_size"], 256)
        self.assertEqual(config["normalization"], "half")
        self.assertEqual(config["evaluation_frame_counts"], [16])
        self.assertEqual(config["aggregation_methods"], ["mean"])
        self.assertIn('"--architecture", candidate_id', self.code_source)
        self.assertIn('"--normalization", CONFIG["normalization"]', self.code_source)

    def test_official_test_occurs_only_after_validation_freeze(self):
        validation_position = self.code_source.index('"--validation-only"')
        comparison_position = self.code_source.index("FROZEN_CANDIDATE =")
        official_position = self.code_source.index("# 10. 고정된 승자 하나만 공식 Test")
        self.assertLess(validation_position, comparison_position)
        self.assertLess(comparison_position, official_position)
        self.assertIn(
            'comparison["official_test_inference_performed_before_freeze"] is False',
            self.code_source,
        )
        self.assertIn(
            'CANDIDATE_PATHS[FROZEN_CANDIDATE]["checkpoint"]',
            self.code_source,
        )

    def test_sanitized_bundle_excludes_private_artifacts(self):
        package_cell = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code" and "SANITIZED_BUNDLE" in "".join(cell["source"])
        )
        self.assertIn("CROP_MANIFEST.name not in names", package_cell)
        self.assertIn('not any(name.endswith((".jpg", ".mp4", ".pt", ".onnx", ".csv"))', package_cell)
        self.assertNotIn("PRIVATE_SCORE_ROOT.rglob", package_cell)


if __name__ == "__main__":
    unittest.main()
