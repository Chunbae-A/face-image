import ast
import base64
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "celebdf_arcface_robustness_colab.ipynb"
BUILDER = ROOT / "scripts" / "build_celebdf_robustness_colab_notebook.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RobustnessNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module("build_celebdf_robustness_colab_notebook", BUILDER)
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_committed_notebook_matches_builder(self):
        self.assertEqual(self.notebook, self.builder.build_notebook())

    def test_embedded_scripts_match_sources(self):
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
            "scripts/celebdf_faceguard.py",
            "scripts/run_celebdf_arcface.py",
            "scripts/audit_celebdf_robustness.py",
        }
        self.assertEqual(set(embedded), expected)
        for relative_path in expected:
            self.assertEqual(
                base64.b64decode(embedded[relative_path]),
                (ROOT / relative_path).read_bytes(),
            )

    def test_notebook_has_fixed_protocol_and_private_artifact_gate(self):
        source = "\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"]
        )
        self.assertIn("EXPECTED_SOURCE_ZIP_BYTES = 928989923", source)
        self.assertIn("590 * FRAMES_PER_VIDEO * len(CONDITIONS)", source)
        self.assertIn('"--input-condition", condition', source)
        self.assertIn('if any(name.endswith(".npz") for name in names)', source)
        self.assertIn('"raw_data_in_bundle": False', source)
        self.assertIn('"embeddings_in_bundle": False', source)
        self.assertIn('"environment": "colab"', source)
        self.assertIn('"Pillow==12.3.0"', source)
        for condition in (
            "clean",
            "jpeg_q30",
            "gaussian_blur_sigma2",
            "low_light_gamma2",
            "downscale_0_25",
            "combined_mobile_stress",
        ):
            self.assertIn(f'"{condition}"', source)

    def test_python_code_cells_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.lstrip().startswith("#@title 2. 라이브러리 설치"):
                continue
            try:
                compile(source, f"notebook-cell-{index}", "exec")
            except SyntaxError as error:
                self.fail(f"code cell {index} has invalid Python: {error}")


if __name__ == "__main__":
    unittest.main()
