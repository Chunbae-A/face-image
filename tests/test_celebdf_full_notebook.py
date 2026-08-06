import ast
import base64
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "celebdf_arcface_full_colab.ipynb"
BUILDER = ROOT / "scripts" / "build_celebdf_colab_notebook.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FullNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module("build_celebdf_colab_notebook_full_test", BUILDER)
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
        }
        self.assertEqual(set(embedded), expected)
        for relative_path in expected:
            self.assertEqual(
                base64.b64decode(embedded[relative_path]),
                (ROOT / relative_path).read_bytes(),
            )

    def test_code_cells_parse_and_have_no_outputs(self):
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

    def test_download_bundle_excludes_runtime_paths_and_ids(self):
        bundle_cell = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
            and "#@title 13. 비식별 결과 묶음 저장" in "".join(cell["source"])
        )
        self.assertIn("PUBLIC_INVENTORY_JSON", bundle_cell)
        self.assertIn("PUBLIC_RUN_REPORT_JSON", bundle_cell)
        self.assertIn("REJECT_COUNTS_JSON", bundle_cell)
        self.assertNotIn("INVENTORY_JSON, RUN_REPORT_JSON", bundle_cell)
        self.assertNotIn("bundle_files.append(REJECTS_CSV)", bundle_cell)
        for private_field in ("video_root", '"output"', '"rejects"'):
            self.assertNotIn(private_field, bundle_cell)


if __name__ == "__main__":
    unittest.main()
