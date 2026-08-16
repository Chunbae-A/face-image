from __future__ import annotations

import ast
import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_kface_full_kaggle_notebook.py"
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "kaggle" / "kface_full_verification" / "notebook.ipynb"
)
METADATA_PATH = NOTEBOOK_PATH.with_name("kernel-metadata.json")


def _load_builder():
    specification = importlib.util.spec_from_file_location(
        "build_kface_full_kaggle_notebook", BUILDER_PATH
    )
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class KFaceFullKaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load_builder()
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def test_generated_files_match_builder_and_stay_private(self) -> None:
        self.assertEqual(self.notebook, self.builder.build_notebook())
        self.assertEqual(self.metadata, self.builder.build_kernel_metadata())
        self.assertTrue(self.metadata["is_private"])
        self.assertTrue(self.metadata["enable_gpu"])
        self.assertFalse(self.metadata["enable_internet"])
        self.assertEqual(
            self.metadata["dataset_sources"],
            ["hywznn/deepsogak-kface-arcface-private-2026-08-17"],
        )

    def test_code_cells_parse_and_have_no_execution_artifacts(self) -> None:
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            ast.parse("".join(cell["source"]))

    def test_notebook_embeds_current_evaluator(self) -> None:
        source = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if "EMBEDDED_EVALUATOR_B64" in "".join(cell.get("source", []))
        )
        encoded = source.split('EMBEDDED_EVALUATOR_B64 = "', 1)[1].split('"\n', 1)[0]
        self.assertEqual(
            base64.b64decode(encoded),
            (ROOT / "scripts" / "evaluate_kface_full_embeddings.py").read_bytes(),
        )

    def test_only_aggregate_outputs_are_persisted(self) -> None:
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in self.notebook["cells"]
        )
        self.assertIn('Path("/kaggle/working/kface_full_verification.json")', source)
        self.assertIn('Path("/kaggle/working/kface_full_verification.png")', source)
        self.assertIn('result["contains_embeddings"] is False', source)
        self.assertIn('Path("/kaggle/temp/kface_private_embeddings")', source)
        self.assertIn('INPUT_DIR.rglob("subject_*__chunk_*.npz")', source)
        self.assertNotIn("np.save", source)


if __name__ == "__main__":
    unittest.main()
