from __future__ import annotations

import ast
import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_kface_lowres_adapter_kaggle_notebook.py"
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "kaggle" / "kface_lowres_embedding_adapter" / "notebook.ipynb"
)
METADATA_PATH = NOTEBOOK_PATH.with_name("kernel-metadata.json")


def _load_builder():
    specification = importlib.util.spec_from_file_location(
        "build_kface_lowres_adapter_kaggle_notebook", BUILDER_PATH
    )
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class KFaceLowResolutionAdapterKaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load_builder()
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def test_generated_files_match_builder_and_stay_private(self) -> None:
        self.assertEqual(self.notebook, self.builder.build_notebook())
        self.assertEqual(self.metadata, self.builder.build_kernel_metadata())
        self.assertEqual(self.metadata["id"], "hywznn/k-face-lowres-embedding-adapter")
        self.assertTrue(self.metadata["is_private"])
        self.assertTrue(self.metadata["enable_gpu"])
        self.assertFalse(self.metadata["enable_internet"])

    def test_code_cells_parse_and_have_no_execution_artifacts(self) -> None:
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            ast.parse("".join(cell["source"]))

    def test_embeds_current_training_sources(self) -> None:
        source = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if "EMBEDDED_FILES_B64" in "".join(cell.get("source", []))
        )
        assignment = next(
            item
            for item in ast.parse(source).body
            if isinstance(item, ast.Assign)
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == "EMBEDDED_FILES_B64"
        )
        embedded = ast.literal_eval(assignment.value)
        for name, path in (
            (
                "evaluate_kface_full_embeddings.py",
                ROOT / "scripts" / "evaluate_kface_full_embeddings.py",
            ),
            (
                "train_kface_lowres_adapter.py",
                ROOT / "scripts" / "train_kface_lowres_adapter.py",
            ),
        ):
            self.assertEqual(base64.b64decode(embedded[name]), path.read_bytes())

    def test_locked_test_and_private_output_guards_exist(self) -> None:
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in self.notebook["cells"]
        )
        self.assertIn(
            'result["split"]["test_used_for_training_or_candidate_selection"] is False',
            source,
        )
        self.assertIn('result["split"]["locked_test_evaluations"] == 1', source)
        self.assertIn('result["contains_embeddings"] is False', source)
        self.assertIn('Path("/kaggle/temp/deepsogak_kface_adapter/scripts")', source)
        self.assertIn(
            'Path("/kaggle/working/kface_lowres_embedding_adapter.json")', source
        )
        self.assertNotIn("np.save", source)


if __name__ == "__main__":
    unittest.main()
