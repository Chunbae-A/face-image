import ast
import base64
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "celebdf_arcface_audit_colab.ipynb"


class AuditNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_notebook_has_no_execution_artifacts_and_code_parses(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            python_source = "\n".join(
                line for line in source.splitlines() if not line.lstrip().startswith("%")
            )
            if python_source.strip():
                ast.parse(python_source)

    def test_protocol_is_fixed_to_issue_four(self):
        source = "\n".join("".join(cell["source"]) for cell in self.notebook["cells"])
        self.assertIn('BRANCH = "exp/4-celebdf-baseline-audit"', source)
        self.assertIn('FRAMES_PER_VIDEO_VALUES = "1,5,10"', source)
        self.assertIn('REFERENCE_COUNTS = "1,3,5"', source)
        self.assertIn('onnxruntime-gpu==1.23.2', source)
        self.assertNotIn("Chunbae-A/deepsogak", source)
        self.assertNotIn("/content/deepsogak", source)

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
        expected_paths = [
            "scripts/celebdf_faceguard.py",
            "scripts/run_celebdf_arcface.py",
            "scripts/audit_celebdf_baseline.py",
        ]
        self.assertEqual(list(embedded), expected_paths)
        for relative in expected_paths:
            self.assertEqual(base64.b64decode(embedded[relative]), (ROOT / relative).read_bytes())

        expected_fingerprint = hashlib.sha256(
            b"".join((ROOT / relative).read_bytes() for relative in expected_paths)
        ).hexdigest()
        self.assertIn(f'EMBEDDED_CODE_SHA256 = "{expected_fingerprint}"', bootstrap)


if __name__ == "__main__":
    unittest.main()
