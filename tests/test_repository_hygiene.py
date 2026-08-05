import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_repository_hygiene.py"
SPEC = importlib.util.spec_from_file_location("check_repository_hygiene", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RepositoryHygieneTests(unittest.TestCase):
    def check(self, root: Path, *relative_paths: str, max_file_bytes: int = 5_000_000):
        paths = [root / relative for relative in relative_paths]
        return MODULE.check_paths(root, paths, max_file_bytes=max_file_bytes)

    def test_clean_text_file_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("safe\n", encoding="utf-8")
            self.assertEqual(self.check(root, "README.md"), [])

    def test_forbidden_artifact_and_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "data" / "raw" / "faces.zip"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"not real face data")
            rules = {item.rule for item in self.check(root, "data/raw/faces.zip")}
            self.assertEqual(rules, {"forbidden-artifact", "forbidden-data-path"})

    def test_large_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "large.txt").write_bytes(b"x" * 11)
            violations = self.check(root, "large.txt", max_file_bytes=10)
            self.assertEqual([item.rule for item in violations], ["large-file"])

    def test_executed_notebook_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "metadata": {},
                        "outputs": [{"output_type": "stream", "name": "stdout", "text": ["x"]}],
                        "source": ["print('x')"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            (root / "run.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
            rules = {item.rule for item in self.check(root, "run.ipynb")}
            self.assertEqual(rules, {"notebook-execution-count", "notebook-output"})

    def test_secret_pattern_is_rejected_without_storing_a_real_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_token = "ghp_" + "a" * 36
            (root / "secret.txt").write_text(fake_token, encoding="utf-8")
            violations = self.check(root, "secret.txt")
            self.assertEqual([item.rule for item in violations], ["secret-pattern"])


if __name__ == "__main__":
    unittest.main()
