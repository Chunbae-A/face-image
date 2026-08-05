import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "validate_faceguard_manifest.py"
SPEC = importlib.util.spec_from_file_location("validate_faceguard_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


HEADER = "path,subject_id,split,source_id,sha256,is_augmented\n"


class ManifestTests(unittest.TestCase):
    def _validate(self, body: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(HEADER + body, encoding="utf-8")
            return validator.validate_manifest(path)

    def test_valid_manifest(self):
        report = self._validate(
            "a.jpg,s1,train,src1,aaa,false\n"
            "b.jpg,s2,validation,src2,bbb,false\n"
            "c.jpg,s3,test,src3,ccc,false\n"
        )
        self.assertTrue(report["valid"])

    def test_subject_leakage_is_rejected(self):
        report = self._validate(
            "a.jpg,s1,train,src1,aaa,false\n"
            "b.jpg,s1,test,src2,bbb,false\n"
        )
        self.assertFalse(report["valid"])
        self.assertIn(
            "cross_split_subject",
            {finding["code"] for finding in report["findings"]},
        )

    def test_test_augmentation_is_rejected(self):
        report = self._validate("a.jpg,s1,test,src1,aaa,true\n")
        self.assertFalse(report["valid"])
        self.assertIn(
            "augmentation_outside_train",
            {finding["code"] for finding in report["findings"]},
        )


if __name__ == "__main__":
    unittest.main()
