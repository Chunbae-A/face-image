import importlib.util
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "faceguard_plan.py"
SPEC = importlib.util.spec_from_file_location("faceguard_plan", MODULE_PATH)
assert SPEC and SPEC.loader
faceguard_plan = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = faceguard_plan
SPEC.loader.exec_module(faceguard_plan)


class DownloadTests(unittest.TestCase):
    def test_100gb_at_100mbps(self):
        self.assertEqual(faceguard_plan.download_seconds(100, 100), 8_000)

    def test_efficiency(self):
        self.assertEqual(faceguard_plan.download_seconds(100, 100, 0.8), 10_000)

    def test_invalid_efficiency(self):
        with self.assertRaises(ValueError):
            faceguard_plan.download_seconds(100, 100, 0)


class StorageTests(unittest.TestCase):
    def test_embedding_size(self):
        self.assertTrue(
            math.isclose(faceguard_plan.embedding_gb(1_000_000), 2.048)
        )

    def test_recommended_is_at_least_three_times_archive(self):
        result = faceguard_plan.storage_estimate(
            compressed_gb=100,
            unpacked_gb=100,
            preprocessed_gb=0,
            images=0,
            checkpoints_gb=0,
            logs_gb=0,
            headroom=1,
        )
        self.assertEqual(result["recommended_gb"], 300)


if __name__ == "__main__":
    unittest.main()
