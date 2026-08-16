from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kface_full_status", ROOT / "scripts" / "kface_full_status.py"
)
assert SPEC is not None and SPEC.loader is not None
kface_full_status = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kface_full_status
SPEC.loader.exec_module(kface_full_status)


def write_checkpoint(
    root: Path,
    *,
    subject: str,
    chunk: int,
    fingerprint: str = "a" * 64,
) -> None:
    path = root / "subjects" / subject / "checkpoints" / f"chunk_{chunk:05d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "complete": True,
                "config_fingerprint": fingerprint,
                "selected_pairs": 2,
                "accepted_pairs": 1,
                "rejected_pairs": 1,
                "reject_reasons": {"NO_FACE": 1},
                "elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )


class KFaceFullStatusTests(unittest.TestCase):
    def test_summarizes_paired_images_and_eta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_checkpoint(root, subject="subject_a", chunk=0)
            write_checkpoint(root, subject="subject_a", chunk=1)

            result = kface_full_status.summarize(
                root, total_subjects=2, pairs_per_subject=4
            )

            self.assertEqual(result["status"], "running")
            self.assertEqual(result["complete_subjects"], 1)
            self.assertEqual(result["total_images"], 16)
            self.assertEqual(result["processed_images"], 8)
            self.assertEqual(result["accepted_images"], 4)
            self.assertEqual(result["rejected_images"], 4)
            self.assertEqual(result["progress_percent"], 50.0)
            self.assertEqual(result["reject_reasons_by_pair"], {"NO_FACE": 2})

    def test_rejects_mixed_configuration_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_checkpoint(root, subject="subject_a", chunk=0)
            write_checkpoint(
                root, subject="subject_b", chunk=0, fingerprint="b" * 64
            )

            with self.assertRaisesRegex(RuntimeError, "서로 다른 설정"):
                kface_full_status.summarize(
                    root, total_subjects=2, pairs_per_subject=2
                )


if __name__ == "__main__":
    unittest.main()
