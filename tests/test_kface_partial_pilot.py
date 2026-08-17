from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "kface_partial_pilot", SCRIPTS / "kface_partial_pilot.py"
)
assert SPEC is not None and SPEC.loader is not None
partial_pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = partial_pilot
SPEC.loader.exec_module(partial_pilot)


def nested_payload(index: int) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as inner:
        inner.writestr("image.jpg", f"image-{index}".encode())
    return buffer.getvalue()


def build_outer(path: Path, subjects: int) -> bytes:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        for index in range(subjects):
            outer.writestr(f"person_{index:03d}.zip", nested_payload(index))
    return path.read_bytes()


class KFacePartialPilotTests(unittest.TestCase):
    def test_builds_reference_matched_archive_from_completed_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.zip"
            full = build_outer(reference, 3)
            central_offset = full.find(b"PK\x01\x02")
            partial = root / "download.crdownload"
            partial.write_bytes(full[:central_offset])
            output = root / "pilot.zip"

            result = partial_pilot.build_pilot_archive(
                partial,
                reference_archive=reference,
                output_path=output,
                max_subjects=2,
            )

            self.assertEqual(result["selected_subjects"], 2)
            self.assertTrue(result["private_artifact"])
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    archive.namelist(), ["person_000.zip", "person_001.zip"]
                )

    def test_rejects_prefix_without_all_requested_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.zip"
            full = build_outer(reference, 2)
            partial = root / "download.crdownload"
            partial.write_bytes(full[:40])

            with self.assertRaisesRegex(RuntimeError, "아직 부족합니다"):
                partial_pilot.build_pilot_archive(
                    partial,
                    reference_archive=reference,
                    output_path=root / "pilot.zip",
                    max_subjects=2,
                )


if __name__ == "__main__":
    unittest.main()
