from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kface_pilot", ROOT / "scripts" / "kface_pilot.py"
)
assert SPEC is not None and SPEC.loader is not None
kface_pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kface_pilot
SPEC.loader.exec_module(kface_pilot)


class FakeEncoder:
    provider = "FakeExecutionProvider"
    model_fingerprint = "f" * 64

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, payload: bytes) -> SimpleNamespace:
        self.calls += 1
        if payload == b"reject":
            error = ValueError("reject")
            error.code = "NO_FACE"  # type: ignore[attr-defined]
            raise error
        vector = np.zeros(512, dtype=np.float32)
        vector[self.calls % 512] = 1.0
        quality = SimpleNamespace(
            detection_score=0.9,
            face_area_ratio=0.2,
            blur_score=100.0,
            brightness_mean=120.0,
            image_width=224,
            image_height=224,
        )
        return SimpleNamespace(embedding=vector, quality=quality)


def build_nested_archive(path: Path, *, subjects: int = 3, images: int = 5) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as outer:
        for subject in range(subjects):
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as inner:
                for image in range(images):
                    payload = b"reject" if image == images - 1 else f"{subject}-{image}".encode()
                    inner.writestr(f"condition/image_{image:03d}.jpg", payload)
                inner.writestr("labels/readme.txt", b"ignored")
            outer.writestr(f"subjects/person_{subject:03d}.zip", buffer.getvalue())


class KFacePilotTests(unittest.TestCase):
    def test_script_can_import_project_package_from_another_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code = f"""
import argparse
import importlib.util
import pathlib
import sys
path = pathlib.Path({str(ROOT / 'scripts' / 'kface_pilot.py')!r})
spec = importlib.util.spec_from_file_location('kface_cli_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
encoder = module._build_encoder(argparse.Namespace(
    accept_noncommercial_model_license=True,
    model_root=pathlib.Path('unused-model-root'),
))
print(type(encoder).__name__)
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("InsightFaceEncoder", completed.stdout)

    def test_archive_size_check_rejects_partial_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "partial.zip"
            archive.write_bytes(b"partial")

            with self.assertRaisesRegex(IOError, "ZIP 크기가 다릅니다"):
                kface_pilot.validate_archive_size(archive, 100)

    def test_inventory_counts_nested_archives_without_raw_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "Low_Resolution.zip"
            build_nested_archive(archive, subjects=3, images=5)

            payload = kface_pilot.inventory_archive(
                archive, resolution="low", inspect_subjects=2
            )

            self.assertEqual(payload["subject_archive_count"], 3)
            self.assertEqual(payload["inspected_subject_count"], 2)
            self.assertEqual(payload["inspected_image_count"], 10)
            self.assertFalse(payload["contains_raw_paths"])
            self.assertNotIn("person_000", json.dumps(payload))

    def test_process_is_bounded_and_resumes_completed_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "Low_Resolution.zip"
            output = root / "private-output"
            build_nested_archive(archive, subjects=3, images=5)
            encoder = FakeEncoder()

            first = kface_pilot.process_archive(
                archive,
                resolution="low",
                output_dir=output,
                max_subjects=2,
                images_per_subject=3,
                encoder=encoder,
                archive_sha256="a" * 64,
            )

            self.assertEqual(first["subject_count"], 2)
            self.assertEqual(first["selected_images"], 6)
            self.assertEqual(first["accepted_images"], 4)
            self.assertEqual(first["rejected_images"], 2)
            self.assertEqual(encoder.calls, 6)
            checkpoints = sorted((output / "checkpoints").glob("*.json"))
            embeddings = sorted((output / "embeddings").glob("*.npz"))
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(len(embeddings), 2)

            resumed_encoder = FakeEncoder()
            resumed = kface_pilot.process_archive(
                archive,
                resolution="low",
                output_dir=output,
                max_subjects=2,
                images_per_subject=3,
                encoder=resumed_encoder,
                archive_sha256="a" * 64,
            )

            self.assertEqual(resumed["skipped_completed_subjects"], 2)
            self.assertEqual(resumed_encoder.calls, 0)

    def test_rejects_unsafe_inner_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "Low_Resolution.zip"
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as inner:
                inner.writestr("../escape.jpg", b"image")
            with zipfile.ZipFile(archive, "w") as outer:
                outer.writestr("person.zip", buffer.getvalue())

            with self.assertRaisesRegex(ValueError, "안전하지 않은 경로"):
                kface_pilot.inventory_archive(archive, resolution="low")


if __name__ == "__main__":
    unittest.main()
