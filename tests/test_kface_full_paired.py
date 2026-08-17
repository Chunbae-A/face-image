from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kface_full_paired", ROOT / "scripts" / "kface_full_paired.py"
)
assert SPEC is not None and SPEC.loader is not None
kface_full_paired = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kface_full_paired
SPEC.loader.exec_module(kface_full_paired)


class FakePairedEncoder:
    provider = "FakeExecutionProvider"
    model_name = "fake_arcface"
    model_fingerprint = "f" * 64
    detection_size = 160
    minimum_detection_score = 0.5
    minimum_face_area_ratio = 0.01
    recognition_batch_size = 64
    recognition_provider = "FakeExecutionProvider"
    fast_detection_size = None
    fast_detection_score = 0.5

    def __init__(self) -> None:
        self.calls = 0

    def encode_payload_pairs(
        self,
        payload_pairs: list[tuple[bytes, bytes]],
        image_indices: list[int],
    ) -> kface_full_paired.PairedChunkResult:
        self.calls += 1
        count = len(payload_pairs)
        embeddings = np.zeros((count, 512), dtype=np.float32)
        for row, image_index in enumerate(image_indices):
            embeddings[row, image_index % 512] = 1.0
        quality = np.tile(
            np.asarray([0.9, 0.2, 100.0, 120.0, 346.0, 230.0], dtype=np.float32),
            (count, 1),
        )
        return kface_full_paired.PairedChunkResult(
            image_indices=np.asarray(image_indices, dtype=np.int32),
            low_embeddings=embeddings,
            medium_embeddings=embeddings,
            low_quality=quality,
            medium_quality=quality,
            reject_reasons={},
        )


def build_paired_archives(
    low_path: Path,
    medium_path: Path,
    *,
    subjects: int = 2,
    images: int = 5,
    mismatch: bool = False,
) -> None:
    for resolution, output in (("low", low_path), ("medium", medium_path)):
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as outer:
            for subject in range(subjects):
                buffer = BytesIO()
                with zipfile.ZipFile(
                    buffer, "w", compression=zipfile.ZIP_STORED
                ) as inner:
                    for image in range(images):
                        name = f"condition/image_{image:03d}.jpg"
                        if mismatch and resolution == "medium" and image == images - 1:
                            name = "condition/different.jpg"
                        inner.writestr(name, f"{resolution}-{subject}-{image}".encode())
                outer.writestr(f"subjects/person_{subject:03d}.zip", buffer.getvalue())


class KFaceFullPairedTests(unittest.TestCase):
    def test_processes_chunks_and_resumes_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "Low_Resolution.zip"
            medium = root / "Middle_Resolution.zip"
            output = root / "private-output"
            build_paired_archives(low, medium)

            encoder = FakePairedEncoder()
            first = kface_full_paired.process_paired_archives(
                low,
                medium,
                output_dir=output,
                subject_start=1,
                subject_end=2,
                chunk_size=2,
                encoder=encoder,
                low_archive_sha256="a" * 64,
                medium_archive_sha256="b" * 64,
            )

            self.assertEqual(first["selected_images"], 20)
            self.assertEqual(first["accepted_pairs"], 10)
            self.assertEqual(first["processed_chunks"], 6)
            self.assertEqual(encoder.calls, 6)
            self.assertEqual(len(list(output.glob("subjects/*/chunks/*.npz"))), 6)
            self.assertNotIn("person_000", json.dumps(first))

            resumed_encoder = FakePairedEncoder()
            resumed = kface_full_paired.process_paired_archives(
                low,
                medium,
                output_dir=output,
                subject_start=1,
                subject_end=2,
                chunk_size=2,
                encoder=resumed_encoder,
                low_archive_sha256="a" * 64,
                medium_archive_sha256="b" * 64,
            )

            self.assertEqual(resumed["skipped_chunks"], 6)
            self.assertEqual(resumed_encoder.calls, 0)

    def test_smoke_limit_is_explicit_in_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "Low_Resolution.zip"
            medium = root / "Middle_Resolution.zip"
            build_paired_archives(low, medium, subjects=1, images=5)

            result = kface_full_paired.process_paired_archives(
                low,
                medium,
                output_dir=root / "private-output",
                subject_start=1,
                subject_end=1,
                chunk_size=2,
                maximum_pairs_per_subject=3,
                encoder=FakePairedEncoder(),
                low_archive_sha256="a" * 64,
                medium_archive_sha256="b" * 64,
            )

            self.assertEqual(result["selected_images"], 6)
            self.assertEqual(result["maximum_pairs_per_subject"], 3)
            self.assertFalse(result["full_subject_images_selected"])

    def test_rejects_mismatched_low_and_medium_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "Low_Resolution.zip"
            medium = root / "Middle_Resolution.zip"
            build_paired_archives(low, medium, subjects=1, mismatch=True)

            with self.assertRaisesRegex(ValueError, "1:1로 일치"):
                kface_full_paired.process_paired_archives(
                    low,
                    medium,
                    output_dir=root / "private-output",
                    subject_start=1,
                    subject_end=1,
                    chunk_size=2,
                    encoder=FakePairedEncoder(),
                    low_archive_sha256="a" * 64,
                    medium_archive_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
