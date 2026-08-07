import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from faceguard_api.deepfake import DeepfakeOnnxAnalyzer, preprocess_aligned_face
from faceguard_api.domain import AlignedEncodedFace, EncodedFace, FaceQuality
from faceguard_api.errors import FaceGuardError
from faceguard_api.settings import Settings

QUALITY = FaceQuality(0.99, 0.25, 120.0, 128.0, 640, 480)


class FakeFaceEncoder:
    def encode_and_align(
        self, payload: bytes, *, aligned_face_size: int
    ) -> AlignedEncodedFace:
        del payload
        return AlignedEncodedFace(
            face=EncodedFace(np.array([1.0, 0.0], dtype=np.float32), QUALITY),
            aligned_bgr=np.full(
                (aligned_face_size, aligned_face_size, 3),
                (10, 20, 30),
                dtype=np.uint8,
            ),
        )


class FakeNode:
    def __init__(self, name: str, shape=None):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, logit: float, input_size: int):
        self.logit = logit
        self.input_size = input_size
        self.received = None

    def get_inputs(self):
        return [FakeNode("input", ["batch", 3, self.input_size, self.input_size])]

    def get_outputs(self):
        return [FakeNode("fake_logit")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, feeds):
        del output_names
        self.received = feeds["input"]
        return [np.asarray([[self.logit]], dtype=np.float32)]


class DeepfakeOnnxAnalyzerTests(unittest.TestCase):
    def build_analyzer(self, directory: str, *, logit: float = 2.0):
        model_path = Path(directory) / "model.onnx"
        model_path.write_bytes(b"private-test-model")
        fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()
        settings = Settings(
            accept_noncommercial_model_license=True,
            deepfake_model_path=model_path,
            deepfake_model_sha256=fingerprint,
            deepfake_threshold=0.75,
        )
        session = FakeSession(logit, settings.deepfake_input_size)
        analyzer = DeepfakeOnnxAnalyzer(
            settings,
            FakeFaceEncoder(),
            session_factory=lambda path, providers: session,
        )
        return analyzer, session, fingerprint

    def test_preprocess_matches_rgb_imagenet_normalization(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 2] = 255
        tensor = preprocess_aligned_face(bgr, 2)
        self.assertEqual(tensor.shape, (1, 3, 2, 2))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(tensor[0, 0, 0, 0], (1.0 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(tensor[0, 1, 0, 0], (0.0 - 0.456) / 0.224, places=5)

    def test_analyze_returns_finite_score_decision_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            analyzer, session, fingerprint = self.build_analyzer(directory, logit=2.0)
            result = analyzer.analyze(b"face")
        self.assertTrue(result.is_suspected_deepfake)
        self.assertAlmostEqual(result.deepfake_score, 1.0 / (1.0 + math.exp(-2.0)))
        self.assertEqual(result.raw_logit, 2.0)
        self.assertEqual(result.model_fingerprint, fingerprint)
        self.assertEqual(result.execution_provider, "CPUExecutionProvider")
        self.assertEqual(session.received.shape, (1, 3, 380, 380))
        self.assertTrue(analyzer.loaded)

    def test_score_below_threshold_is_not_suspected(self):
        with tempfile.TemporaryDirectory() as directory:
            analyzer, _, _ = self.build_analyzer(directory, logit=0.0)
            result = analyzer.analyze(b"face")
        self.assertFalse(result.is_suspected_deepfake)
        self.assertEqual(result.deepfake_score, 0.5)

    def test_missing_model_returns_stable_unavailable_error(self):
        settings = Settings(
            deepfake_model_path=Path("/definitely/missing/model.onnx"),
        )
        analyzer = DeepfakeOnnxAnalyzer(settings, FakeFaceEncoder())
        with self.assertRaises(FaceGuardError) as raised:
            analyzer.smoke()
        self.assertEqual(raised.exception.code, "MODEL_UNAVAILABLE")

    def test_hash_mismatch_is_rejected_before_session_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            model_path.write_bytes(b"unexpected")
            settings = Settings(
                deepfake_model_path=model_path,
                deepfake_model_sha256="0" * 64,
            )
            called = []
            analyzer = DeepfakeOnnxAnalyzer(
                settings,
                FakeFaceEncoder(),
                session_factory=lambda path, providers: called.append(path),
            )
            with self.assertRaises(FaceGuardError) as raised:
                analyzer.smoke()
        self.assertEqual(raised.exception.code, "MODEL_UNAVAILABLE")
        self.assertEqual(called, [])

    def test_nonfinite_model_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            analyzer, _, _ = self.build_analyzer(directory, logit=float("nan"))
            with self.assertRaises(FaceGuardError) as raised:
                analyzer.smoke()
        self.assertEqual(raised.exception.code, "MODEL_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
