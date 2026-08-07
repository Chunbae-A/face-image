import unittest

import numpy as np

from faceguard_api.deepfake import DeepfakeAnalysis
from faceguard_api.domain import AlignedEncodedFace, EncodedFace, FaceQuality
from faceguard_api.errors import FaceGuardError
from faceguard_api.settings import Settings
from faceguard_api.video import (
    DecodedVideo,
    DecodedVideoFrame,
    VideoDeepfakeAnalyzer,
    sample_video_frame_indices,
)

QUALITY = FaceQuality(0.99, 0.25, 120.0, 128.0, 640, 480)


class FakeVideoFaceEncoder:
    def encode(self, payload: bytes) -> EncodedFace:
        embedding = (
            np.array([0.0, 1.0], dtype=np.float32)
            if payload == b"reference-second"
            else np.array([1.0, 0.0], dtype=np.float32)
        )
        return EncodedFace(embedding, QUALITY)

    def encode_frame_faces(
        self, frame_bgr: np.ndarray, *, aligned_face_size: int
    ) -> tuple[AlignedEncodedFace, ...]:
        del aligned_face_size
        marker = int(frame_bgr[0, 0, 0])
        if marker == 99:
            raise FaceGuardError("NO_FACE", "얼굴 없음")
        if marker == 7:
            return (
                AlignedEncodedFace(
                    EncodedFace(np.array([1.0, 0.0], dtype=np.float32), QUALITY),
                    np.full((2, 2, 3), 10, dtype=np.uint8),
                ),
                AlignedEncodedFace(
                    EncodedFace(np.array([0.0, 1.0], dtype=np.float32), QUALITY),
                    np.full((2, 2, 3), 20, dtype=np.uint8),
                ),
            )
        return (
            AlignedEncodedFace(
                EncodedFace(np.array([1.0, 0.0], dtype=np.float32), QUALITY),
                np.full((2, 2, 3), marker, dtype=np.uint8),
            ),
        )


class FakeAlignedAnalyzer:
    def __init__(self, scores: list[float] | None = None):
        self.scores = list(scores or [])
        self.selected_markers: list[int] = []

    def analyze_aligned(self, aligned: AlignedEncodedFace) -> DeepfakeAnalysis:
        marker = int(aligned.aligned_bgr[0, 0, 0])
        self.selected_markers.append(marker)
        score = self.scores.pop(0) if self.scores else marker / 100.0
        return DeepfakeAnalysis(
            is_suspected_deepfake=score >= 0.75,
            deepfake_score=score,
            raw_logit=0.0,
            threshold=0.75,
            quality=aligned.face.quality,
            processing_ms=2.0,
            inference_ms=1.0,
            model_name="fake-video-model",
            execution_provider="FakeExecutionProvider",
            model_fingerprint="c" * 64,
        )


def decoded_video(markers: list[int]) -> DecodedVideo:
    return DecodedVideo(
        duration_seconds=float(len(markers)),
        fps=10.0,
        total_frame_count=len(markers) * 10,
        requested_frame_count=len(markers),
        frames=tuple(
            DecodedVideoFrame(
                frame_index=index * 10,
                timestamp_seconds=index + 0.5,
                bgr=np.full((2, 2, 3), marker, dtype=np.uint8),
            )
            for index, marker in enumerate(markers)
        ),
    )


class DeepfakeVideoTests(unittest.TestCase):
    def test_sampling_matches_training_preprocessing_policy(self):
        indices = sample_video_frame_indices(100, 16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 8)
        self.assertEqual(indices[-1], 91)
        self.assertEqual(sample_video_frame_indices(3, 16), [0, 1, 2])

    def test_video_score_is_mean_and_adjacent_suspicious_frames_are_grouped(self):
        settings = Settings(
            deepfake_threshold=0.75,
            deepfake_video_frame_count=4,
            deepfake_video_minimum_valid_frames=4,
        )
        frame_analyzer = FakeAlignedAnalyzer([0.1, 0.8, 0.9, 0.2])
        analyzer = VideoDeepfakeAnalyzer(
            settings,
            FakeVideoFaceEncoder(),
            frame_analyzer,
            decoder=lambda payload, suffix: decoded_video([1, 2, 3, 4]),
        )
        result = analyzer.analyze(b"video", suffix=".mp4")
        self.assertEqual(result.status, "completed")
        self.assertAlmostEqual(result.video_score, 0.5)
        self.assertFalse(result.is_suspected_deepfake)
        self.assertEqual(result.analyzed_frame_count, 4)
        self.assertEqual(len(result.suspicious_segments), 1)
        self.assertAlmostEqual(result.suspicious_segments[0].start_seconds, 1.0)
        self.assertAlmostEqual(result.suspicious_segments[0].end_seconds, 3.0)
        self.assertEqual(result.suspicious_segments[0].analyzed_frame_count, 2)

    def test_registered_face_selects_matching_person_from_multiple_faces(self):
        settings = Settings(
            retrieval_similarity_threshold=0.2,
            deepfake_video_frame_count=1,
            deepfake_video_minimum_valid_frames=1,
        )
        frame_analyzer = FakeAlignedAnalyzer()
        analyzer = VideoDeepfakeAnalyzer(
            settings,
            FakeVideoFaceEncoder(),
            frame_analyzer,
            decoder=lambda payload, suffix: decoded_video([7]),
        )
        result = analyzer.analyze(
            b"video",
            suffix=".mp4",
            reference_payloads=[b"reference-second"],
        )
        self.assertEqual(frame_analyzer.selected_markers, [20])
        self.assertAlmostEqual(result.frames[0].face_similarity, 1.0)
        self.assertEqual(result.reference_count, 1)

    def test_too_few_valid_face_frames_returns_stable_error(self):
        settings = Settings(
            deepfake_video_frame_count=4,
            deepfake_video_minimum_valid_frames=4,
        )
        analyzer = VideoDeepfakeAnalyzer(
            settings,
            FakeVideoFaceEncoder(),
            FakeAlignedAnalyzer([0.1, 0.2, 0.3]),
            decoder=lambda payload, suffix: decoded_video([1, 99, 2, 3]),
        )
        with self.assertRaises(FaceGuardError) as raised:
            analyzer.analyze(b"video", suffix=".mp4")
        self.assertEqual(raised.exception.code, "INSUFFICIENT_VALID_VIDEO_FRAMES")

    def test_decode_or_face_failure_is_counted_as_partial_result(self):
        settings = Settings(
            deepfake_video_frame_count=4,
            deepfake_video_minimum_valid_frames=2,
        )
        partial = DecodedVideo(
            duration_seconds=4.0,
            fps=10.0,
            total_frame_count=40,
            requested_frame_count=4,
            frames=decoded_video([1, 2, 3]).frames,
        )
        analyzer = VideoDeepfakeAnalyzer(
            settings,
            FakeVideoFaceEncoder(),
            FakeAlignedAnalyzer([0.1, 0.2, 0.3]),
            decoder=lambda payload, suffix: partial,
        )
        result = analyzer.analyze(b"video", suffix=".mp4")
        self.assertEqual(result.status, "partial_failed")
        self.assertEqual(result.decoded_frame_count, 3)
        self.assertEqual(result.analyzed_frame_count, 3)
        self.assertEqual(result.skipped_frame_count, 1)


if __name__ == "__main__":
    unittest.main()
