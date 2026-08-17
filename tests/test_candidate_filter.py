import asyncio
import json
import math
import unittest
from datetime import datetime, timezone

import numpy as np

from faceguard_api.candidate_filter import CandidateFilterService
from faceguard_api.deepfake import DeepfakeAnalysis
from faceguard_api.domain import EncodedFace, FaceQuality
from faceguard_api.errors import FaceGuardError
from faceguard_api.media import CandidateDownloadError, DownloadedImage
from faceguard_api.search import SearchCandidate
from faceguard_api.settings import Settings

FIXED_TIME = datetime(2026, 8, 7, tzinfo=timezone.utc)
QUALITY = FaceQuality(0.99, 0.25, 100.0, 128.0, 640, 480)


class FakeEncoder:
    loaded = True
    provider = "FakeExecutionProvider"
    model_fingerprint = "a" * 64

    def encode(self, payload: bytes) -> EncodedFace:
        if payload == b"no-face":
            raise FaceGuardError("NO_FACE", "얼굴을 찾지 못했습니다.")
        if payload == b"broad":
            embedding = np.array([0.3, math.sqrt(1.0 - 0.3**2)], dtype=np.float32)
        elif payload == b"different":
            embedding = np.array([0.0, 1.0], dtype=np.float32)
        else:
            embedding = np.array([1.0, 0.0], dtype=np.float32)
        return EncodedFace(embedding=embedding, quality=QUALITY)


class FakeDownloader:
    async def download(self, url: str) -> DownloadedImage:
        if "download-fail" in url:
            raise CandidateDownloadError("CANDIDATE_DOWNLOAD_FAILED")
        if "broad" in url:
            payload = b"broad"
        elif "different" in url:
            payload = b"different"
        elif "no-face" in url:
            payload = b"no-face"
        else:
            payload = b"same"
        return DownloadedImage(payload, url, "image/jpeg")


class FakeDeepfakeAnalyzer:
    def analyze(self, payload: bytes) -> DeepfakeAnalysis:
        score = 0.9 if payload == b"same" else 0.4
        return DeepfakeAnalysis(
            is_suspected_deepfake=score >= 0.75,
            deepfake_score=score,
            raw_logit=2.0 if score >= 0.75 else -0.4,
            threshold=0.75,
            quality=QUALITY,
            processing_ms=10.0,
            inference_ms=4.0,
            model_name="fake",
            execution_provider="FakeDeepfakeProvider",
            model_fingerprint="b" * 64,
        )


class UnavailableDeepfakeAnalyzer:
    def analyze(self, payload: bytes) -> DeepfakeAnalysis:
        del payload
        raise FaceGuardError("MODEL_UNAVAILABLE", "모델 없음", 503)


def candidate(name: str, rank: int) -> SearchCandidate:
    return SearchCandidate(
        page_url=f"https://example.com/{name}",
        media_url=f"https://cdn.example.com/{name}.jpg",
        thumbnail_url=None,
        provider="google_vision",
        providers=("google_vision",),
        rank=rank,
        retrieved_at=FIXED_TIME,
        source_engine="vision_web_detection",
    )


class CandidateFilterServiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            accept_noncommercial_model_license=True,
            retrieval_similarity_threshold=0.2,
            similarity_threshold=0.5,
        )
        self.service = CandidateFilterService(
            self.settings,
            FakeEncoder(),
            FakeDownloader(),
            FakeDeepfakeAnalyzer(),
        )

    def test_batch_reports_identity_retrieval_rejection_and_partial_failure(self):
        result = asyncio.run(
            self.service.filter(
                [b"ref-1", b"ref-2", b"ref-3"],
                [
                    candidate("same", 1),
                    candidate("broad", 2),
                    candidate("different", 3),
                    candidate("no-face", 4),
                    candidate("download-fail", 5),
                ],
            )
        )
        self.assertEqual(result.reference_count, 3)
        self.assertEqual(result.analyzed_candidate_count, 3)
        self.assertEqual(result.skipped_candidate_count, 2)
        self.assertEqual(result.retrieval_match_count, 2)
        self.assertEqual(result.identity_match_count, 1)
        self.assertEqual(result.deepfake_analyzed_candidate_count, 2)
        self.assertEqual(result.deepfake_suspected_candidate_count, 1)
        self.assertEqual(result.deepfake_failed_candidate_count, 0)
        self.assertEqual(result.candidates[0].deepfake.deepfake_score, 0.9)
        self.assertEqual(result.candidates[2].deepfake.status, "not_analyzed")
        self.assertEqual(
            [item.status for item in result.candidates],
            [
                "identity_match",
                "retrieval_match",
                "not_matched",
                "skipped",
                "skipped",
            ],
        )
        self.assertEqual(result.candidates[3].error_code, "NO_FACE")
        self.assertEqual(
            result.candidates[4].error_code, "CANDIDATE_DOWNLOAD_FAILED"
        )
        serialized = json.dumps(result, default=lambda value: value.__dict__)
        self.assertNotIn("embedding", serialized)
        self.assertNotIn("payload", serialized)

    def test_references_are_prepared_before_candidates(self):
        prepared = asyncio.run(
            self.service.prepare_references([b"ref-1", b"ref-2", b"ref-3"])
        )
        result = asyncio.run(
            self.service.filter_prepared(prepared, [candidate("same", 1)])
        )
        self.assertEqual(result.identity_match_count, 1)
        self.assertEqual(result.reference_count, 3)

    def test_progress_callback_reports_each_finished_candidate(self):
        progress = []

        async def scenario():
            prepared = await self.service.prepare_references([b"ref"])

            async def record(decisions):
                progress.append(len(decisions))

            await self.service.filter_prepared(
                prepared,
                [candidate("same", 1), candidate("different", 2)],
                progress_callback=record,
            )

        asyncio.run(scenario())
        self.assertEqual(progress, [1, 2])

    def test_limits_pipeline_candidate_count(self):
        settings = Settings(
            accept_noncommercial_model_license=True,
            retrieval_similarity_threshold=0.2,
            similarity_threshold=0.5,
            maximum_pipeline_candidates=1,
        )
        service = CandidateFilterService(settings, FakeEncoder(), FakeDownloader())
        with self.assertRaises(FaceGuardError) as raised:
            asyncio.run(
                service.filter(
                    [b"ref"], [candidate("same", 1), candidate("different", 2)]
                )
            )
        self.assertEqual(raised.exception.code, "TOO_MANY_PIPELINE_CANDIDATES")

    def test_deepfake_failure_keeps_arcface_result_without_fake_score(self):
        service = CandidateFilterService(
            self.settings,
            FakeEncoder(),
            FakeDownloader(),
            UnavailableDeepfakeAnalyzer(),
        )
        result = asyncio.run(service.filter([b"ref"], [candidate("same", 1)]))
        self.assertEqual(result.candidates[0].status, "identity_match")
        self.assertEqual(result.candidates[0].deepfake.status, "unavailable")
        self.assertIsNone(result.candidates[0].deepfake.deepfake_score)
        self.assertEqual(result.deepfake_failed_candidate_count, 1)


if __name__ == "__main__":
    unittest.main()
