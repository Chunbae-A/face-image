import json
import math
import time
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient

from faceguard_api.app import create_app
from faceguard_api.deepfake import DeepfakeAnalysis
from faceguard_api.domain import EncodedFace, FaceQuality
from faceguard_api.errors import FaceGuardError
from faceguard_api.exposure import EphemeralEnrollmentStore
from faceguard_api.media import CandidateDownloadError, DownloadedImage
from faceguard_api.settings import Settings

QUALITY = FaceQuality(0.99, 0.25, 100.0, 128.0, 640, 480)


class FakeEncoder:
    loaded = True
    provider = "FakeExecutionProvider"
    model_fingerprint = "a" * 64

    def encode(self, payload: bytes) -> EncodedFace:
        if payload == b"different":
            embedding = np.array([0.0, 1.0], dtype=np.float32)
        elif payload == b"broad":
            embedding = np.array([0.3, math.sqrt(1.0 - 0.3**2)], dtype=np.float32)
        else:
            embedding = np.array([1.0, 0.0], dtype=np.float32)
        return EncodedFace(embedding=embedding, quality=QUALITY)


class FakeDownloader:
    async def download(self, url: str) -> DownloadedImage:
        if "download-fail" in url:
            raise CandidateDownloadError("CANDIDATE_DOWNLOAD_FAILED")
        if "different" in url:
            payload = b"different"
        elif "broad" in url:
            payload = b"broad"
        else:
            payload = b"same"
        return DownloadedImage(payload, url, "image/jpeg")


class FakeDeepfakeAnalyzer:
    loaded = True
    provider = "FakeDeepfakeExecutionProvider"
    model_fingerprint = "b" * 64

    def analyze(self, payload: bytes) -> DeepfakeAnalysis:
        score = 0.9 if payload == b"same" else 0.1
        return DeepfakeAnalysis(
            is_suspected_deepfake=score >= 0.75,
            deepfake_score=score,
            raw_logit=2.0,
            threshold=0.75,
            quality=QUALITY,
            processing_ms=5.0,
            inference_ms=2.0,
            model_name="fake",
            execution_provider=self.provider,
            model_fingerprint=self.model_fingerprint,
        )


def settings(**changes) -> Settings:
    values = {
        "accept_noncommercial_model_license": True,
        "similarity_threshold": 0.5,
        "retrieval_similarity_threshold": 0.2,
        "exposure_enrollment_ttl_seconds": 60,
        "exposure_scan_ttl_seconds": 60,
    }
    values.update(changes)
    return Settings(**values)


def image_file(field: str, name: str, payload: bytes):
    return (field, (name, payload, "image/jpeg"))


class ExposureScanHttpTests(unittest.TestCase):
    def make_app(self, **setting_changes):
        return create_app(
            settings(**setting_changes),
            FakeEncoder(),
            image_downloader=FakeDownloader(),
            deepfake_analyzer=FakeDeepfakeAnalyzer(),
        )

    @staticmethod
    def wait_for_final(client: TestClient, status_url: str) -> dict:
        for _ in range(100):
            response = client.get(status_url)
            if response.json()["status"] in {
                "completed",
                "partial_failed",
                "failed",
            }:
                return response.json()
            time.sleep(0.005)
        raise AssertionError("비동기 스캔이 시간 내에 끝나지 않았습니다.")

    def test_end_to_end_async_scan_returns_progress_candidates_and_detail(self):
        with TestClient(self.make_app()) as client:
            enrollment = client.post(
                "/v1/faceguard/enrollments",
                files=[
                    image_file("reference_images", "ref-1.jpg", b"ref-1"),
                    image_file("reference_images", "ref-2.jpg", b"ref-2"),
                    image_file("reference_images", "ref-3.jpg", b"ref-3"),
                ],
            )
            self.assertEqual(enrollment.status_code, 201, enrollment.text)
            enrollment_body = enrollment.json()
            self.assertEqual(enrollment_body["storage"], "memory_only")
            self.assertEqual(enrollment_body["reference_count"], 3)

            scan = client.post(
                "/v1/exposure-scans",
                headers={"Idempotency-Key": "demo-scan-0001"},
                json={
                    "enrollment_id": enrollment_body["enrollment_id"],
                    "privacy_mode": "privacy_strict",
                    "candidates": [
                        {
                            "page_url": "https://example.com/same-post",
                            "media_url": "https://cdn.example.com/same.jpg",
                        },
                        {
                            "page_url": "https://example.com/different-post",
                            "media_url": "https://cdn.example.com/different.jpg",
                        },
                    ],
                },
            )
            self.assertEqual(scan.status_code, 202, scan.text)
            created = scan.json()
            self.assertFalse(created["reused"])
            self.assertEqual(
                created["client_candidates_url"],
                f"/v1/exposure-scans/{created['scan_id']}/client-candidates",
            )

            status = self.wait_for_final(client, created["status_url"])
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["progress_percent"], 100)
            self.assertEqual(status["progress"]["searched_candidate_count"], 2)
            self.assertEqual(status["progress"]["identity_match_count"], 1)
            self.assertEqual(status["progress"]["deepfake_completed_count"], 1)
            self.assertIn("searching", status["stage_durations_ms"])
            self.assertIn("identity_filtering", status["stage_durations_ms"])
            self.assertIn("deepfake_analyzing", status["stage_durations_ms"])

            candidates = client.get(created["candidates_url"])
            self.assertEqual(candidates.status_code, 200, candidates.text)
            candidate_body = candidates.json()
            self.assertEqual(candidate_body["candidate_count"], 2)
            self.assertEqual(
                [item["result"]["status"] for item in candidate_body["candidates"]],
                ["identity_match", "not_matched"],
            )
            first = candidate_body["candidates"][0]
            self.assertEqual(first["result"]["deepfake"]["deepfake_score"], 0.9)
            detail = client.get(
                f"/v1/exposure-candidates/{first['candidate_id']}/analysis"
            )
            self.assertEqual(detail.status_code, 200, detail.text)
            self.assertEqual(detail.json()["candidate_id"], first["candidate_id"])

            client_candidates = client.get(created["client_candidates_url"])
            self.assertEqual(
                client_candidates.status_code, 200, client_candidates.text
            )
            client_body = client_candidates.json()
            self.assertEqual(client_body["candidate_count"], 2)
            self.assertEqual(client_body["identity_match_count"], 1)
            self.assertEqual(client_body["review_candidate_count"], 1)
            self.assertEqual(
                client_body["candidates"][0]["recommended_action"],
                "review_required",
            )
            self.assertEqual(
                client_body["candidates"][0]["deepfake_signal"], "suspected"
            )
            self.assertEqual(
                client_body["candidates"][1]["recommended_action"],
                "exclude_recommended",
            )
            self.assertNotIn("probability", json.dumps(client_body))

            serialized = json.dumps(
                {
                    "enrollment": enrollment_body,
                    "status": status,
                    "detail": detail.json(),
                },
                ensure_ascii=False,
            )
            self.assertNotIn("embedding", serialized)
            self.assertNotIn("ref-1", serialized)
            internal_record = client.app.state.exposure_scan_manager._records[
                created["scan_id"]
            ]
            self.assertEqual(internal_record.query.submitted_candidates, ())
            self.assertNotEqual(internal_record.idempotency_key, "demo-scan-0001")

    def test_client_candidates_marks_broad_face_match_for_identity_review(self):
        with TestClient(self.make_app()) as client:
            enrollment_id = client.post(
                "/v1/faceguard/enrollments",
                files=[image_file("reference_images", "ref.jpg", b"ref")],
            ).json()["enrollment_id"]
            created = client.post(
                "/v1/exposure-scans",
                json={
                    "enrollment_id": enrollment_id,
                    "candidates": [
                        {
                            "page_url": "https://example.com/broad-post",
                            "media_url": "https://cdn.example.com/broad.jpg",
                        }
                    ],
                },
            ).json()
            self.wait_for_final(client, created["status_url"])

            body = client.get(created["client_candidates_url"]).json()
            candidate = body["candidates"][0]
            self.assertEqual(candidate["face_match_level"], "review")
            self.assertEqual(
                candidate["recommended_action"], "identity_review_required"
            )
            self.assertEqual(body["identity_match_count"], 0)
            self.assertEqual(body["review_candidate_count"], 1)

    def test_idempotency_reuses_same_scan_and_rejects_different_request(self):
        with TestClient(self.make_app()) as client:
            enrollment_id = client.post(
                "/v1/faceguard/enrollments",
                files=[image_file("reference_images", "ref.jpg", b"ref")],
            ).json()["enrollment_id"]
            body = {
                "enrollment_id": enrollment_id,
                "candidates": [{"page_url": "https://example.com/one"}],
            }
            first = client.post(
                "/v1/exposure-scans",
                headers={"Idempotency-Key": "retry-safe-key"},
                json=body,
            )
            second = client.post(
                "/v1/exposure-scans",
                headers={"Idempotency-Key": "retry-safe-key"},
                json=body,
            )
            self.assertEqual(first.status_code, 202, first.text)
            self.assertEqual(second.status_code, 202, second.text)
            self.assertEqual(first.json()["scan_id"], second.json()["scan_id"])
            self.assertTrue(second.json()["reused"])

            changed = client.post(
                "/v1/exposure-scans",
                headers={"Idempotency-Key": "retry-safe-key"},
                json={
                    "enrollment_id": enrollment_id,
                    "candidates": [{"page_url": "https://example.com/two"}],
                },
            )
            self.assertEqual(changed.status_code, 409, changed.text)
            self.assertEqual(changed.json()["error"]["code"], "IDEMPOTENCY_KEY_REUSED")

    def test_candidate_failure_is_partial_and_keeps_successful_result(self):
        with TestClient(self.make_app()) as client:
            enrollment_id = client.post(
                "/v1/faceguard/enrollments",
                files=[image_file("reference_images", "ref.jpg", b"ref")],
            ).json()["enrollment_id"]
            created = client.post(
                "/v1/exposure-scans",
                json={
                    "enrollment_id": enrollment_id,
                    "candidates": [
                        {"page_url": "https://example.com/same"},
                        {"page_url": "https://example.com/download-fail"},
                    ],
                },
            ).json()
            status = self.wait_for_final(client, created["status_url"])
            self.assertEqual(status["status"], "partial_failed")
            self.assertEqual(status["progress"]["analyzed_candidate_count"], 1)
            self.assertEqual(status["progress"]["skipped_candidate_count"], 1)
            candidates = client.get(created["candidates_url"]).json()["candidates"]
            self.assertEqual(len(candidates), 2)
            self.assertEqual(
                candidates[1]["result"]["error_code"], "CANDIDATE_DOWNLOAD_FAILED"
            )

    def test_unknown_enrollment_is_rejected_before_job_creation(self):
        with TestClient(self.make_app()) as client:
            response = client.post(
                "/v1/exposure-scans",
                json={
                    "enrollment_id": "missing",
                    "candidates": [{"page_url": "https://example.com/post"}],
                },
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["error"]["code"], "ENROLLMENT_NOT_FOUND")

    def test_expired_enrollment_is_removed(self):
        now = datetime(2026, 8, 8, tzinfo=timezone.utc)

        class MutableClock:
            value = now

            def __call__(self):
                return self.value

        clock = MutableClock()
        store = EphemeralEnrollmentStore(ttl_seconds=1, maximum_entries=1, clock=clock)
        prepared = type(
            "Prepared",
            (),
            {"faces": (), "embedding": np.array([1.0]), "processing_ms": 0.0},
        )()

        async def scenario():
            record = await store.create(prepared)
            clock.value += timedelta(seconds=2)
            with self.assertRaises(FaceGuardError) as raised:
                await store.get(record.enrollment_id)
            self.assertEqual(raised.exception.code, "ENROLLMENT_EXPIRED")

        import asyncio

        asyncio.run(scenario())

    def test_active_enrollment_is_not_silently_evicted_at_capacity(self):
        store = EphemeralEnrollmentStore(ttl_seconds=60, maximum_entries=1)
        prepared = type(
            "Prepared",
            (),
            {"faces": (), "embedding": np.array([1.0]), "processing_ms": 0.0},
        )()

        async def scenario():
            first = await store.create(prepared)
            with self.assertRaises(FaceGuardError) as raised:
                await store.create(prepared)
            self.assertEqual(raised.exception.code, "ENROLLMENT_CAPACITY_EXCEEDED")
            self.assertEqual(
                (await store.get(first.enrollment_id)).enrollment_id,
                first.enrollment_id,
            )

        import asyncio

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
