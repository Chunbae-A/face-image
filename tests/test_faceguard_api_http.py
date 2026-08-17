import json
import unittest
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient

from faceguard_api.app import create_app
from faceguard_api.calibration import ScoreCalibration
from faceguard_api.deepfake import DeepfakeAnalysis
from faceguard_api.domain import EncodedFace, FaceQuality
from faceguard_api.errors import FaceGuardError
from faceguard_api.media import DownloadedImage
from faceguard_api.search import (
    SearchService,
    SearXNGProvider,
    UserSubmittedUrlProvider,
)
from faceguard_api.settings import Settings
from faceguard_api.video import (
    DeepfakeVideoAnalysis,
    SuspiciousSegment,
    VideoFrameAnalysis,
)

QUALITY = FaceQuality(
    detection_score=0.99,
    face_area_ratio=0.25,
    blur_score=123.0,
    brightness_mean=128.0,
    image_width=640,
    image_height=480,
)


class FakeEncoder:
    loaded = True
    provider = "FakeExecutionProvider"
    model_fingerprint = "a" * 64

    def encode(self, payload: bytes) -> EncodedFace:
        if payload == b"no-face":
            raise FaceGuardError("NO_FACE", "얼굴을 찾지 못했습니다.")
        if payload == b"different":
            embedding = np.array([0.0, 1.0], dtype=np.float32)
        else:
            embedding = np.array([1.0, 0.0], dtype=np.float32)
        return EncodedFace(embedding=embedding, quality=QUALITY)


class FakeImageDownloader:
    async def download(self, url: str) -> DownloadedImage:
        payload = b"different" if "different" in url else b"same-candidate"
        return DownloadedImage(payload, url, "image/jpeg")


class FakeDeepfakeAnalyzer:
    loaded = True
    provider = "FakeDeepfakeExecutionProvider"
    model_fingerprint = "b" * 64

    def analyze(self, payload: bytes) -> DeepfakeAnalysis:
        score = 0.9 if payload != b"real" else 0.1
        return DeepfakeAnalysis(
            is_suspected_deepfake=score >= 0.75,
            deepfake_score=score,
            raw_logit=2.1972246 if score >= 0.75 else -2.1972246,
            threshold=0.75,
            quality=QUALITY,
            processing_ms=12.0,
            inference_ms=4.0,
            model_name="fake_deepfake_model",
            execution_provider=self.provider,
            model_fingerprint=self.model_fingerprint,
        )


class FakeVideoDeepfakeAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(
        self,
        payload: bytes,
        *,
        suffix: str,
        reference_payloads=(),
    ) -> DeepfakeVideoAnalysis:
        self.calls.append((payload, suffix, tuple(reference_payloads)))
        frames = (
            VideoFrameAnalysis(
                frame_index=10,
                timestamp_seconds=1.0,
                status="analyzed",
                deepfake_score=0.9,
                is_suspected_deepfake=True,
                face_similarity=1.0 if reference_payloads else None,
                quality=QUALITY,
                error_code=None,
                processing_ms=5.0,
                inference_ms=2.0,
            ),
            VideoFrameAnalysis(
                frame_index=20,
                timestamp_seconds=2.0,
                status="analyzed",
                deepfake_score=0.8,
                is_suspected_deepfake=True,
                face_similarity=0.9 if reference_payloads else 1.0,
                quality=QUALITY,
                error_code=None,
                processing_ms=5.0,
                inference_ms=2.0,
            ),
        )
        return DeepfakeVideoAnalysis(
            status="completed",
            is_suspected_deepfake=True,
            video_score=0.85,
            threshold=0.75,
            aggregation="mean",
            duration_seconds=3.0,
            fps=10.0,
            total_frame_count=30,
            requested_frame_count=2,
            decoded_frame_count=2,
            analyzed_frame_count=2,
            skipped_frame_count=0,
            reference_count=len(reference_payloads),
            frames=frames,
            suspicious_segments=(SuspiciousSegment(0.5, 2.5, 0.9, 2),),
            processing_ms=15.0,
            inference_ms=4.0,
            model_name="fake_deepfake_video_model",
            execution_provider="FakeDeepfakeExecutionProvider",
            model_fingerprint="b" * 64,
        )


def test_settings(*, license_accepted: bool = True) -> Settings:
    return Settings(
        accept_noncommercial_model_license=license_accepted,
        similarity_threshold=0.5,
        threshold_status="research_only_unapproved",
    )


def image_file(field: str, name: str, payload: bytes, content_type: str = "image/jpeg"):
    return (field, (name, payload, content_type))


class FaceguardHttpTests(unittest.TestCase):
    def setUp(self):
        self.video_analyzer = FakeVideoDeepfakeAnalyzer()
        self.client = TestClient(
            create_app(
                test_settings(),
                FakeEncoder(),
                deepfake_analyzer=FakeDeepfakeAnalyzer(),
                video_deepfake_analyzer=self.video_analyzer,
            )
        )

    def test_health_explains_model_and_threshold_state(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["model_loaded"])
        self.assertEqual(
            response.json()["threshold_status"], "research_only_unapproved"
        )
        self.assertEqual(response.json()["search_providers"], ["user_url"])
        self.assertFalse(response.json()["web_search_enabled"])
        self.assertTrue(response.json()["deepfake_model_loaded"])
        self.assertEqual(
            response.json()["deepfake_execution_provider"],
            "FakeDeepfakeExecutionProvider",
        )
        self.assertEqual(
            response.json()["deepfake_video_threshold_status"],
            "research_only_unapproved",
        )
        self.assertEqual(
            response.json()["deepfake_video_calibration_status"], "not_available"
        )
        self.assertIsNone(response.json()["deepfake_video_calibration_version"])

    def test_capabilities_explain_safe_client_contract(self):
        response = self.client.get("/v1/capabilities")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["api_version"], "0.9.0")
        self.assertEqual(body["deployment_mode"], "research_demo")
        self.assertIn("public_exposure_scan", body["workflows"])
        self.assertFalse(body["scores_are_probabilities"])
        self.assertFalse(body["automatic_enforcement_allowed"])
        self.assertFalse(body["original_media_persisted"])
        self.assertEqual(body["state_storage"], "process_memory_ttl")
        by_id = {item["component_id"]: item for item in body["models"]}
        self.assertEqual(
            by_id["face_verification"]["score_semantics"],
            "cosine_similarity",
        )
        self.assertEqual(by_id["face_verification"]["load_state"], "loaded")
        self.assertEqual(by_id["deepfake_image"]["load_state"], "loaded")
        self.assertEqual(
            by_id["deepfake_video"]["decision_status"],
            "research_only_unapproved",
        )

    def test_capabilities_block_models_until_license_is_confirmed(self):
        client = TestClient(
            create_app(
                test_settings(license_accepted=False),
                FakeEncoder(),
                deepfake_analyzer=FakeDeepfakeAnalyzer(),
            )
        )
        body = client.get("/v1/capabilities").json()
        self.assertTrue(all(item["load_state"] == "blocked" for item in body["models"]))
        self.assertTrue(all(not item["default_enabled"] for item in body["models"]))

    def test_deepfake_image_endpoint_returns_score_without_filename(self):
        response = self.client.post(
            "/v1/deepfake/analyze",
            files=[image_file("image", "private-candidate.jpg", b"candidate")],
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["is_suspected_deepfake"])
        self.assertEqual(body["deepfake_score"], 0.9)
        self.assertEqual(body["raw_score"], 0.9)
        self.assertIsNone(body["calibrated_probability"])
        self.assertEqual(body["calibration_status"], "not_applicable_single_image")
        self.assertEqual(body["decision_threshold"], body["threshold"])
        self.assertEqual(
            body["threshold_status"], "research_only_single_image_unvalidated"
        )
        self.assertEqual(body["config_version"], "deepfake-single-image-v1")
        self.assertNotIn("private-candidate", json.dumps(body))
        self.assertIn("단일 얼굴 이미지 점수", body["warning"])

    def test_deepfake_endpoint_reports_missing_private_model(self):
        settings = Settings(
            accept_noncommercial_model_license=True,
            deepfake_model_path=Path("/definitely/missing/model.onnx"),
        )
        client = TestClient(create_app(settings, FakeEncoder()))
        response = client.post(
            "/v1/deepfake/analyze",
            files=[image_file("image", "candidate.jpg", b"candidate")],
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "MODEL_UNAVAILABLE")

    def test_deepfake_endpoint_checks_model_license_first(self):
        client = TestClient(
            create_app(
                test_settings(license_accepted=False),
                FakeEncoder(),
                deepfake_analyzer=FakeDeepfakeAnalyzer(),
            )
        )
        response = client.post(
            "/v1/deepfake/analyze",
            files=[image_file("image", "candidate.jpg", b"candidate")],
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "MODEL_LICENSE_NOT_ACCEPTED")

    def test_deepfake_video_endpoint_returns_mean_score_and_segments(self):
        response = self.client.post(
            "/v1/deepfake/analyze-video",
            files=[
                image_file("video", "private-video.mp4", b"video-bytes", "video/mp4"),
                image_file("reference_images", "private-reference.jpg", b"reference"),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["is_suspected_deepfake"])
        self.assertEqual(body["video_score"], 0.85)
        self.assertEqual(body["raw_score"], 0.85)
        self.assertIsNone(body["calibrated_probability"])
        self.assertEqual(body["calibration_status"], "not_available")
        self.assertIsNone(body["risk_level"])
        self.assertEqual(body["decision_threshold"], body["threshold"])
        self.assertEqual(body["aggregation"], "mean")
        self.assertEqual(body["reference_count"], 1)
        self.assertEqual(body["analyzed_frame_count"], 2)
        self.assertEqual(len(body["suspicious_segments"]), 1)
        self.assertEqual(body["config_version"], "deepfake-video-16-frame-mean-v1")
        self.assertEqual(
            self.video_analyzer.calls,
            [(b"video-bytes", ".mp4", (b"reference",))],
        )
        serialized = json.dumps(body, ensure_ascii=False)
        self.assertNotIn("private-video", serialized)
        self.assertNotIn("private-reference", serialized)
        self.assertIn("오경고율", body["warning"])

    def test_deepfake_video_exposes_probability_only_for_approved_calibration(self):
        calibration = ScoreCalibration(
            version="test-approved-v1",
            scope="deepfake_video_mean_16_frames",
            method="platt",
            parameters={"slope": 0.5, "intercept": 0.0},
            model_fingerprint="b" * 64,
            low_threshold=0.2,
            high_threshold=0.8,
            review_band_empty=False,
            status="validated",
            display_approved=True,
            warning="내부 검증 Gate 통과",
        )
        client = TestClient(
            create_app(
                test_settings(),
                FakeEncoder(),
                deepfake_analyzer=FakeDeepfakeAnalyzer(),
                video_deepfake_analyzer=FakeVideoDeepfakeAnalyzer(),
                video_score_calibration=calibration,
            )
        )

        response = client.post(
            "/v1/deepfake/analyze-video",
            files=[image_file("video", "candidate.mp4", b"video", "video/mp4")],
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsNotNone(body["calibrated_probability"])
        self.assertEqual(body["calibration_status"], "validated")
        self.assertEqual(body["calibration_version"], "test-approved-v1")
        self.assertEqual(body["risk_level"], "high")

    def test_deepfake_video_rejects_unsupported_content_type(self):
        response = self.client.post(
            "/v1/deepfake/analyze-video",
            files=[image_file("video", "clip.avi", b"video", "video/x-msvideo")],
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response.json()["error"]["code"],
            "UNSUPPORTED_VIDEO_CONTENT_TYPE",
        )
        self.assertEqual(self.video_analyzer.calls, [])

    def test_deepfake_video_config_version_uses_configured_frame_count(self):
        settings = Settings(
            accept_noncommercial_model_license=True,
            similarity_threshold=0.5,
            deepfake_video_frame_count=4,
            deepfake_video_minimum_valid_frames=2,
        )
        client = TestClient(
            create_app(
                settings,
                FakeEncoder(),
                deepfake_analyzer=FakeDeepfakeAnalyzer(),
                video_deepfake_analyzer=FakeVideoDeepfakeAnalyzer(),
            )
        )
        response = client.post(
            "/v1/deepfake/analyze-video",
            files=[image_file("video", "clip.mp4", b"video", "video/mp4")],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["config_version"],
            "deepfake-video-4-frame-mean-v1",
        )

    def test_deepfake_video_total_request_limit_runs_before_multipart_parser(self):
        settings = Settings(
            accept_noncommercial_model_license=True,
            maximum_image_bytes=1,
            maximum_video_bytes=8,
            maximum_video_request_bytes=20,
        )
        client = TestClient(
            create_app(
                settings,
                FakeEncoder(),
                deepfake_analyzer=FakeDeepfakeAnalyzer(),
                video_deepfake_analyzer=FakeVideoDeepfakeAnalyzer(),
            )
        )
        response = client.post(
            "/v1/deepfake/analyze-video",
            content=b"x" * 21,
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "REQUEST_BODY_TOO_LARGE")

    def test_health_reports_configured_searxng_provider(self):
        settings = Settings(
            accept_noncommercial_model_license=True,
            similarity_threshold=0.5,
            searxng_base_url="http://searxng:8080",
        )
        client = TestClient(create_app(settings, FakeEncoder()))
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["search_providers"], ["user_url", "searxng"])
        self.assertTrue(response.json()["web_search_enabled"])

    def test_verify_returns_same_person_without_embedding_or_filename(self):
        response = self.client.post(
            "/v1/faceguard/verify",
            files=[
                image_file("reference_images", "private-one.jpg", b"same-1"),
                image_file("reference_images", "private-two.jpg", b"same-2"),
                image_file("reference_images", "private-three.jpg", b"same-3"),
                image_file("query_image", "private-query.jpg", b"same-query"),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["is_same_person"])
        self.assertAlmostEqual(body["similarity"], 1.0)
        self.assertEqual(body["raw_score"], body["similarity"])
        self.assertIsNone(body["calibrated_probability"])
        self.assertEqual(body["calibration_status"], "not_available")
        self.assertEqual(body["decision_threshold"], body["threshold"])
        self.assertEqual(body["reference_count"], 3)
        self.assertEqual(body["recommended_reference_count"], 3)
        self.assertEqual(body["execution_provider"], "FakeExecutionProvider")
        self.assertEqual(body["model_fingerprint"], "a" * 64)
        serialized = json.dumps(body)
        self.assertNotIn("embedding", serialized)
        self.assertNotIn("private-one", serialized)

    def test_verify_returns_different_person_candidate(self):
        response = self.client.post(
            "/v1/faceguard/verify",
            files=[
                image_file("reference_images", "ref.jpg", b"same"),
                image_file("query_image", "query.jpg", b"different"),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["is_same_person"])
        self.assertAlmostEqual(response.json()["similarity"], 0.0)

    def test_too_many_references_are_rejected_before_inference(self):
        files = [
            image_file("reference_images", f"ref-{index}.jpg", b"same")
            for index in range(6)
        ]
        files.append(image_file("query_image", "query.jpg", b"same"))
        response = self.client.post("/v1/faceguard/verify", files=files)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "TOO_MANY_REFERENCES")

    def test_unsupported_content_type_is_rejected(self):
        response = self.client.post(
            "/v1/faceguard/verify",
            files=[
                image_file("reference_images", "ref.heic", b"same", "image/heic"),
                image_file("query_image", "query.jpg", b"same"),
            ],
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["error"]["code"], "UNSUPPORTED_CONTENT_TYPE")

    def test_model_license_confirmation_is_required(self):
        client = TestClient(
            create_app(test_settings(license_accepted=False), FakeEncoder())
        )
        response = client.post(
            "/v1/faceguard/verify",
            files=[
                image_file("reference_images", "ref.jpg", b"same"),
                image_file("query_image", "query.jpg", b"same"),
            ],
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "MODEL_LICENSE_NOT_ACCEPTED")

    def test_model_input_error_has_stable_error_shape(self):
        response = self.client.post(
            "/v1/faceguard/verify",
            files=[
                image_file("reference_images", "ref.jpg", b"same"),
                image_file("query_image", "query.jpg", b"no-face"),
            ],
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {"error": {"code": "NO_FACE", "message": "얼굴을 찾지 못했습니다."}},
        )

    def test_missing_multipart_field_has_stable_error_shape(self):
        response = self.client.post(
            "/v1/faceguard/verify",
            files=[image_file("reference_images", "ref.jpg", b"same")],
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_free_search_mode_normalizes_and_deduplicates_user_urls(self):
        response = self.client.post(
            "/v1/search/candidates",
            json={
                "privacy_mode": "privacy_strict",
                "web_monitoring_consent": False,
                "candidates": [
                    {
                        "page_url": "https://Example.com/post?utm_source=demo",
                        "media_url": "https://cdn.example.com/photo.jpg",
                        "content_sha256": "a" * 64,
                    },
                    {
                        "page_url": "https://mirror.example.com/post",
                        "media_url": "https://cdn.example.com/photo-copy.jpg",
                        "content_sha256": "a" * 64,
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["privacy_mode"], "privacy_strict")
        self.assertEqual(body["raw_candidate_count"], 2)
        self.assertEqual(body["candidate_count"], 1)
        self.assertEqual(body["duplicate_count"], 1)
        self.assertEqual(body["truncated_count"], 0)
        self.assertEqual(body["candidates"][0]["provider"], "user_url")
        self.assertEqual(body["candidates"][0]["providers"], ["user_url"])
        self.assertNotIn("content_sha256", body["candidates"][0])
        self.assertIn("인터넷 자동 검색", body["warning"])

    def test_search_blocks_private_network_url(self):
        response = self.client.post(
            "/v1/search/candidates",
            json={"candidates": [{"page_url": "http://127.0.0.1/private"}]},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["error"]["code"], "PRIVATE_NETWORK_URL_BLOCKED"
        )

    def test_search_request_validation_has_search_specific_message(self):
        response = self.client.post(
            "/v1/search/candidates",
            json={"privacy_mode": "privacy_strict", "candidates": []},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
        self.assertIn("공개 후보 URL", response.json()["error"]["message"])

    def test_web_monitoring_mode_requires_consent(self):
        response = self.client.post(
            "/v1/search/candidates",
            json={
                "privacy_mode": "web_monitoring",
                "web_monitoring_consent": False,
                "query_text": "동의하지 않은 검색어",
                "candidates": [{"page_url": "https://example.com/post"}],
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["error"]["code"], "WEB_MONITORING_CONSENT_REQUIRED"
        )

    def test_web_monitoring_mode_requires_configured_external_provider(self):
        response = self.client.post(
            "/v1/search/candidates",
            json={
                "privacy_mode": "web_monitoring",
                "web_monitoring_consent": True,
                "query_text": "검색 제공자 확인",
                "candidates": [{"page_url": "https://example.com/post"}],
            },
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"], "SEARCH_PROVIDER_UNAVAILABLE"
        )

    def test_search_does_not_require_face_model_license(self):
        client = TestClient(
            create_app(test_settings(license_accepted=False), FakeEncoder())
        )
        response = client.post(
            "/v1/search/candidates",
            json={"candidates": [{"page_url": "https://example.com/post"}]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["candidate_count"], 1)

    def test_searxng_keyword_search_returns_normalized_candidate(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://example.com/public-post",
                            "img_src": "https://cdn.example.com/candidate.jpg",
                            "engine": "duckduckgo images",
                        }
                    ]
                },
            )

        provider = SearXNGProvider(
            "http://searxng:8080", transport=httpx.MockTransport(handler)
        )
        search_service = SearchService([UserSubmittedUrlProvider(), provider])
        client = TestClient(
            create_app(test_settings(), FakeEncoder(), search_service=search_service)
        )
        response = client.post(
            "/v1/search/candidates",
            json={
                "privacy_mode": "web_monitoring",
                "web_monitoring_consent": True,
                "query_text": "동의받은 이름 공개 이미지",
                "categories": ["images"],
                "maximum_results": 10,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["candidate_count"], 1)
        self.assertEqual(body["candidates"][0]["provider"], "searxng")
        self.assertEqual(body["candidates"][0]["source_engine"], "duckduckgo images")
        self.assertIn("역검색이 아닙니다", body["warning"])
        self.assertNotIn("query_text", body)
        self.assertNotIn("동의받은 이름", json.dumps(body, ensure_ascii=False))

    def test_web_monitoring_requires_query_text(self):
        response = self.client.post(
            "/v1/search/candidates",
            json={
                "privacy_mode": "web_monitoring",
                "web_monitoring_consent": True,
                "candidates": [{"page_url": "https://example.com/post"}],
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_search_and_filter_pipeline_returns_numeric_face_decisions(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://example.com/same-post",
                            "img_src": "https://cdn.example.com/same.jpg",
                            "engine": "test images",
                        },
                        {
                            "url": "https://example.com/different-post",
                            "img_src": "https://cdn.example.com/different.jpg",
                            "engine": "test images",
                        },
                    ]
                },
            )

        provider = SearXNGProvider(
            "http://searxng:8080", transport=httpx.MockTransport(handler)
        )
        search_service = SearchService([provider])
        client = TestClient(
            create_app(
                test_settings(),
                FakeEncoder(),
                search_service=search_service,
                image_downloader=FakeImageDownloader(),
                deepfake_analyzer=FakeDeepfakeAnalyzer(),
            )
        )
        response = client.post(
            "/v1/pipeline/search-and-filter",
            data={
                "query_text": "동의받은 비공개 테스트 검색어",
                "web_monitoring_consent": "true",
                "maximum_results": "2",
            },
            files=[
                image_file("reference_images", "private-ref-1.jpg", b"ref-1"),
                image_file("reference_images", "private-ref-2.jpg", b"ref-2"),
                image_file("reference_images", "private-ref-3.jpg", b"ref-3"),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["searched_candidate_count"], 2)
        self.assertEqual(body["analyzed_candidate_count"], 2)
        self.assertEqual(body["skipped_candidate_count"], 0)
        self.assertEqual(body["retrieval_match_count"], 1)
        self.assertEqual(body["identity_match_count"], 1)
        self.assertEqual(body["deepfake_analyzed_candidate_count"], 1)
        self.assertEqual(body["deepfake_suspected_candidate_count"], 1)
        self.assertEqual(body["deepfake_failed_candidate_count"], 0)
        self.assertEqual(
            [item["status"] for item in body["candidates"]],
            ["identity_match", "not_matched"],
        )
        self.assertEqual(body["candidates"][0]["matched_frame_count"], 1)
        self.assertEqual(body["candidates"][0]["analyzed_frame_count"], 1)
        self.assertIn("quality_summary", body["candidates"][0])
        self.assertEqual(body["candidates"][0]["deepfake"]["deepfake_score"], 0.9)
        self.assertEqual(body["candidates"][1]["deepfake"]["status"], "not_analyzed")
        self.assertEqual(
            body["candidates"][1]["deepfake"]["calibration_status"],
            "not_analyzed",
        )
        self.assertEqual(body["config_version"], "search-arcface-deepfake-image-v1")
        serialized = json.dumps(body, ensure_ascii=False)
        self.assertNotIn("동의받은 비공개 테스트 검색어", serialized)
        self.assertNotIn("private-ref", serialized)
        self.assertNotIn("embedding", serialized)

    def test_search_and_filter_checks_model_license_before_web_search(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"results": []})

        provider = SearXNGProvider(
            "http://searxng:8080", transport=httpx.MockTransport(handler)
        )
        client = TestClient(
            create_app(
                test_settings(license_accepted=False),
                FakeEncoder(),
                search_service=SearchService([provider]),
                image_downloader=FakeImageDownloader(),
            )
        )
        response = client.post(
            "/v1/pipeline/search-and-filter",
            data={
                "query_text": "외부로 나가면 안 되는 검색어",
                "web_monitoring_consent": "true",
            },
            files=[image_file("reference_images", "ref.jpg", b"ref")],
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "MODEL_LICENSE_NOT_ACCEPTED")
        self.assertEqual(calls, [])

    def test_search_and_filter_validates_reference_face_before_web_search(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"results": []})

        provider = SearXNGProvider(
            "http://searxng:8080", transport=httpx.MockTransport(handler)
        )
        client = TestClient(
            create_app(
                test_settings(),
                FakeEncoder(),
                search_service=SearchService([provider]),
                image_downloader=FakeImageDownloader(),
            )
        )
        response = client.post(
            "/v1/pipeline/search-and-filter",
            data={
                "query_text": "등록 사진 실패 시 전송하면 안 되는 검색어",
                "web_monitoring_consent": "true",
            },
            files=[image_file("reference_images", "ref.jpg", b"no-face")],
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "NO_FACE")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
