import json
import unittest

import httpx
import numpy as np
from fastapi.testclient import TestClient

from faceguard_api.app import create_app
from faceguard_api.domain import EncodedFace, FaceQuality
from faceguard_api.errors import FaceGuardError
from faceguard_api.search import (
    SearchService,
    SearXNGProvider,
    UserSubmittedUrlProvider,
)
from faceguard_api.settings import Settings

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
        self.client = TestClient(create_app(test_settings(), FakeEncoder()))

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
            json={
                "candidates": [{"page_url": "http://127.0.0.1/private"}]
            },
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
        self.assertEqual(
            body["candidates"][0]["source_engine"], "duckduckgo images"
        )
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


if __name__ == "__main__":
    unittest.main()
