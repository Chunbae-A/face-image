import json
import unittest

import numpy as np
from fastapi.testclient import TestClient

from faceguard_api.app import create_app
from faceguard_api.domain import EncodedFace, FaceQuality
from faceguard_api.errors import FaceGuardError
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


if __name__ == "__main__":
    unittest.main()
