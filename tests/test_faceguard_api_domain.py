import unittest

import numpy as np

from faceguard_api.domain import (
    cosine_similarity,
    l2_normalize,
    pool_reference_embeddings,
)
from faceguard_api.settings import Settings


class FaceguardDomainTests(unittest.TestCase):
    def test_l2_normalize_returns_unit_vector(self):
        normalized = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(normalized, np.array([0.6, 0.8]), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=6)

    def test_pooling_normalizes_each_reference_and_mean(self):
        pooled = pool_reference_embeddings([np.array([2.0, 0.0]), np.array([1.0, 1.0])])
        self.assertAlmostEqual(float(np.linalg.norm(pooled)), 1.0, places=6)
        self.assertGreater(pooled[0], pooled[1])

    def test_cosine_similarity_is_bounded(self):
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])),
            1.0,
        )
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            0.0,
        )

    def test_invalid_embeddings_are_rejected(self):
        with self.assertRaises(ValueError):
            l2_normalize(np.zeros(2))
        with self.assertRaises(ValueError):
            pool_reference_embeddings([])
        with self.assertRaises(ValueError):
            cosine_similarity(np.ones(2), np.ones(3))

    def test_settings_reject_invalid_operating_values(self):
        with self.assertRaises(ValueError):
            Settings(device="metal")
        with self.assertRaises(ValueError):
            Settings(similarity_threshold=2.0)
        with self.assertRaises(ValueError):
            Settings(
                retrieval_similarity_threshold=0.6,
                similarity_threshold=0.5,
            )
        with self.assertRaises(ValueError):
            Settings(minimum_face_area_ratio=0.0)
        with self.assertRaises(ValueError):
            Settings(maximum_search_candidates=0)
        with self.assertRaises(ValueError):
            Settings(search_provider_timeout_seconds=0.0)
        with self.assertRaises(ValueError):
            Settings(searxng_base_url="file:///tmp/searxng")
        with self.assertRaises(ValueError):
            Settings(searxng_base_url="https://user:password@example.com")
        with self.assertRaises(ValueError):
            Settings(searxng_request_timeout_seconds=0.0)
        with self.assertRaises(ValueError):
            Settings(searxng_maximum_retries=-1)
        with self.assertRaises(ValueError):
            Settings(searxng_retry_backoff_seconds=-0.1)
        with self.assertRaises(ValueError):
            Settings(maximum_pipeline_candidates=0)
        with self.assertRaises(ValueError):
            Settings(candidate_download_timeout_seconds=0.0)
        with self.assertRaises(ValueError):
            Settings(candidate_download_maximum_redirects=-1)
        with self.assertRaises(ValueError):
            Settings(deepfake_device="metal")
        with self.assertRaises(ValueError):
            Settings(deepfake_input_size=0)
        with self.assertRaises(ValueError):
            Settings(deepfake_threshold=1.1)
        with self.assertRaises(ValueError):
            Settings(deepfake_model_sha256="not-a-sha")


if __name__ == "__main__":
    unittest.main()
