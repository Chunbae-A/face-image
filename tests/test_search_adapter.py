import asyncio
import unittest
from datetime import datetime, timezone
from urllib.parse import parse_qs

import httpx

from faceguard_api.errors import FaceGuardError
from faceguard_api.search import (
    SearchCandidate,
    SearchQuery,
    SearchService,
    SearXNGProvider,
    SubmittedCandidate,
    UserSubmittedUrlProvider,
    normalize_public_url,
)

FIXED_TIME = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


def query(
    *candidates: SubmittedCandidate,
    privacy_mode: str = "privacy_strict",
    consent: bool = False,
    text_query: str | None = None,
) -> SearchQuery:
    return SearchQuery(
        privacy_mode=privacy_mode,
        web_monitoring_consent=consent,
        submitted_candidates=list(candidates),
        text_query=text_query,
    )


def result_candidate(provider: str, rank: int = 1) -> SearchCandidate:
    return SearchCandidate(
        page_url=f"https://{provider}.example.com/post/{rank}",
        media_url=f"https://cdn.example.com/media/{rank}.jpg",
        thumbnail_url=None,
        provider=provider,
        providers=(provider,),
        rank=rank,
        retrieved_at=FIXED_TIME,
    )


class StaticProvider:
    accesses_external_network = False
    transmits_query_image = False

    def __init__(self, name: str, candidates=None):
        self.name = name
        self.candidates = candidates or [result_candidate(name)]
        self.called = False

    async def search(self, search_query):
        del search_query
        self.called = True
        return self.candidates

    def is_applicable(self, search_query):
        del search_query
        return True


class FailingProvider(StaticProvider):
    async def search(self, search_query):
        del search_query
        self.called = True
        raise RuntimeError("비밀 외부 응답은 노출되면 안 됨")


class SlowProvider(StaticProvider):
    async def search(self, search_query):
        del search_query
        self.called = True
        await asyncio.sleep(0.05)
        return self.candidates


class ExternalProvider(StaticProvider):
    accesses_external_network = True
    transmits_query_image = True


class PublicUrlNormalizationTests(unittest.TestCase):
    def test_normalizes_host_query_tracking_and_fragment(self):
        normalized = normalize_public_url(
            "HTTPS://Example.COM:443/path?utm_source=test&b=2&a=1#private-fragment"
        )
        self.assertEqual(normalized, "https://example.com/path?a=1&b=2")

    def test_rejects_private_local_and_credential_urls(self):
        cases = [
            ("http://127.0.0.1/admin", "PRIVATE_NETWORK_URL_BLOCKED"),
            ("http://192.168.1.10/file", "PRIVATE_NETWORK_URL_BLOCKED"),
            ("http://service.internal/file", "PRIVATE_NETWORK_URL_BLOCKED"),
            ("https://user:password@example.com/", "URL_CREDENTIALS_NOT_ALLOWED"),
            ("file:///tmp/private.jpg", "UNSAFE_PUBLIC_URL"),
            ("https://example.com:8443/file", "UNSAFE_PUBLIC_URL_PORT"),
        ]
        for value, code in cases:
            with self.subTest(value=value):
                with self.assertRaises(FaceGuardError) as raised:
                    normalize_public_url(value)
                self.assertEqual(raised.exception.code, code)

    def test_rejects_secret_query_parameter_without_echoing_value(self):
        with self.assertRaises(FaceGuardError) as raised:
            normalize_public_url("https://example.com/file?token=do-not-leak")
        self.assertEqual(raised.exception.code, "URL_SECRET_PARAMETER_NOT_ALLOWED")
        self.assertNotIn("do-not-leak", raised.exception.message)

    def test_rejects_control_characters(self):
        with self.assertRaises(FaceGuardError) as raised:
            normalize_public_url("https://example.com/path\nprivate")
        self.assertEqual(raised.exception.code, "INVALID_PUBLIC_URL")


class SearXNGProviderTests(unittest.TestCase):
    def run_provider(self, provider: SearXNGProvider, search_query: SearchQuery):
        return asyncio.run(provider.search(search_query))

    def test_maps_image_results_without_sending_a_face_image(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["form"] = parse_qs(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://news.example.com/post?utm_source=search",
                            "img_src": "https://cdn.example.com/person.jpg",
                            "thumbnail_src": "https://thumb.example.com/person.jpg",
                            "engines": ["duckduckgo images"],
                        }
                    ]
                },
            )

        provider = SearXNGProvider(
            "http://searxng:8080",
            transport=httpx.MockTransport(handler),
            clock=lambda: FIXED_TIME,
        )
        results = self.run_provider(
            provider,
            query(
                privacy_mode="web_monitoring",
                consent=True,
                text_query="동의받은 검색어",
            ),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_url, "https://news.example.com/post")
        self.assertEqual(results[0].media_url, "https://cdn.example.com/person.jpg")
        self.assertEqual(results[0].provider, "searxng")
        self.assertEqual(results[0].source_engine, "duckduckgo images")
        self.assertEqual(captured["form"]["q"], ["동의받은 검색어"])
        self.assertEqual(captured["form"]["categories"], ["images"])
        self.assertNotIn("image", captured["form"])
        self.assertNotIn("image_url", captured["form"])

    def test_skips_unsafe_result_urls_and_keeps_safe_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"url": "http://127.0.0.1/private", "img_src": "x"},
                        {
                            "url": "https://example.com/post",
                            "img_src": "http://192.168.0.2/private.jpg",
                            "thumbnail_src": "/relative-thumbnail",
                        },
                    ]
                },
            )

        provider = SearXNGProvider(
            "http://searxng:8080", transport=httpx.MockTransport(handler)
        )
        results = self.run_provider(
            provider,
            query(
                privacy_mode="web_monitoring",
                consent=True,
                text_query="public candidate",
            ),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_url, "https://example.com/post")
        self.assertIsNone(results[0].media_url)
        self.assertIsNone(results[0].thumbnail_url)

    def test_retries_retryable_status_then_succeeds(self):
        calls = []
        delays = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, text="temporary")
            return httpx.Response(200, json={"results": []})

        async def record_sleep(delay: float):
            delays.append(delay)

        provider = SearXNGProvider(
            "http://searxng:8080",
            maximum_retries=1,
            retry_backoff_seconds=0.1,
            transport=httpx.MockTransport(handler),
            sleep=record_sleep,
        )
        results = self.run_provider(
            provider,
            query(
                privacy_mode="web_monitoring",
                consent=True,
                text_query="retry query",
            ),
        )
        self.assertEqual(results, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(delays, [0.1])

    def test_invalid_json_becomes_provider_failure_without_body_leak(self):
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(200, content=b"private upstream body")

        provider = SearXNGProvider(
            "http://searxng:8080", transport=httpx.MockTransport(handler)
        )
        service = SearchService([provider])
        with self.assertRaises(FaceGuardError) as raised:
            asyncio.run(
                service.search(
                    query(
                        privacy_mode="web_monitoring",
                        consent=True,
                        text_query="query",
                    )
                )
            )
        self.assertEqual(raised.exception.code, "ALL_SEARCH_PROVIDERS_FAILED")
        self.assertNotIn("private upstream body", raised.exception.message)


class SearchServiceTests(unittest.TestCase):
    def run_search(self, service: SearchService, search_query: SearchQuery):
        return asyncio.run(service.search(search_query))

    def test_user_urls_are_normalized_and_deduplicated(self):
        provider = UserSubmittedUrlProvider(clock=lambda: FIXED_TIME)
        service = SearchService([provider])
        result = self.run_search(
            service,
            query(
                SubmittedCandidate(
                    page_url="https://example.com/post?utm_source=one"
                ),
                SubmittedCandidate(page_url="https://example.com/post"),
            ),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.raw_candidate_count, 2)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.truncated_count, 0)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].page_url, "https://example.com/post")
        self.assertEqual(result.candidates[0].retrieved_at, FIXED_TIME)

    def test_content_hash_deduplicates_different_urls(self):
        provider = UserSubmittedUrlProvider(clock=lambda: FIXED_TIME)
        service = SearchService([provider])
        shared_hash = "a" * 64
        result = self.run_search(
            service,
            query(
                SubmittedCandidate(
                    page_url="https://one.example.com/post",
                    content_sha256=shared_hash,
                ),
                SubmittedCandidate(
                    page_url="https://two.example.com/post",
                    content_sha256=shared_hash,
                ),
            ),
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.duplicate_count, 1)

    def test_one_provider_failure_returns_partial_success(self):
        success = StaticProvider("fixture")
        failure = FailingProvider("broken")
        service = SearchService([success, failure])
        result = self.run_search(
            service,
            query(SubmittedCandidate(page_url="https://example.com/post")),
        )
        self.assertEqual(result.status, "partial_failed")
        self.assertEqual(len(result.candidates), 1)
        failure_report = next(
            report for report in result.providers if report.provider == "broken"
        )
        self.assertEqual(failure_report.error_code, "SEARCH_PROVIDER_FAILED")

    def test_provider_timeout_is_reported_without_blocking_success(self):
        success = StaticProvider("fixture")
        slow = SlowProvider("slow")
        service = SearchService(
            [success, slow], provider_timeout_seconds=0.001
        )
        result = self.run_search(
            service,
            query(SubmittedCandidate(page_url="https://example.com/post")),
        )
        self.assertEqual(result.status, "partial_failed")
        slow_report = next(
            report for report in result.providers if report.provider == "slow"
        )
        self.assertEqual(slow_report.error_code, "SEARCH_PROVIDER_TIMEOUT")

    def test_all_provider_failures_return_stable_service_error(self):
        service = SearchService([FailingProvider("broken")])
        with self.assertRaises(FaceGuardError) as raised:
            self.run_search(
                service,
                query(SubmittedCandidate(page_url="https://example.com/post")),
            )
        self.assertEqual(raised.exception.code, "ALL_SEARCH_PROVIDERS_FAILED")
        self.assertEqual(raised.exception.http_status, 503)
        self.assertNotIn("비밀", raised.exception.message)

    def test_strict_mode_never_calls_external_image_provider(self):
        local = StaticProvider("user_url")
        external = ExternalProvider("external")
        service = SearchService([local, external])
        result = self.run_search(
            service,
            query(SubmittedCandidate(page_url="https://example.com/post")),
        )
        self.assertEqual(result.status, "completed")
        self.assertTrue(local.called)
        self.assertFalse(external.called)

    def test_web_monitoring_calls_external_text_provider_with_consent(self):
        external = ExternalProvider("external")
        service = SearchService([external])
        result = self.run_search(
            service,
            query(
                privacy_mode="web_monitoring",
                consent=True,
                text_query="consented query",
            ),
        )
        self.assertEqual(result.status, "completed")
        self.assertTrue(external.called)

    def test_web_monitoring_requires_explicit_consent(self):
        service = SearchService([ExternalProvider("external")])
        with self.assertRaises(FaceGuardError) as raised:
            self.run_search(
                service,
                query(
                    SubmittedCandidate(page_url="https://example.com/post"),
                    privacy_mode="web_monitoring",
                ),
            )
        self.assertEqual(raised.exception.code, "WEB_MONITORING_CONSENT_REQUIRED")

    def test_web_monitoring_requires_external_provider_configuration(self):
        service = SearchService([StaticProvider("user_url")])
        with self.assertRaises(FaceGuardError) as raised:
            self.run_search(
                service,
                query(
                    SubmittedCandidate(page_url="https://example.com/post"),
                    privacy_mode="web_monitoring",
                    consent=True,
                ),
            )
        self.assertEqual(raised.exception.code, "SEARCH_PROVIDER_UNAVAILABLE")
        self.assertEqual(raised.exception.http_status, 503)

    def test_candidate_limit_is_enforced_before_provider_call(self):
        provider = StaticProvider("user_url")
        service = SearchService([provider], maximum_candidates=1)
        with self.assertRaises(FaceGuardError) as raised:
            self.run_search(
                service,
                query(
                    SubmittedCandidate(page_url="https://one.example.com/post"),
                    SubmittedCandidate(page_url="https://two.example.com/post"),
                ),
            )
        self.assertEqual(raised.exception.code, "TOO_MANY_SEARCH_CANDIDATES")
        self.assertFalse(provider.called)


if __name__ == "__main__":
    unittest.main()
