import asyncio
import unittest
from datetime import datetime, timezone

from faceguard_api.errors import FaceGuardError
from faceguard_api.search import (
    SearchCandidate,
    SearchQuery,
    SearchService,
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
