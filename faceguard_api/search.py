"""공개 웹 후보를 안전한 공통 형식으로 정규화하는 검색 어댑터."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Literal, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from .errors import FaceGuardError

PrivacyMode = Literal["privacy_strict", "web_monitoring"]

TRACKING_QUERY_KEYS = {
    "dclid",
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}
SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "key",
    "password",
    "secret",
    "session",
    "sig",
    "signature",
    "token",
}
BLOCKED_HOST_SUFFIXES = (".internal", ".local", ".localhost", ".home.arpa")


@dataclass(frozen=True)
class SubmittedCandidate:
    """사용자가 직접 제보한 공개 페이지와 선택적 미디어 URL."""

    page_url: str
    media_url: str | None = None
    thumbnail_url: str | None = None
    content_sha256: str | None = None
    perceptual_hash: str | None = None


@dataclass(frozen=True)
class SearchQuery:
    privacy_mode: PrivacyMode
    web_monitoring_consent: bool
    submitted_candidates: Sequence[SubmittedCandidate]
    text_query: str | None = None
    categories: tuple[str, ...] = ("images",)
    language: str = "ko-KR"
    safe_search: int = 2
    maximum_results: int = 20


@dataclass(frozen=True)
class SearchCandidate:
    page_url: str
    media_url: str | None
    thumbnail_url: str | None
    provider: str
    providers: tuple[str, ...]
    rank: int
    retrieved_at: datetime
    source_engine: str | None = None
    content_sha256: str | None = None
    perceptual_hash: str | None = None


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    status: Literal["completed", "failed"]
    candidate_count: int
    processing_ms: float
    error_code: str | None = None


@dataclass(frozen=True)
class SearchResult:
    status: Literal["completed", "partial_failed"]
    privacy_mode: PrivacyMode
    candidates: tuple[SearchCandidate, ...]
    providers: tuple[ProviderResult, ...]
    raw_candidate_count: int
    duplicate_count: int
    truncated_count: int
    processing_ms: float


class SearchProvider(Protocol):
    """외부 검색 제공자와 사용자 URL 제공자가 지켜야 할 공통 계약."""

    name: str
    accesses_external_network: bool
    transmits_query_image: bool

    def is_applicable(self, query: SearchQuery) -> bool: ...

    async def search(self, query: SearchQuery) -> Sequence[SearchCandidate]: ...


def _normalized_hash(value: str | None, *, name: str, length: int) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise FaceGuardError(
            "INVALID_CANDIDATE_HASH",
            f"{name} 값은 {length}자리 16진수여야 합니다.",
        )
    return normalized


def normalize_public_url(value: str) -> str:
    """공개 HTTP(S) URL만 허용하고 추적·비밀 쿼리를 제거하거나 거절한다."""

    raw = value.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise FaceGuardError(
            "INVALID_PUBLIC_URL", "제어 문자가 포함된 URL은 사용할 수 없습니다."
        )
    if not raw or len(raw) > 2048:
        raise FaceGuardError(
            "INVALID_PUBLIC_URL", "공개 URL은 1~2,048자여야 합니다."
        )
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as error:
        raise FaceGuardError(
            "INVALID_PUBLIC_URL", "공개 URL 형식이 올바르지 않습니다."
        ) from error

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise FaceGuardError(
            "UNSAFE_PUBLIC_URL", "공개 URL은 http 또는 https만 사용할 수 있습니다."
        )
    if parsed.username is not None or parsed.password is not None:
        raise FaceGuardError(
            "URL_CREDENTIALS_NOT_ALLOWED",
            "아이디·비밀번호가 포함된 URL은 사용할 수 없습니다.",
        )
    if parsed.hostname is None:
        raise FaceGuardError(
            "INVALID_PUBLIC_URL", "공개 URL에는 호스트 이름이 필요합니다."
        )

    try:
        hostname = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise FaceGuardError(
            "INVALID_PUBLIC_URL", "공개 URL의 호스트 이름이 올바르지 않습니다."
        ) from error
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is None and (
        hostname == "localhost"
        or hostname.endswith(BLOCKED_HOST_SUFFIXES)
        or "." not in hostname
    ):
        raise FaceGuardError(
            "PRIVATE_NETWORK_URL_BLOCKED",
            "내부망·로컬 주소는 공개 후보로 사용할 수 없습니다.",
        )
    if address is not None and not address.is_global:
        raise FaceGuardError(
            "PRIVATE_NETWORK_URL_BLOCKED",
            "내부망·로컬 주소는 공개 후보로 사용할 수 없습니다.",
        )
    if port is not None and port not in {80, 443}:
        raise FaceGuardError(
            "UNSAFE_PUBLIC_URL_PORT",
            "공개 URL은 기본 HTTP·HTTPS 포트만 사용할 수 있습니다.",
        )

    query_items: list[tuple[str, str]] = []
    for key, item_value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key in SENSITIVE_QUERY_KEYS:
            raise FaceGuardError(
                "URL_SECRET_PARAMETER_NOT_ALLOWED",
                "비밀값으로 보이는 쿼리 파라미터가 포함된 URL은 사용할 수 없습니다.",
            )
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, item_value))
    query_items.sort(key=lambda item: (item[0], item[1]))

    default_port = (scheme == "http" and port in {None, 80}) or (
        scheme == "https" and port in {None, 443}
    )
    if ":" in hostname:
        rendered_host = f"[{hostname}]"
    else:
        rendered_host = hostname
    netloc = rendered_host if default_port else f"{rendered_host}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, urlencode(query_items, doseq=True), ""))


class UserSubmittedUrlProvider:
    """외부 이미지 전송 없이 사용자가 제보한 공개 URL만 후보로 만든다."""

    name = "user_url"
    accesses_external_network = False
    transmits_query_image = False

    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def is_applicable(self, query: SearchQuery) -> bool:
        return bool(query.submitted_candidates)

    async def search(self, query: SearchQuery) -> Sequence[SearchCandidate]:
        retrieved_at = self._clock()
        candidates: list[SearchCandidate] = []
        for rank, submitted in enumerate(query.submitted_candidates, start=1):
            candidates.append(
                SearchCandidate(
                    page_url=normalize_public_url(submitted.page_url),
                    media_url=(
                        normalize_public_url(submitted.media_url)
                        if submitted.media_url
                        else None
                    ),
                    thumbnail_url=(
                        normalize_public_url(submitted.thumbnail_url)
                        if submitted.thumbnail_url
                        else None
                    ),
                    provider=self.name,
                    providers=(self.name,),
                    rank=rank,
                    retrieved_at=retrieved_at,
                    content_sha256=_normalized_hash(
                        submitted.content_sha256,
                        name="content_sha256",
                        length=64,
                    ),
                    perceptual_hash=_normalized_hash(
                        submitted.perceptual_hash,
                        name="perceptual_hash",
                        length=16,
                    ),
                )
            )
        return candidates


class SearXNGProvider:
    """자체 호스팅 SearXNG에 검색어만 보내 이미지·영상 후보를 가져온다."""

    name = "searxng"
    accesses_external_network = True
    transmits_query_image = False
    retryable_status_codes = frozenset({429, 502, 503, 504})

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float = 5.0,
        maximum_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[object]] | None = None,
    ) -> None:
        parsed = urlsplit(base_url.strip())
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("SearXNG 주소는 인증정보·쿼리 없는 HTTP(S) URL이어야 합니다.")
        if request_timeout_seconds <= 0 or maximum_retries < 0:
            raise ValueError("SearXNG timeout은 양수이고 재시도 횟수는 0 이상이어야 합니다.")
        if retry_backoff_seconds < 0:
            raise ValueError("SearXNG 재시도 대기시간은 0 이상이어야 합니다.")
        self.search_url = f"{base_url.rstrip('/')}/search"
        self.request_timeout_seconds = request_timeout_seconds
        self.maximum_retries = maximum_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or asyncio.sleep

    def is_applicable(self, query: SearchQuery) -> bool:
        return bool(query.text_query and query.text_query.strip())

    async def _request(self, form: dict[str, str]) -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=self.request_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
        ) as client:
            last_error: Exception | None = None
            for attempt in range(self.maximum_retries + 1):
                try:
                    response = await client.post(self.search_url, data=form)
                    if response.status_code not in self.retryable_status_codes:
                        response.raise_for_status()
                        return response
                    last_error = httpx.HTTPStatusError(
                        "SearXNG retryable response",
                        request=response.request,
                        response=response,
                    )
                except httpx.RequestError as error:
                    last_error = error
                if attempt < self.maximum_retries:
                    delay = self.retry_backoff_seconds * (2**attempt)
                    await self._sleep(delay)
            if last_error is None:  # pragma: no cover - 방어적 분기
                raise RuntimeError("SearXNG request failed")
            raise last_error

    @staticmethod
    def _optional_public_url(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return normalize_public_url(value)
        except FaceGuardError:
            return None

    async def search(self, query: SearchQuery) -> Sequence[SearchCandidate]:
        if not self.is_applicable(query):
            return []
        assert query.text_query is not None
        form = {
            "q": query.text_query.strip(),
            "categories": ",".join(query.categories),
            "language": query.language,
            "safesearch": str(query.safe_search),
            "format": "json",
        }
        response = await self._request(form)
        try:
            body = response.json()
        except ValueError as error:
            raise RuntimeError("SearXNG returned invalid JSON") from error
        raw_results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(raw_results, list):
            raise TypeError("SearXNG results are missing")

        retrieved_at = self._clock()
        candidates: list[SearchCandidate] = []
        for rank, raw in enumerate(raw_results, start=1):
            if rank > query.maximum_results:
                break
            if not isinstance(raw, dict):
                continue
            page_url = self._optional_public_url(raw.get("url"))
            if page_url is None:
                continue
            media_url = self._optional_public_url(
                raw.get("img_src") or raw.get("iframe_src")
            )
            thumbnail_url = self._optional_public_url(
                raw.get("thumbnail_src") or raw.get("thumbnail")
            )
            source_engine = raw.get("engine")
            if not isinstance(source_engine, str) or not source_engine:
                engines = raw.get("engines")
                source_engine = (
                    engines[0]
                    if isinstance(engines, list)
                    and engines
                    and isinstance(engines[0], str)
                    else None
                )
            candidates.append(
                SearchCandidate(
                    page_url=page_url,
                    media_url=media_url,
                    thumbnail_url=thumbnail_url,
                    provider=self.name,
                    providers=(self.name,),
                    rank=rank,
                    retrieved_at=retrieved_at,
                    source_engine=source_engine,
                )
            )
        return candidates


def _candidate_key(candidate: SearchCandidate) -> str:
    if candidate.content_sha256:
        return f"sha256:{candidate.content_sha256}"
    if candidate.perceptual_hash:
        return f"phash:{candidate.perceptual_hash}"
    primary_url = candidate.media_url or candidate.page_url
    digest = hashlib.sha256(primary_url.encode("utf-8")).hexdigest()
    return f"url_sha256:{digest}"


def _deduplicate(
    candidates: Sequence[SearchCandidate],
) -> tuple[list[SearchCandidate], int]:
    unique: dict[str, SearchCandidate] = {}
    duplicates = 0
    for candidate in candidates:
        key = _candidate_key(candidate)
        existing = unique.get(key)
        if existing is None:
            unique[key] = candidate
            continue
        duplicates += 1
        providers = tuple(dict.fromkeys((*existing.providers, *candidate.providers)))
        if (candidate.rank, candidate.provider) < (existing.rank, existing.provider):
            unique[key] = replace(candidate, providers=providers)
        else:
            unique[key] = replace(existing, providers=providers)
    ordered = sorted(
        unique.values(), key=lambda item: (item.rank, item.provider, item.page_url)
    )
    return ordered, duplicates


class SearchService:
    """검색 제공자를 실행하고 부분 실패와 중복 제거를 일관되게 처리한다."""

    def __init__(
        self,
        providers: Sequence[SearchProvider],
        *,
        maximum_candidates: int = 100,
        provider_timeout_seconds: float = 5.0,
    ) -> None:
        if not providers:
            raise ValueError("검색 제공자가 하나 이상 필요합니다.")
        if maximum_candidates <= 0 or provider_timeout_seconds <= 0:
            raise ValueError("검색 제한값은 양수여야 합니다.")
        names = [provider.name for provider in providers]
        if len(names) != len(set(names)):
            raise ValueError("검색 제공자 이름은 중복될 수 없습니다.")
        self.providers = tuple(providers)
        self.maximum_candidates = maximum_candidates
        self.provider_timeout_seconds = provider_timeout_seconds

    async def _invoke(
        self, provider: SearchProvider, query: SearchQuery
    ) -> tuple[Sequence[SearchCandidate] | None, ProviderResult]:
        started = time.perf_counter()
        try:
            candidates = await asyncio.wait_for(
                provider.search(query), timeout=self.provider_timeout_seconds
            )
        except FaceGuardError:
            raise
        except TimeoutError:
            return None, ProviderResult(
                provider=provider.name,
                status="failed",
                candidate_count=0,
                processing_ms=(time.perf_counter() - started) * 1000.0,
                error_code="SEARCH_PROVIDER_TIMEOUT",
            )
        except Exception:  # noqa: BLE001 - 제공자 예외와 응답 본문을 API에 노출하지 않는다.
            return None, ProviderResult(
                provider=provider.name,
                status="failed",
                candidate_count=0,
                processing_ms=(time.perf_counter() - started) * 1000.0,
                error_code="SEARCH_PROVIDER_FAILED",
            )
        return candidates, ProviderResult(
            provider=provider.name,
            status="completed",
            candidate_count=len(candidates),
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def search(self, query: SearchQuery) -> SearchResult:
        if len(query.submitted_candidates) > self.maximum_candidates:
            raise FaceGuardError(
                "TOO_MANY_SEARCH_CANDIDATES",
                f"공개 후보는 한 번에 최대 {self.maximum_candidates}개까지 처리할 수 있습니다.",
            )
        if query.privacy_mode == "web_monitoring" and not query.web_monitoring_consent:
            raise FaceGuardError(
                "WEB_MONITORING_CONSENT_REQUIRED",
                "외부 웹 검색을 사용하려면 검색어 또는 검색용 이미지 전송 동의가 필요합니다.",
            )
        if query.privacy_mode == "privacy_strict" and query.text_query:
            raise FaceGuardError(
                "WEB_MONITORING_MODE_REQUIRED",
                "외부 검색어 조회는 web_monitoring 모드에서만 사용할 수 있습니다.",
            )
        if query.privacy_mode == "web_monitoring" and not any(
            provider.accesses_external_network for provider in self.providers
        ):
            raise FaceGuardError(
                "SEARCH_PROVIDER_UNAVAILABLE",
                "외부 웹 검색 제공자가 설정되지 않았습니다. 무료 URL 제보 모드를 사용하세요.",
                503,
            )

        eligible = [
            provider
            for provider in self.providers
            if provider.is_applicable(query)
            and (
                query.privacy_mode == "web_monitoring"
                or not provider.accesses_external_network
            )
        ]
        if not eligible:
            raise FaceGuardError(
                "SEARCH_PROVIDER_UNAVAILABLE",
                "선택한 개인정보 모드에서 사용할 수 있는 검색 제공자가 없습니다.",
                503,
            )

        started = time.perf_counter()
        outcomes = await asyncio.gather(
            *(self._invoke(provider, query) for provider in eligible)
        )
        raw_candidates = [
            candidate
            for candidates, _ in outcomes
            if candidates is not None
            for candidate in candidates
        ]
        provider_results = tuple(result for _, result in outcomes)
        failed_count = sum(result.status == "failed" for result in provider_results)
        if failed_count == len(provider_results):
            raise FaceGuardError(
                "ALL_SEARCH_PROVIDERS_FAILED",
                "모든 검색 제공자 호출에 실패했습니다. 잠시 후 다시 시도하세요.",
                503,
            )

        unique, duplicate_count = _deduplicate(raw_candidates)
        limited = tuple(unique[: self.maximum_candidates])
        truncated_count = max(0, len(unique) - len(limited))
        return SearchResult(
            status="partial_failed" if failed_count else "completed",
            privacy_mode=query.privacy_mode,
            candidates=limited,
            providers=provider_results,
            raw_candidate_count=len(raw_candidates),
            duplicate_count=duplicate_count,
            truncated_count=truncated_count,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )
