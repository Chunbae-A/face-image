"""공개 후보 이미지를 내부망 접근 없이 제한적으로 내려받는다."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .errors import FaceGuardError
from .search import normalize_public_url

ALLOWED_IMAGE_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


class CandidateDownloadError(Exception):
    """후보별로 안전하게 집계할 수 있는 다운로드 실패 코드."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class DownloadedImage:
    payload: bytes
    source_url: str
    content_type: str


@dataclass(frozen=True)
class ResolvedPublicTarget:
    normalized_url: str
    connection_url: str
    host_header: str
    sni_hostname: str


Resolver = Callable[[str, int], Awaitable[Sequence[str]]]


async def _default_resolver(hostname: str, port: int) -> Sequence[str]:
    """시스템 DNS를 비동기로 조회하고 주소 문자열만 반환한다."""

    try:
        records = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise CandidateDownloadError("CANDIDATE_DNS_FAILED") from error
    return tuple(dict.fromkeys(record[4][0] for record in records))


class PublicImageDownloader:
    """후보 URL의 DNS·응답 형식·크기를 검사하고 메모리에서만 처리한다."""

    def __init__(
        self,
        *,
        maximum_bytes: int,
        timeout_seconds: float,
        maximum_redirects: int = 2,
        resolver: Resolver | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """다운로드 크기·전체 제한시간·리다이렉트 횟수를 설정한다."""

        if maximum_bytes <= 0 or timeout_seconds <= 0 or maximum_redirects < 0:
            raise ValueError("후보 이미지 다운로드 제한값이 올바르지 않습니다.")
        self.maximum_bytes = maximum_bytes
        self.timeout_seconds = timeout_seconds
        self.maximum_redirects = maximum_redirects
        self._resolver = resolver or _default_resolver
        self._transport = transport

    async def _validate_resolved_address(
        self, value: str
    ) -> ResolvedPublicTarget:
        """URL과 모든 DNS 결과를 검사하고 실제 접속할 공인 IP를 고정한다."""

        try:
            normalized = normalize_public_url(value)
        except FaceGuardError as error:
            raise CandidateDownloadError(error.code) from error
        parsed = urlsplit(normalized)
        assert parsed.hostname is not None
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            literal_address = ipaddress.ip_address(hostname)
        except ValueError:
            literal_address = None
        if literal_address is not None:
            addresses = (literal_address,)
        else:
            try:
                raw_addresses = await self._resolver(hostname, port)
            except CandidateDownloadError:
                raise
            except OSError as error:
                raise CandidateDownloadError("CANDIDATE_DNS_FAILED") from error
            try:
                addresses = tuple(ipaddress.ip_address(item) for item in raw_addresses)
            except ValueError as error:
                raise CandidateDownloadError("CANDIDATE_DNS_FAILED") from error
        if not addresses:
            raise CandidateDownloadError("CANDIDATE_DNS_FAILED")
        if any(not address.is_global for address in addresses):
            raise CandidateDownloadError("PRIVATE_NETWORK_URL_BLOCKED")
        selected_address = next(
            (address for address in addresses if address.version == 4), addresses[0]
        )
        rendered_address = (
            f"[{selected_address.compressed}]"
            if selected_address.version == 6
            else selected_address.compressed
        )
        default_port = (parsed.scheme == "http" and port == 80) or (
            parsed.scheme == "https" and port == 443
        )
        connection_netloc = (
            rendered_address if default_port else f"{rendered_address}:{port}"
        )
        connection_url = urlunsplit(
            (
                parsed.scheme,
                connection_netloc,
                parsed.path,
                parsed.query,
                "",
            )
        )
        return ResolvedPublicTarget(
            normalized_url=normalized,
            connection_url=connection_url,
            host_header=parsed.netloc,
            sni_hostname=hostname,
        )

    async def _download_with_deadline(self, url: str) -> DownloadedImage:
        """검증된 IP로 접속하며 각 리다이렉트 대상도 다시 검증한다."""

        current_url = url
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
            headers={
                "Accept": "image/jpeg,image/png,image/webp",
                "User-Agent": "DeepSogak-FaceGuard/0.4",
            },
        ) as client:
            for redirect_count in range(self.maximum_redirects + 1):
                target = await self._validate_resolved_address(current_url)
                try:
                    async with client.stream(
                        "GET",
                        target.connection_url,
                        headers={
                            "Host": target.host_header,
                            "Connection": "close",
                        },
                        extensions={"sni_hostname": target.sni_hostname},
                    ) as response:
                        if response.status_code in REDIRECT_STATUS_CODES:
                            location = response.headers.get("location")
                            if not location:
                                raise CandidateDownloadError(
                                    "CANDIDATE_REDIRECT_INVALID"
                                )
                            if redirect_count >= self.maximum_redirects:
                                raise CandidateDownloadError(
                                    "CANDIDATE_REDIRECT_LIMIT"
                                )
                            current_url = urljoin(target.normalized_url, location)
                            continue
                        response.raise_for_status()
                        content_type = (
                            response.headers.get("content-type", "")
                            .split(";", 1)[0]
                            .strip()
                            .lower()
                        )
                        if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
                            raise CandidateDownloadError(
                                "UNSUPPORTED_CANDIDATE_CONTENT_TYPE"
                            )
                        content_length = response.headers.get("content-length")
                        if content_length:
                            try:
                                declared_size = int(content_length)
                            except ValueError:
                                declared_size = 0
                            if declared_size > self.maximum_bytes:
                                raise CandidateDownloadError(
                                    "CANDIDATE_IMAGE_TOO_LARGE"
                                )
                        payload = bytearray()
                        async for chunk in response.aiter_bytes():
                            payload.extend(chunk)
                            if len(payload) > self.maximum_bytes:
                                raise CandidateDownloadError(
                                    "CANDIDATE_IMAGE_TOO_LARGE"
                                )
                        if not payload:
                            raise CandidateDownloadError("EMPTY_CANDIDATE_IMAGE")
                        return DownloadedImage(
                            payload=bytes(payload),
                            source_url=target.normalized_url,
                            content_type=content_type,
                        )
                except CandidateDownloadError:
                    raise
                except httpx.TimeoutException as error:
                    raise CandidateDownloadError("CANDIDATE_DOWNLOAD_TIMEOUT") from error
                except (httpx.HTTPStatusError, httpx.RequestError) as error:
                    raise CandidateDownloadError("CANDIDATE_DOWNLOAD_FAILED") from error
        raise CandidateDownloadError("CANDIDATE_REDIRECT_LIMIT")  # pragma: no cover

    async def download(self, url: str) -> DownloadedImage:
        """DNS부터 마지막 바이트까지 하나의 절대 제한시간으로 다운로드한다."""

        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._download_with_deadline(url)
        except CandidateDownloadError:
            raise
        except TimeoutError as error:
            raise CandidateDownloadError("CANDIDATE_DOWNLOAD_TIMEOUT") from error
