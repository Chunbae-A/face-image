import asyncio
import unittest

import httpx

from faceguard_api.media import CandidateDownloadError, PublicImageDownloader


async def public_resolver(hostname: str, port: int):
    del hostname, port
    return ("8.8.8.8",)


class SlowByteStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"1"
        await asyncio.sleep(0.05)
        yield b"2"


class PublicImageDownloaderTests(unittest.TestCase):
    def run_download(self, downloader: PublicImageDownloader, url: str):
        return asyncio.run(downloader.download(url))

    def test_downloads_supported_public_image(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["accept"], "image/jpeg,image/png,image/webp")
            self.assertEqual(str(request.url), "https://8.8.8.8/person.jpg")
            self.assertEqual(request.headers["host"], "cdn.example.com")
            self.assertEqual(request.extensions["sni_hostname"], "cdn.example.com")
            return httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=b"image-bytes",
            )

        downloader = PublicImageDownloader(
            maximum_bytes=1024,
            timeout_seconds=1.0,
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        result = self.run_download(
            downloader, "https://cdn.example.com/person.jpg?utm_source=test"
        )
        self.assertEqual(result.payload, b"image-bytes")
        self.assertEqual(result.content_type, "image/jpeg")
        self.assertEqual(result.source_url, "https://cdn.example.com/person.jpg")

    def test_blocks_hostname_resolving_to_private_address_before_request(self):
        requested = False

        async def private_resolver(hostname: str, port: int):
            del hostname, port
            return ("10.0.0.5",)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requested
            requested = True
            return httpx.Response(200, content=b"must-not-run")

        downloader = PublicImageDownloader(
            maximum_bytes=1024,
            timeout_seconds=1.0,
            resolver=private_resolver,
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(CandidateDownloadError) as raised:
            self.run_download(downloader, "https://internal.example.com/image.jpg")
        self.assertEqual(raised.exception.code, "PRIVATE_NETWORK_URL_BLOCKED")
        self.assertFalse(requested)

    def test_maps_dns_failure_to_stable_candidate_error(self):
        async def failing_resolver(hostname: str, port: int):
            del hostname, port
            raise OSError("private resolver detail")

        downloader = PublicImageDownloader(
            maximum_bytes=1024,
            timeout_seconds=1.0,
            resolver=failing_resolver,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"must-not-run")
            ),
        )
        with self.assertRaises(CandidateDownloadError) as raised:
            self.run_download(downloader, "https://missing.example.com/image.jpg")
        self.assertEqual(raised.exception.code, "CANDIDATE_DNS_FAILED")
        self.assertNotIn("private resolver detail", str(raised.exception))

    def test_revalidates_redirect_destination_and_blocks_private_address(self):
        calls = []

        async def resolver(hostname: str, port: int):
            del port
            return ("10.0.0.8",) if hostname.startswith("private") else ("8.8.8.8",)

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(
                302,
                headers={"location": "http://private.example.com/secret.jpg"},
            )

        downloader = PublicImageDownloader(
            maximum_bytes=1024,
            timeout_seconds=1.0,
            resolver=resolver,
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(CandidateDownloadError) as raised:
            self.run_download(downloader, "https://public.example.com/image.jpg")
        self.assertEqual(raised.exception.code, "PRIVATE_NETWORK_URL_BLOCKED")
        self.assertEqual(calls, ["https://8.8.8.8/image.jpg"])

    def test_rejects_non_image_content_type(self):
        downloader = PublicImageDownloader(
            maximum_bytes=1024,
            timeout_seconds=1.0,
            resolver=public_resolver,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    content=b"private body must not leak",
                )
            ),
        )
        with self.assertRaises(CandidateDownloadError) as raised:
            self.run_download(downloader, "https://example.com/page")
        self.assertEqual(
            raised.exception.code, "UNSUPPORTED_CANDIDATE_CONTENT_TYPE"
        )
        self.assertNotIn("private body", str(raised.exception))

    def test_stops_when_body_exceeds_size_limit(self):
        downloader = PublicImageDownloader(
            maximum_bytes=4,
            timeout_seconds=1.0,
            resolver=public_resolver,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "image/png"},
                    content=b"12345",
                )
            ),
        )
        with self.assertRaises(CandidateDownloadError) as raised:
            self.run_download(downloader, "https://cdn.example.com/large.png")
        self.assertEqual(raised.exception.code, "CANDIDATE_IMAGE_TOO_LARGE")

    def test_absolute_deadline_includes_delayed_stream_chunks(self):
        downloader = PublicImageDownloader(
            maximum_bytes=1024,
            timeout_seconds=0.01,
            resolver=public_resolver,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "image/jpeg"},
                    stream=SlowByteStream(),
                )
            ),
        )
        with self.assertRaises(CandidateDownloadError) as raised:
            self.run_download(downloader, "https://cdn.example.com/slow.jpg")
        self.assertEqual(raised.exception.code, "CANDIDATE_DOWNLOAD_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
