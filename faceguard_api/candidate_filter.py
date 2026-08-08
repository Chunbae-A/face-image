"""검색 후보 이미지를 등록 얼굴과 비교하는 ArcFace 배치 필터."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .deepfake import DeepfakeAnalysis
from .domain import (
    EncodedFace,
    FaceQuality,
    cosine_similarity,
    pool_reference_embeddings,
)
from .errors import FaceGuardError
from .media import CandidateDownloadError, DownloadedImage
from .search import SearchCandidate
from .service import FaceEncoder
from .settings import Settings


class ImageDownloader(Protocol):
    async def download(self, url: str) -> DownloadedImage: ...


class DeepfakeImageAnalyzer(Protocol):
    def analyze(self, payload: bytes) -> DeepfakeAnalysis: ...


@dataclass(frozen=True)
class CandidateDeepfakeDecision:
    status: Literal["analyzed", "not_analyzed", "failed", "unavailable"]
    deepfake_score: float | None
    is_suspected_deepfake: bool | None
    error_code: str | None
    processing_ms: float
    inference_ms: float | None
    execution_provider: str | None
    model_fingerprint: str | None


@dataclass(frozen=True)
class CandidateFaceDecision:
    page_url: str
    media_url: str | None
    thumbnail_url: str | None
    provider: str
    source_engine: str | None
    status: Literal["identity_match", "retrieval_match", "not_matched", "skipped"]
    similarity_raw: float | None
    retrieval_match: bool | None
    identity_match: bool | None
    analyzed_url: str | None
    error_code: str | None
    quality: FaceQuality | None
    deepfake: CandidateDeepfakeDecision
    processing_ms: float


@dataclass(frozen=True)
class CandidateFilterResult:
    candidates: tuple[CandidateFaceDecision, ...]
    reference_count: int
    analyzed_candidate_count: int
    skipped_candidate_count: int
    retrieval_match_count: int
    identity_match_count: int
    deepfake_analyzed_candidate_count: int
    deepfake_suspected_candidate_count: int
    deepfake_failed_candidate_count: int
    processing_ms: float


@dataclass(frozen=True)
class PreparedReferences:
    faces: Sequence[EncodedFace]
    embedding: np.ndarray
    processing_ms: float


class CandidateFilterService:
    def __init__(
        self,
        settings: Settings,
        encoder: FaceEncoder,
        downloader: ImageDownloader,
        deepfake_analyzer: DeepfakeImageAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.encoder = encoder
        self.downloader = downloader
        self.deepfake_analyzer = deepfake_analyzer

    @staticmethod
    def _deepfake_not_analyzed() -> CandidateDeepfakeDecision:
        return CandidateDeepfakeDecision(
            status="not_analyzed",
            deepfake_score=None,
            is_suspected_deepfake=None,
            error_code=None,
            processing_ms=0.0,
            inference_ms=None,
            execution_provider=None,
            model_fingerprint=None,
        )

    async def _analyze_deepfake(self, payload: bytes) -> CandidateDeepfakeDecision:
        if self.deepfake_analyzer is None:
            return CandidateDeepfakeDecision(
                status="unavailable",
                deepfake_score=None,
                is_suspected_deepfake=None,
                error_code="DEEPFAKE_ANALYZER_UNAVAILABLE",
                processing_ms=0.0,
                inference_ms=None,
                execution_provider=None,
                model_fingerprint=None,
            )
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(self.deepfake_analyzer.analyze, payload)
            return CandidateDeepfakeDecision(
                status="analyzed",
                deepfake_score=result.deepfake_score,
                is_suspected_deepfake=result.is_suspected_deepfake,
                error_code=None,
                processing_ms=result.processing_ms,
                inference_ms=result.inference_ms,
                execution_provider=result.execution_provider,
                model_fingerprint=result.model_fingerprint,
            )
        except FaceGuardError as error:
            return CandidateDeepfakeDecision(
                status=("unavailable" if error.code == "MODEL_UNAVAILABLE" else "failed"),
                deepfake_score=None,
                is_suspected_deepfake=None,
                error_code=error.code,
                processing_ms=(time.perf_counter() - started) * 1000.0,
                inference_ms=None,
                execution_provider=None,
                model_fingerprint=None,
            )

    def _encode_references(self, references: Sequence[bytes]) -> PreparedReferences:
        started = time.perf_counter()
        reference_faces = [self.encoder.encode(payload) for payload in references]
        reference_embedding = pool_reference_embeddings(
            [face.embedding for face in reference_faces]
        )
        return PreparedReferences(
            faces=reference_faces,
            embedding=reference_embedding,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _source_urls(candidate: SearchCandidate) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                url
                for url in (
                    candidate.media_url,
                    candidate.thumbnail_url,
                    candidate.page_url,
                )
                if url
            )
        )

    async def _download_candidate(
        self, candidate: SearchCandidate
    ) -> DownloadedImage:
        last_error: CandidateDownloadError | None = None
        for url in self._source_urls(candidate):
            try:
                return await self.downloader.download(url)
            except CandidateDownloadError as error:
                last_error = error
        if last_error is not None:
            raise last_error
        raise CandidateDownloadError("NO_DOWNLOADABLE_IMAGE_URL")

    async def prepare_references(
        self, references: Sequence[bytes]
    ) -> PreparedReferences:
        if len(references) < self.settings.minimum_reference_images:
            raise FaceGuardError(
                "TOO_FEW_REFERENCES",
                f"등록 사진이 최소 {self.settings.minimum_reference_images}장 필요합니다.",
            )
        if len(references) > self.settings.maximum_reference_images:
            raise FaceGuardError(
                "TOO_MANY_REFERENCES",
                f"등록 사진은 최대 {self.settings.maximum_reference_images}장까지 사용할 수 있습니다.",
            )
        return await asyncio.to_thread(self._encode_references, references)

    async def filter_prepared(
        self,
        references: PreparedReferences,
        candidates: Sequence[SearchCandidate],
        *,
        deepfake_stage_callback: Callable[[], Awaitable[None]] | None = None,
        progress_callback: Callable[
            [tuple[CandidateFaceDecision, ...]], Awaitable[None]
        ]
        | None = None,
    ) -> CandidateFilterResult:
        if len(candidates) > self.settings.maximum_pipeline_candidates:
            raise FaceGuardError(
                "TOO_MANY_PIPELINE_CANDIDATES",
                f"얼굴 비교 후보는 최대 {self.settings.maximum_pipeline_candidates}개까지 처리할 수 있습니다.",
            )

        started = time.perf_counter()
        decisions: list[CandidateFaceDecision] = []
        deepfake_stage_started = False
        for candidate in candidates:
            candidate_started = time.perf_counter()
            try:
                downloaded = await self._download_candidate(candidate)
                query_face = await asyncio.to_thread(
                    self.encoder.encode, downloaded.payload
                )
                similarity = cosine_similarity(
                    references.embedding, query_face.embedding
                )
                retrieval_match = (
                    similarity >= self.settings.retrieval_similarity_threshold
                )
                identity_match = similarity >= self.settings.similarity_threshold
                if (
                    retrieval_match
                    and deepfake_stage_callback is not None
                    and not deepfake_stage_started
                ):
                    await deepfake_stage_callback()
                    deepfake_stage_started = True
                deepfake = (
                    await self._analyze_deepfake(downloaded.payload)
                    if retrieval_match
                    else self._deepfake_not_analyzed()
                )
                if identity_match:
                    status = "identity_match"
                elif retrieval_match:
                    status = "retrieval_match"
                else:
                    status = "not_matched"
                decisions.append(
                    CandidateFaceDecision(
                        page_url=candidate.page_url,
                        media_url=candidate.media_url,
                        thumbnail_url=candidate.thumbnail_url,
                        provider=candidate.provider,
                        source_engine=candidate.source_engine,
                        status=status,
                        similarity_raw=similarity,
                        retrieval_match=retrieval_match,
                        identity_match=identity_match,
                        analyzed_url=downloaded.source_url,
                        error_code=None,
                        quality=query_face.quality,
                        deepfake=deepfake,
                        processing_ms=(time.perf_counter() - candidate_started)
                        * 1000.0,
                    )
                )
            except CandidateDownloadError as error:
                decisions.append(
                    CandidateFaceDecision(
                        page_url=candidate.page_url,
                        media_url=candidate.media_url,
                        thumbnail_url=candidate.thumbnail_url,
                        provider=candidate.provider,
                        source_engine=candidate.source_engine,
                        status="skipped",
                        similarity_raw=None,
                        retrieval_match=None,
                        identity_match=None,
                        analyzed_url=None,
                        error_code=error.code,
                        quality=None,
                        deepfake=self._deepfake_not_analyzed(),
                        processing_ms=(time.perf_counter() - candidate_started)
                        * 1000.0,
                    )
                )
            except FaceGuardError as error:
                if error.code == "MODEL_UNAVAILABLE":
                    raise
                decisions.append(
                    CandidateFaceDecision(
                        page_url=candidate.page_url,
                        media_url=candidate.media_url,
                        thumbnail_url=candidate.thumbnail_url,
                        provider=candidate.provider,
                        source_engine=candidate.source_engine,
                        status="skipped",
                        similarity_raw=None,
                        retrieval_match=None,
                        identity_match=None,
                        analyzed_url=None,
                        error_code=error.code,
                        quality=None,
                        deepfake=self._deepfake_not_analyzed(),
                        processing_ms=(time.perf_counter() - candidate_started)
                        * 1000.0,
                    )
                )
            if progress_callback is not None:
                await progress_callback(tuple(decisions))

        analyzed_count = sum(item.status != "skipped" for item in decisions)
        skipped_count = len(decisions) - analyzed_count
        return CandidateFilterResult(
            candidates=tuple(decisions),
            reference_count=len(references.faces),
            analyzed_candidate_count=analyzed_count,
            skipped_candidate_count=skipped_count,
            retrieval_match_count=sum(
                item.retrieval_match is True for item in decisions
            ),
            identity_match_count=sum(item.identity_match is True for item in decisions),
            deepfake_analyzed_candidate_count=sum(
                item.deepfake.status == "analyzed" for item in decisions
            ),
            deepfake_suspected_candidate_count=sum(
                item.deepfake.is_suspected_deepfake is True for item in decisions
            ),
            deepfake_failed_candidate_count=sum(
                item.deepfake.status in {"failed", "unavailable"}
                for item in decisions
            ),
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def filter(
        self,
        references: Sequence[bytes],
        candidates: Sequence[SearchCandidate],
    ) -> CandidateFilterResult:
        prepared = await self.prepare_references(references)
        filtered = await self.filter_prepared(prepared, candidates)
        return CandidateFilterResult(
            candidates=filtered.candidates,
            reference_count=filtered.reference_count,
            analyzed_candidate_count=filtered.analyzed_candidate_count,
            skipped_candidate_count=filtered.skipped_candidate_count,
            retrieval_match_count=filtered.retrieval_match_count,
            identity_match_count=filtered.identity_match_count,
            deepfake_analyzed_candidate_count=(
                filtered.deepfake_analyzed_candidate_count
            ),
            deepfake_suspected_candidate_count=(
                filtered.deepfake_suspected_candidate_count
            ),
            deepfake_failed_candidate_count=filtered.deepfake_failed_candidate_count,
            processing_ms=prepared.processing_ms + filtered.processing_ms,
        )
