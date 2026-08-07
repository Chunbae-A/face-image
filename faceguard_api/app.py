"""FastAPI 기반 딥소각 얼굴가드 서비스."""

from __future__ import annotations

import time
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .candidate_filter import CandidateFilterService
from .domain import FaceQuality
from .engine import InsightFaceEncoder
from .errors import FaceGuardError
from .media import PublicImageDownloader
from .schemas import (
    CandidateFaceDecisionResponse,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ImageQualityResponse,
    SearchAndFilterResponse,
    SearchCandidateResponse,
    SearchCandidatesRequest,
    SearchCandidatesResponse,
    SearchProviderResponse,
    VerificationResponse,
)
from .search import (
    SearchQuery,
    SearchService,
    SearXNGProvider,
    SubmittedCandidate,
    UserSubmittedUrlProvider,
)
from .service import FaceGuardService
from .settings import Settings

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
RESEARCH_WARNING = (
    "현재 판정 기준값은 Celeb-real 연구 기준선이며 운영 확정값이 아닙니다. "
    "Kaggle 열화 실험과 외부 한국인·실제 촬영 데이터 검증이 필요합니다."
)
SEARCH_WARNING = (
    "현재 무료 모드는 사용자가 직접 넣은 공개 URL의 안전성 검사·정규화·중복 제거만 "
    "수행합니다. 인터넷 자동 검색이나 후보 발견을 완료했다는 뜻이 아닙니다."
)
SEARXNG_WARNING = (
    "SearXNG은 검색어 기반 공개 후보 수집이며 얼굴 사진 역검색이 아닙니다. "
    "후보가 본인인지와 딥페이크인지는 ArcFace·딥페이크 모델 단계에서 별도로 확인해야 합니다."
)
PIPELINE_WARNING = (
    "검색 후보 이미지를 ArcFace로 비교한 연구용 결과입니다. 검색 누락과 오인식이 가능하며, "
    "retrieval_match 또는 identity_match만으로 딥페이크나 피해 사실을 확정하지 않습니다."
)


def _quality_response(quality: FaceQuality) -> ImageQualityResponse:
    return ImageQualityResponse(
        detection_score=quality.detection_score,
        face_area_ratio=quality.face_area_ratio,
        blur_score=quality.blur_score,
        brightness_mean=quality.brightness_mean,
        image_width=quality.image_width,
        image_height=quality.image_height,
    )


async def _read_upload(upload: UploadFile, settings: Settings) -> bytes:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise FaceGuardError(
            "UNSUPPORTED_CONTENT_TYPE",
            "JPEG, PNG, WEBP 파일만 전송할 수 있습니다.",
            415,
        )
    payload = await upload.read(settings.maximum_image_bytes + 1)
    await upload.close()
    if len(payload) > settings.maximum_image_bytes:
        raise FaceGuardError(
            "IMAGE_TOO_LARGE",
            f"이미지 한 장은 {settings.maximum_image_bytes} bytes 이하여야 합니다.",
            413,
        )
    if not payload:
        raise FaceGuardError("EMPTY_IMAGE", "빈 이미지 파일은 사용할 수 없습니다.")
    return payload


def create_app(
    settings: Settings | None = None,
    encoder: Any | None = None,
    search_service: Any | None = None,
    image_downloader: Any | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_environment()
    active_encoder = encoder or InsightFaceEncoder(active_settings)
    service = FaceGuardService(active_settings, active_encoder)
    if search_service is None:
        search_providers: list[Any] = [UserSubmittedUrlProvider()]
        if active_settings.searxng_base_url:
            search_providers.append(
                SearXNGProvider(
                    active_settings.searxng_base_url,
                    request_timeout_seconds=(
                        active_settings.searxng_request_timeout_seconds
                    ),
                    maximum_retries=active_settings.searxng_maximum_retries,
                    retry_backoff_seconds=(
                        active_settings.searxng_retry_backoff_seconds
                    ),
                )
            )
        active_search_service = SearchService(
            search_providers,
            maximum_candidates=active_settings.maximum_search_candidates,
            provider_timeout_seconds=active_settings.search_provider_timeout_seconds,
        )
    else:
        active_search_service = search_service
    active_image_downloader = image_downloader or PublicImageDownloader(
        maximum_bytes=active_settings.maximum_image_bytes,
        timeout_seconds=active_settings.candidate_download_timeout_seconds,
        maximum_redirects=active_settings.candidate_download_maximum_redirects,
    )
    candidate_filter_service = CandidateFilterService(
        active_settings,
        active_encoder,
        active_image_downloader,
    )

    application = FastAPI(
        title="딥소각 얼굴가드 API",
        description=(
            "등록 얼굴 사진과 확인 사진을 비교하고 공개 웹 후보를 수집·정규화하는 연구용 API입니다. "
            "원본과 임베딩은 애플리케이션에 영구 저장하지 않습니다."
        ),
        version=active_settings.api_version,
    )

    @application.exception_handler(FaceGuardError)
    async def faceguard_error_handler(
        request: Request, error: FaceGuardError
    ) -> JSONResponse:
        del request
        body = ErrorResponse(error=ErrorBody(code=error.code, message=error.message))
        return JSONResponse(status_code=error.http_status, content=body.model_dump())

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del error
        if request.url.path == "/v1/search/candidates":
            message = "privacy_mode와 공개 후보 URL 목록을 JSON 형식으로 보내세요."
        elif request.url.path == "/v1/pipeline/search-and-filter":
            message = "등록 사진과 검색어를 multipart/form-data 형식으로 보내세요."
        else:
            message = "등록 사진과 확인 사진을 multipart/form-data 형식으로 보내세요."
        body = ErrorResponse(
            error=ErrorBody(
                code="INVALID_REQUEST",
                message=message,
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @application.get("/health", response_model=HealthResponse, tags=["운영"])
    async def health() -> HealthResponse:
        license_accepted = active_settings.accept_noncommercial_model_license
        providers = list(getattr(active_search_service, "providers", ()))
        return HealthResponse(
            status="ok" if license_accepted else "license_confirmation_required",
            api_version=active_settings.api_version,
            model_name=active_settings.model_name,
            model_loaded=bool(active_encoder.loaded),
            execution_provider=active_encoder.provider,
            model_fingerprint=active_encoder.model_fingerprint,
            license_accepted=license_accepted,
            threshold_status=active_settings.threshold_status,
            search_providers=[provider.name for provider in providers],
            web_search_enabled=any(
                provider.accesses_external_network for provider in providers
            ),
        )

    @application.post(
        "/v1/faceguard/verify",
        response_model=VerificationResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["얼굴가드"],
        summary="등록 사진과 확인 사진의 동일인 여부 비교",
    )
    async def verify(
        reference_images: Annotated[
            list[UploadFile],
            File(description="등록 얼굴 사진 1~5장, 3장 권장"),
        ],
        query_image: Annotated[
            UploadFile,
            File(description="동일인 여부를 확인할 얼굴 사진 1장"),
        ],
    ) -> VerificationResponse:
        if not active_settings.accept_noncommercial_model_license:
            raise FaceGuardError(
                "MODEL_LICENSE_NOT_ACCEPTED",
                "InsightFace 비상업 연구용 가중치 조건 확인이 필요합니다.",
                503,
            )
        if len(reference_images) < active_settings.minimum_reference_images:
            raise FaceGuardError(
                "TOO_FEW_REFERENCES",
                f"등록 사진이 최소 {active_settings.minimum_reference_images}장 필요합니다.",
            )
        if len(reference_images) > active_settings.maximum_reference_images:
            raise FaceGuardError(
                "TOO_MANY_REFERENCES",
                f"등록 사진은 최대 {active_settings.maximum_reference_images}장까지 사용할 수 있습니다.",
            )

        reference_payloads = [
            await _read_upload(upload, active_settings) for upload in reference_images
        ]
        query_payload = await _read_upload(query_image, active_settings)
        verification = await run_in_threadpool(
            service.verify, reference_payloads, query_payload
        )
        return VerificationResponse(
            request_id=str(uuid4()),
            is_same_person=verification.is_same_person,
            similarity=verification.similarity,
            threshold=verification.threshold,
            threshold_status=verification.threshold_status,
            threshold_source=verification.threshold_source,
            warning=RESEARCH_WARNING,
            reference_count=verification.reference_count,
            recommended_reference_count=active_settings.recommended_reference_images,
            reference_quality=[
                _quality_response(face.quality) for face in verification.reference_faces
            ],
            query_quality=_quality_response(verification.query_face.quality),
            processing_ms=verification.processing_ms,
            model_name=active_settings.model_name,
            execution_provider=active_encoder.provider,
            model_fingerprint=active_encoder.model_fingerprint,
        )

    @application.post(
        "/v1/search/candidates",
        response_model=SearchCandidatesResponse,
        responses={
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["공개 후보 검색"],
        summary="공개 URL 제보 또는 SearXNG 검색어로 후보 수집·중복 제거",
    )
    async def search_candidates(
        payload: SearchCandidatesRequest,
    ) -> SearchCandidatesResponse:
        query = SearchQuery(
            privacy_mode=payload.privacy_mode,
            web_monitoring_consent=payload.web_monitoring_consent,
            text_query=payload.query_text,
            categories=tuple(payload.categories),
            language=payload.language,
            safe_search=payload.safe_search,
            maximum_results=payload.maximum_results,
            submitted_candidates=[
                SubmittedCandidate(
                    page_url=candidate.page_url,
                    media_url=candidate.media_url,
                    thumbnail_url=candidate.thumbnail_url,
                    content_sha256=candidate.content_sha256,
                    perceptual_hash=candidate.perceptual_hash,
                )
                for candidate in payload.candidates
            ],
        )
        result = await active_search_service.search(query)
        return SearchCandidatesResponse(
            request_id=str(uuid4()),
            status=result.status,
            privacy_mode=result.privacy_mode,
            candidates=[
                SearchCandidateResponse(
                    page_url=candidate.page_url,
                    media_url=candidate.media_url,
                    thumbnail_url=candidate.thumbnail_url,
                    provider=candidate.provider,
                    providers=list(candidate.providers),
                    rank=candidate.rank,
                    retrieved_at=candidate.retrieved_at,
                    source_engine=candidate.source_engine,
                )
                for candidate in result.candidates
            ],
            providers=[
                SearchProviderResponse(
                    provider=provider.provider,
                    status=provider.status,
                    candidate_count=provider.candidate_count,
                    processing_ms=provider.processing_ms,
                    error_code=provider.error_code,
                )
                for provider in result.providers
            ],
            raw_candidate_count=result.raw_candidate_count,
            candidate_count=len(result.candidates),
            duplicate_count=result.duplicate_count,
            truncated_count=result.truncated_count,
            processing_ms=result.processing_ms,
            warning=(
                SEARXNG_WARNING
                if any(provider.provider == "searxng" for provider in result.providers)
                else SEARCH_WARNING
            ),
        )

    @application.post(
        "/v1/pipeline/search-and-filter",
        response_model=SearchAndFilterResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["통합 파이프라인"],
        summary="SearXNG 이미지 후보를 ArcFace로 동일인 가능성 선별",
    )
    async def search_and_filter(
        reference_images: Annotated[
            list[UploadFile],
            File(description="등록 얼굴 사진 1~5장, 3장 권장"),
        ],
        query_text: Annotated[
            str,
            Form(
                min_length=1,
                max_length=200,
                description="공개 검색에 동의한 검색어",
            ),
        ],
        web_monitoring_consent: Annotated[
            bool,
            Form(description="검색어 외부 전송에 대한 명시적 동의"),
        ],
        maximum_results: Annotated[
            int,
            Form(
                ge=1,
                le=10,
                description="검색 후 얼굴 비교할 최대 후보 수",
            ),
        ] = 5,
        language: Annotated[
            str,
            Form(
                min_length=2,
                max_length=6,
                pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2})?$",
            ),
        ] = "ko-KR",
        safe_search: Annotated[int, Form(ge=1, le=2)] = 2,
    ) -> SearchAndFilterResponse:
        if not active_settings.accept_noncommercial_model_license:
            raise FaceGuardError(
                "MODEL_LICENSE_NOT_ACCEPTED",
                "InsightFace 비상업 연구용 가중치 조건 확인이 필요합니다.",
                503,
            )
        if not web_monitoring_consent:
            raise FaceGuardError(
                "WEB_MONITORING_CONSENT_REQUIRED",
                "외부 웹 검색을 사용하려면 검색어 전송 동의가 필요합니다.",
            )
        if len(reference_images) < active_settings.minimum_reference_images:
            raise FaceGuardError(
                "TOO_FEW_REFERENCES",
                f"등록 사진이 최소 {active_settings.minimum_reference_images}장 필요합니다.",
            )
        if len(reference_images) > active_settings.maximum_reference_images:
            raise FaceGuardError(
                "TOO_MANY_REFERENCES",
                f"등록 사진은 최대 {active_settings.maximum_reference_images}장까지 사용할 수 있습니다.",
            )
        if maximum_results > active_settings.maximum_pipeline_candidates:
            raise FaceGuardError(
                "TOO_MANY_PIPELINE_CANDIDATES",
                f"얼굴 비교 후보는 최대 {active_settings.maximum_pipeline_candidates}개까지 처리할 수 있습니다.",
            )

        started = time.perf_counter()
        reference_payloads = [
            await _read_upload(upload, active_settings) for upload in reference_images
        ]
        prepared_references = await candidate_filter_service.prepare_references(
            reference_payloads
        )
        search_result = await active_search_service.search(
            SearchQuery(
                privacy_mode="web_monitoring",
                web_monitoring_consent=True,
                text_query=query_text,
                categories=("images",),
                language=language,
                safe_search=safe_search,
                maximum_results=maximum_results,
                submitted_candidates=[],
            )
        )
        filtered = await candidate_filter_service.filter_prepared(
            prepared_references,
            search_result.candidates,
        )
        status = (
            "partial_failed"
            if search_result.status == "partial_failed"
            or filtered.skipped_candidate_count
            else "completed"
        )
        return SearchAndFilterResponse(
            request_id=str(uuid4()),
            status=status,
            search_status=search_result.status,
            searched_candidate_count=len(search_result.candidates),
            analyzed_candidate_count=filtered.analyzed_candidate_count,
            skipped_candidate_count=filtered.skipped_candidate_count,
            retrieval_match_count=filtered.retrieval_match_count,
            identity_match_count=filtered.identity_match_count,
            retrieval_threshold=active_settings.retrieval_similarity_threshold,
            identity_threshold=active_settings.similarity_threshold,
            threshold_status=active_settings.threshold_status,
            retrieval_threshold_source=(
                "데모 연결용 임시 후보수집 기준값이며 공개 웹 validation으로 보정되지 않음"
            ),
            identity_threshold_source=active_settings.threshold_source,
            reference_count=filtered.reference_count,
            candidates=[
                CandidateFaceDecisionResponse(
                    page_url=item.page_url,
                    media_url=item.media_url,
                    thumbnail_url=item.thumbnail_url,
                    provider=item.provider,
                    source_engine=item.source_engine,
                    status=item.status,
                    similarity_raw=item.similarity_raw,
                    retrieval_match=item.retrieval_match,
                    identity_match=item.identity_match,
                    analyzed_url=item.analyzed_url,
                    matched_frame_count=1 if item.identity_match else 0,
                    analyzed_frame_count=0 if item.status == "skipped" else 1,
                    error_code=item.error_code,
                    quality_summary=(
                        _quality_response(item.quality) if item.quality else None
                    ),
                    processing_ms=item.processing_ms,
                )
                for item in filtered.candidates
            ],
            providers=[
                SearchProviderResponse(
                    provider=provider.provider,
                    status=provider.status,
                    candidate_count=provider.candidate_count,
                    processing_ms=provider.processing_ms,
                    error_code=provider.error_code,
                )
                for provider in search_result.providers
            ],
            processing_ms=(time.perf_counter() - started) * 1000.0,
            model_name=active_settings.model_name,
            execution_provider=active_encoder.provider,
            model_fingerprint=active_encoder.model_fingerprint,
            config_version="search-arcface-image-v1",
            warning=PIPELINE_WARNING,
        )

    return application


app = create_app()
