"""FastAPI 기반 딥소각 얼굴가드 서비스."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .calibration import ScoreCalibration, unavailable_calibration_result
from .candidate_filter import CandidateFilterService
from .deepfake import DeepfakeOnnxAnalyzer
from .domain import FaceQuality
from .engine import InsightFaceEncoder
from .errors import FaceGuardError
from .exposure import (
    EphemeralEnrollmentStore,
    ExposureCandidate,
    ExposureScanManager,
    ExposureScanRecord,
)
from .media import PublicImageDownloader
from .schemas import (
    ApiCapabilitiesResponse,
    CandidateDeepfakeDecisionResponse,
    CandidateFaceDecisionResponse,
    ClientExposureCandidateResponse,
    ClientExposureCandidatesResponse,
    DeepfakeAnalysisResponse,
    DeepfakeVideoAnalysisResponse,
    ErrorBody,
    ErrorResponse,
    ExposureCandidateResponse,
    ExposureCandidatesResponse,
    ExposureScanCreatedResponse,
    ExposureScanProgressResponse,
    ExposureScanRequest,
    ExposureScanStatusResponse,
    FaceEnrollmentResponse,
    HealthResponse,
    ImageQualityResponse,
    ModelCapabilityResponse,
    SearchCandidateResponse,
    SearchCandidatesRequest,
    SearchCandidatesResponse,
    SearchProviderResponse,
    SuspiciousSegmentResponse,
    VerificationResponse,
    VideoFrameAnalysisResponse,
)
from .search import (
    SearchQuery,
    SearchService,
    SubmittedCandidate,
    UserSubmittedUrlProvider,
)
from .service import FaceGuardService
from .settings import Settings
from .video import VideoDeepfakeAnalyzer

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
}
RESEARCH_WARNING = (
    "현재 판정 기준값은 Celeb-real 연구 기준선이며 운영 확정값이 아닙니다. "
    "Kaggle 열화 실험과 외부 한국인·실제 촬영 데이터 검증이 필요합니다."
)
SEARCH_WARNING = (
    "현재 무료 모드는 사용자가 직접 넣은 공개 URL의 안전성 검사·정규화·중복 제거만 "
    "수행합니다. 인터넷 자동 검색이나 후보 발견을 완료했다는 뜻이 아닙니다."
)
PIPELINE_WARNING = (
    "검색 후보를 ArcFace로 선별하고 넓은 후보 기준을 통과한 단일 얼굴 이미지만 ONNX로 "
    "분석한 연구용 결과입니다. deepfake_score는 보정된 확률이나 확정 신뢰도가 아니며, "
    "사람 검토 없이 피해 사실을 확정하거나 자동 신고·삭제하지 않습니다."
)
EXPOSURE_WARNING = (
    "현재는 이미지 후보만 비동기로 처리하는 로컬 데모입니다. "
    "등록 임베딩과 작업 결과는 TTL 동안 프로세스 메모리에만 보관되며, "
    "서버를 재시작하면 사라집니다. 수치는 연구용이며 자동 신고·삭제에 사용하지 않습니다."
)
DEEPFAKE_WARNING = (
    "deepfake_score는 Celeb-DF-v2로 학습한 모델의 단일 얼굴 이미지 점수입니다. "
    "현재 기준값은 영상 16프레임 평균에서 선택됐으므로 단일 이미지 정확도와 운영 신뢰도를 "
    "보장하지 않습니다."
)
DEEPFAKE_VIDEO_WARNING = (
    "영상 점수는 대표 얼굴 프레임의 ONNX 점수를 평균한 연구 결과입니다. "
    "Celeb-DF-v2 공식 Test에서 실제 영상 오경고율 목표를 통과하지 못했으므로, "
    "사람 검토 없이 피해 사실을 확정하거나 자동 신고·삭제하지 않습니다."
)


class _RequestBodyTooLarge(Exception):
    pass


class _RequestBodyLimitMiddleware:
    """영상 multipart 본문을 파싱하기 전 전체 전송량을 제한한다."""

    def __init__(self, app: Any, *, path: str, maximum_bytes: int) -> None:
        self.app = app
        self.path = path
        self.maximum_bytes = maximum_bytes

    async def _reject(self, scope: Any, receive: Any, send: Any) -> None:
        response = JSONResponse(
            status_code=413,
            content=ErrorResponse(
                error=ErrorBody(
                    code="REQUEST_BODY_TOO_LARGE",
                    message=(
                        f"영상 분석 요청 전체는 {self.maximum_bytes} bytes 이하여야 합니다."
                    ),
                )
            ).model_dump(),
        )
        await response(scope, receive, send)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != self.path:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = 0
            if declared_bytes > self.maximum_bytes:
                await self._reject(scope, receive, send)
                return

        received_bytes = 0

        async def limited_receive() -> Any:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.maximum_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(scope, receive, send)


def _quality_response(quality: FaceQuality) -> ImageQualityResponse:
    return ImageQualityResponse(
        detection_score=quality.detection_score,
        face_area_ratio=quality.face_area_ratio,
        blur_score=quality.blur_score,
        brightness_mean=quality.brightness_mean,
        image_width=quality.image_width,
        image_height=quality.image_height,
    )


def _candidate_decision_response(item: Any) -> CandidateFaceDecisionResponse:
    return CandidateFaceDecisionResponse(
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
        quality_summary=_quality_response(item.quality) if item.quality else None,
        deepfake=CandidateDeepfakeDecisionResponse(
            status=item.deepfake.status,
            deepfake_score=item.deepfake.deepfake_score,
            raw_score=item.deepfake.deepfake_score,
            calibrated_probability=None,
            calibration_status=(
                "not_applicable_single_image"
                if item.deepfake.status == "analyzed"
                else "not_analyzed"
            ),
            calibration_version=None,
            risk_level=None,
            is_suspected_deepfake=item.deepfake.is_suspected_deepfake,
            error_code=item.deepfake.error_code,
            processing_ms=item.deepfake.processing_ms,
            inference_ms=item.deepfake.inference_ms,
            execution_provider=item.deepfake.execution_provider,
            model_fingerprint=item.deepfake.model_fingerprint,
        ),
        processing_ms=item.processing_ms,
    )


def _exposure_candidate_response(
    scan_id: str, candidate: ExposureCandidate
) -> ExposureCandidateResponse:
    return ExposureCandidateResponse(
        candidate_id=candidate.candidate_id,
        scan_id=scan_id,
        result=_candidate_decision_response(candidate.decision),
        warning=PIPELINE_WARNING,
    )


def _client_exposure_candidate_response(
    candidate: ExposureCandidate,
) -> ClientExposureCandidateResponse:
    """연구 원점수를 확률로 오해하지 않는 화면용 후보로 변환한다."""

    decision = candidate.decision
    if decision.status == "identity_match":
        face_match_level = "matched"
    elif decision.status == "retrieval_match":
        face_match_level = "review"
    elif decision.status == "not_matched":
        face_match_level = "not_matched"
    else:
        face_match_level = "unavailable"

    if decision.deepfake.status == "analyzed":
        deepfake_signal = (
            "suspected"
            if decision.deepfake.is_suspected_deepfake is True
            else "not_suspected"
        )
    elif decision.deepfake.status == "not_analyzed":
        deepfake_signal = "not_analyzed"
    else:
        deepfake_signal = "unavailable"

    if face_match_level == "not_matched":
        recommended_action = "exclude_recommended"
        analysis_status = "completed"
    elif face_match_level == "unavailable":
        recommended_action = "analysis_unavailable"
        analysis_status = "unavailable"
    elif decision.deepfake.status in {"failed", "unavailable"}:
        recommended_action = "analysis_unavailable"
        analysis_status = "partial_failed"
    elif face_match_level == "review":
        recommended_action = "identity_review_required"
        analysis_status = "completed"
    elif deepfake_signal == "suspected":
        recommended_action = "review_required"
        analysis_status = "completed"
    else:
        recommended_action = "monitor"
        analysis_status = "completed"

    return ClientExposureCandidateResponse(
        candidate_id=candidate.candidate_id,
        source_url=decision.page_url,
        media_url=decision.media_url,
        thumbnail_url=decision.thumbnail_url,
        source_type=decision.provider,
        source_engine=decision.source_engine,
        face_similarity=decision.similarity_raw,
        face_match_level=face_match_level,
        deepfake_score=decision.deepfake.deepfake_score,
        deepfake_signal=deepfake_signal,
        recommended_action=recommended_action,
        analysis_status=analysis_status,
        warning=PIPELINE_WARNING,
    )


def _scan_progress_percent(record: ExposureScanRecord) -> int:
    if record.status in {"completed", "partial_failed", "failed"}:
        return 100
    return {
        "queued": 0,
        "searching": 15,
        "identity_filtering": 45,
        "deepfake_analyzing": 80,
    }[record.status]


def _scan_status_response(record: ExposureScanRecord) -> ExposureScanStatusResponse:
    return ExposureScanStatusResponse(
        scan_id=record.scan_id,
        status=record.status,
        progress_percent=_scan_progress_percent(record),
        progress=ExposureScanProgressResponse(
            searched_candidate_count=record.progress.searched_candidate_count,
            analyzed_candidate_count=record.progress.analyzed_candidate_count,
            skipped_candidate_count=record.progress.skipped_candidate_count,
            identity_match_count=record.progress.identity_match_count,
            deepfake_completed_count=record.progress.deepfake_completed_count,
            deepfake_failed_count=record.progress.deepfake_failed_count,
        ),
        stage_durations_ms=record.stage_durations_ms,
        error_code=record.error_code,
        created_at=record.created_at,
        started_at=record.started_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
        expires_at=record.expires_at,
        processing_ms=record.processing_ms,
        warning=EXPOSURE_WARNING,
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


async def _read_video_upload(
    upload: UploadFile, settings: Settings
) -> tuple[bytes, str]:
    suffix = ALLOWED_VIDEO_CONTENT_TYPES.get(upload.content_type or "")
    if suffix is None:
        raise FaceGuardError(
            "UNSUPPORTED_VIDEO_CONTENT_TYPE",
            "MP4 또는 MOV 영상만 전송할 수 있습니다.",
            415,
        )
    payload = await upload.read(settings.maximum_video_bytes + 1)
    await upload.close()
    if len(payload) > settings.maximum_video_bytes:
        raise FaceGuardError(
            "VIDEO_TOO_LARGE",
            f"영상은 {settings.maximum_video_bytes} bytes 이하여야 합니다.",
            413,
        )
    if not payload:
        raise FaceGuardError("EMPTY_VIDEO", "빈 영상 파일은 사용할 수 없습니다.")
    return payload, suffix


def create_app(
    settings: Settings | None = None,
    encoder: Any | None = None,
    search_service: Any | None = None,
    image_downloader: Any | None = None,
    deepfake_analyzer: Any | None = None,
    video_deepfake_analyzer: Any | None = None,
    video_score_calibration: ScoreCalibration | None = None,
    exposure_scan_manager: Any | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_environment()
    active_encoder = encoder or InsightFaceEncoder(active_settings)
    service = FaceGuardService(active_settings, active_encoder)
    if search_service is None:
        search_providers: list[Any] = [UserSubmittedUrlProvider()]
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
    active_deepfake_analyzer = deepfake_analyzer or DeepfakeOnnxAnalyzer(
        active_settings,
        active_encoder,
    )
    active_video_deepfake_analyzer = video_deepfake_analyzer or VideoDeepfakeAnalyzer(
        active_settings,
        active_encoder,
        active_deepfake_analyzer,
    )
    active_video_score_calibration = video_score_calibration or ScoreCalibration.load(
        active_settings.deepfake_calibration_path,
        expected_model_fingerprint=active_settings.deepfake_model_sha256,
        expected_scope="deepfake_video_mean_16_frames",
    )
    candidate_filter_service = CandidateFilterService(
        active_settings,
        active_encoder,
        active_image_downloader,
        active_deepfake_analyzer,
    )
    owns_exposure_scan_manager = exposure_scan_manager is None
    active_exposure_scan_manager = exposure_scan_manager or ExposureScanManager(
        search_service=active_search_service,
        candidate_filter_service=candidate_filter_service,
        enrollment_store=EphemeralEnrollmentStore(
            ttl_seconds=active_settings.exposure_enrollment_ttl_seconds,
            maximum_entries=active_settings.maximum_exposure_enrollments,
        ),
        scan_ttl_seconds=active_settings.exposure_scan_ttl_seconds,
        maximum_scans=active_settings.maximum_exposure_scans,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        del application
        yield
        if owns_exposure_scan_manager:
            await active_exposure_scan_manager.close()

    application = FastAPI(
        title="딥소각 얼굴가드 API",
        description=(
            "등록 얼굴 비교, 공개 웹 후보 수집·선별과 이미지·짧은 영상 딥페이크 ONNX 분석을 제공하는 "
            "연구용 API입니다. "
            "원본과 임베딩은 애플리케이션에 영구 저장하지 않습니다."
        ),
        version=active_settings.api_version,
        lifespan=lifespan,
    )
    application.state.exposure_scan_manager = active_exposure_scan_manager
    application.add_middleware(
        _RequestBodyLimitMiddleware,
        path="/v1/deepfake/analyze-video",
        maximum_bytes=active_settings.maximum_video_request_bytes,
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
        elif request.url.path == "/v1/faceguard/enrollments":
            message = "등록 사진 1~5장을 multipart/form-data 형식으로 보내세요."
        elif request.url.path == "/v1/exposure-scans":
            message = "enrollment_id와 공개 후보 URL 목록을 JSON 형식으로 보내세요."
        elif request.url.path == "/v1/deepfake/analyze":
            message = "분석할 얼굴 이미지를 multipart/form-data 형식으로 보내세요."
        elif request.url.path == "/v1/deepfake/analyze-video":
            message = (
                "분석할 MP4 또는 MOV 영상을 multipart/form-data 형식으로 보내세요."
            )
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
            deepfake_model_name=active_settings.deepfake_model_name,
            deepfake_model_loaded=bool(active_deepfake_analyzer.loaded),
            deepfake_execution_provider=active_deepfake_analyzer.provider,
            deepfake_model_fingerprint=(active_deepfake_analyzer.model_fingerprint),
            deepfake_threshold_status=active_settings.deepfake_threshold_status,
            deepfake_video_threshold_status=(
                active_settings.deepfake_video_threshold_status
            ),
            deepfake_video_calibration_status=(
                active_video_score_calibration.status
                if active_video_score_calibration is not None
                else "not_available"
            ),
            deepfake_video_calibration_version=(
                active_video_score_calibration.version
                if active_video_score_calibration is not None
                else None
            ),
        )

    @application.get(
        "/v1/capabilities",
        response_model=ApiCapabilitiesResponse,
        tags=["운영"],
        summary="클라이언트 서버용 얼굴가드 기능·모델 상태 확인",
    )
    async def capabilities() -> ApiCapabilitiesResponse:
        license_accepted = active_settings.accept_noncommercial_model_license
        providers = list(getattr(active_search_service, "providers", ()))
        face_state = (
            "blocked"
            if not license_accepted
            else "loaded"
            if active_encoder.loaded
            else "lazy"
        )
        deepfake_state = (
            "blocked"
            if not license_accepted
            else "loaded"
            if active_deepfake_analyzer.loaded
            else "lazy"
            if active_settings.deepfake_model_path.is_file()
            else "unavailable"
        )
        return ApiCapabilitiesResponse(
            api_version=active_settings.api_version,
            deployment_mode="research_demo",
            workflows=[
                "face_verification",
                "deepfake_image_analysis",
                "deepfake_video_analysis",
                "public_exposure_scan",
            ],
            models=[
                ModelCapabilityResponse(
                    component_id="face_verification",
                    role="등록 얼굴과 공개 후보의 동일인 가능성 선별",
                    model_name=active_settings.model_name,
                    load_state=face_state,
                    decision_status=active_settings.threshold_status,
                    score_semantics="cosine_similarity",
                    default_enabled=license_accepted,
                ),
                ModelCapabilityResponse(
                    component_id="deepfake_image",
                    role="단일 얼굴 이미지의 딥페이크 의심 신호 분석",
                    model_name=active_settings.deepfake_model_name,
                    load_state=deepfake_state,
                    decision_status=active_settings.deepfake_threshold_status,
                    score_semantics="raw_model_score",
                    default_enabled=deepfake_state in {"loaded", "lazy"},
                ),
                ModelCapabilityResponse(
                    component_id="deepfake_video",
                    role="영상 대표 얼굴 프레임 16개의 평균 의심 신호 분석",
                    model_name=active_settings.deepfake_model_name,
                    load_state=deepfake_state,
                    decision_status=active_settings.deepfake_video_threshold_status,
                    score_semantics="raw_model_score",
                    default_enabled=deepfake_state in {"loaded", "lazy"},
                ),
            ],
            search_providers=[provider.name for provider in providers],
            web_search_enabled=any(
                provider.accesses_external_network for provider in providers
            ),
            scores_are_probabilities=False,
            automatic_enforcement_allowed=False,
            original_media_persisted=False,
            state_storage="process_memory_ttl",
            warning=PIPELINE_WARNING,
        )

    @application.post(
        "/v1/faceguard/enrollments",
        response_model=FaceEnrollmentResponse,
        status_code=201,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["비동기 노출 스캔"],
        summary="비동기 스캔에 쓸 본인 얼굴을 임시 등록",
    )
    async def create_face_enrollment(
        reference_images: Annotated[
            list[UploadFile],
            File(description="등록 얼굴 사진 1~5장, 3장 권장"),
        ],
    ) -> FaceEnrollmentResponse:
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
        record = await active_exposure_scan_manager.create_enrollment(
            reference_payloads
        )
        return FaceEnrollmentResponse(
            enrollment_id=record.enrollment_id,
            status="active",
            reference_count=len(record.references.faces),
            recommended_reference_count=active_settings.recommended_reference_images,
            reference_quality=[
                _quality_response(face.quality) for face in record.references.faces
            ],
            created_at=record.created_at,
            expires_at=record.expires_at,
            storage="memory_only",
            warning=EXPOSURE_WARNING,
        )

    @application.post(
        "/v1/exposure-scans",
        response_model=ExposureScanCreatedResponse,
        status_code=202,
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            410: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
        tags=["비동기 노출 스캔"],
        summary="공개 후보 검색·얼굴 선별·ONNX 분석 작업 시작",
    )
    async def create_exposure_scan(
        payload: ExposureScanRequest,
        idempotency_key: Annotated[
            str | None,
            Header(
                alias="Idempotency-Key",
                min_length=8,
                max_length=128,
                description="재시도 시 같은 scan_id를 받기 위한 요청 키",
            ),
        ] = None,
    ) -> ExposureScanCreatedResponse:
        if payload.maximum_results > active_settings.maximum_pipeline_candidates:
            raise FaceGuardError(
                "TOO_MANY_PIPELINE_CANDIDATES",
                f"얼굴 비교 후보는 최대 {active_settings.maximum_pipeline_candidates}개까지 처리할 수 있습니다.",
            )
        query = SearchQuery(
            privacy_mode=payload.privacy_mode,
            web_monitoring_consent=False,
            text_query=None,
            categories=("images",),
            language="ko-KR",
            safe_search=2,
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
        record, reused = await active_exposure_scan_manager.create_scan(
            enrollment_id=payload.enrollment_id,
            query=query,
            idempotency_key=idempotency_key,
        )
        return ExposureScanCreatedResponse(
            scan_id=record.scan_id,
            status=record.status,
            reused=reused,
            status_url=f"/v1/exposure-scans/{record.scan_id}",
            candidates_url=f"/v1/exposure-scans/{record.scan_id}/candidates",
            client_candidates_url=(
                f"/v1/exposure-scans/{record.scan_id}/client-candidates"
            ),
            created_at=record.created_at,
            expires_at=record.expires_at,
            warning=EXPOSURE_WARNING,
        )

    @application.get(
        "/v1/exposure-scans/{scan_id}",
        response_model=ExposureScanStatusResponse,
        responses={404: {"model": ErrorResponse}, 410: {"model": ErrorResponse}},
        tags=["비동기 노출 스캔"],
        summary="스캔 진행 단계와 처리 개수 확인",
    )
    async def get_exposure_scan(scan_id: str) -> ExposureScanStatusResponse:
        record = await active_exposure_scan_manager.get_scan(scan_id)
        return _scan_status_response(record)

    @application.get(
        "/v1/exposure-scans/{scan_id}/candidates",
        response_model=ExposureCandidatesResponse,
        responses={404: {"model": ErrorResponse}, 410: {"model": ErrorResponse}},
        tags=["비동기 노출 스캔"],
        summary="스캔에서 찾은 노출 후보 목록 확인",
    )
    async def get_exposure_candidates(scan_id: str) -> ExposureCandidatesResponse:
        record = await active_exposure_scan_manager.get_scan(scan_id)
        return ExposureCandidatesResponse(
            scan_id=record.scan_id,
            status=record.status,
            candidate_count=len(record.candidates),
            candidates=[
                _exposure_candidate_response(record.scan_id, candidate)
                for candidate in record.candidates
            ],
            warning=EXPOSURE_WARNING,
        )

    @application.get(
        "/v1/exposure-scans/{scan_id}/client-candidates",
        response_model=ClientExposureCandidatesResponse,
        responses={404: {"model": ErrorResponse}, 410: {"model": ErrorResponse}},
        tags=["비동기 노출 스캔"],
        summary="딥소각 후보 화면용 얼굴·딥페이크 결과 확인",
    )
    async def get_client_exposure_candidates(
        scan_id: str,
    ) -> ClientExposureCandidatesResponse:
        record = await active_exposure_scan_manager.get_scan(scan_id)
        candidates = [
            _client_exposure_candidate_response(candidate)
            for candidate in record.candidates
        ]
        return ClientExposureCandidatesResponse(
            scan_id=record.scan_id,
            status=record.status,
            candidate_count=len(candidates),
            identity_match_count=sum(
                item.face_match_level == "matched" for item in candidates
            ),
            review_candidate_count=sum(
                item.recommended_action
                in {"review_required", "identity_review_required"}
                for item in candidates
            ),
            candidates=candidates,
            warning=EXPOSURE_WARNING,
        )

    @application.get(
        "/v1/exposure-candidates/{candidate_id}/analysis",
        response_model=ExposureCandidateResponse,
        responses={404: {"model": ErrorResponse}, 410: {"model": ErrorResponse}},
        tags=["비동기 노출 스캔"],
        summary="후보 하나의 얼굴 유사도와 딥페이크 분석 확인",
    )
    async def get_exposure_candidate_analysis(
        candidate_id: str,
    ) -> ExposureCandidateResponse:
        record, candidate = await active_exposure_scan_manager.get_candidate(
            candidate_id
        )
        return _exposure_candidate_response(record.scan_id, candidate)

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
            raw_score=verification.similarity,
            calibrated_probability=None,
            calibration_status="not_available",
            calibration_version=None,
            threshold=verification.threshold,
            decision_threshold=verification.threshold,
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
        "/v1/deepfake/analyze",
        response_model=DeepfakeAnalysisResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["딥페이크 분석"],
        summary="얼굴 이미지 한 장을 Celeb-DF 연구용 ONNX로 분석",
    )
    async def analyze_deepfake(
        image: Annotated[
            UploadFile,
            File(description="얼굴 한 명이 선명하게 나온 JPEG, PNG 또는 WEBP"),
        ],
    ) -> DeepfakeAnalysisResponse:
        if not active_settings.accept_noncommercial_model_license:
            raise FaceGuardError(
                "MODEL_LICENSE_NOT_ACCEPTED",
                "InsightFace 비상업 연구용 얼굴 검출 가중치 조건 확인이 필요합니다.",
                503,
            )
        payload = await _read_upload(image, active_settings)
        result = await run_in_threadpool(active_deepfake_analyzer.analyze, payload)
        return DeepfakeAnalysisResponse(
            request_id=str(uuid4()),
            status="completed",
            is_suspected_deepfake=result.is_suspected_deepfake,
            deepfake_score=result.deepfake_score,
            raw_score=result.deepfake_score,
            calibrated_probability=None,
            calibration_status="not_applicable_single_image",
            calibration_version=None,
            risk_level=None,
            raw_logit=result.raw_logit,
            threshold=result.threshold,
            decision_threshold=result.threshold,
            threshold_status=active_settings.deepfake_threshold_status,
            threshold_source=active_settings.deepfake_threshold_source,
            warning=DEEPFAKE_WARNING,
            quality_summary=_quality_response(result.quality),
            processing_ms=result.processing_ms,
            inference_ms=result.inference_ms,
            model_name=result.model_name,
            execution_provider=result.execution_provider,
            model_fingerprint=result.model_fingerprint,
            config_version="deepfake-single-image-v1",
        )

    @application.post(
        "/v1/deepfake/analyze-video",
        response_model=DeepfakeVideoAnalysisResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["딥페이크 분석"],
        summary="짧은 영상의 16개 대표 얼굴 프레임을 분석",
    )
    async def analyze_deepfake_video(
        video: Annotated[
            UploadFile,
            File(description="최대 120초의 MP4 또는 MOV 영상"),
        ],
        reference_images: Annotated[
            list[UploadFile] | None,
            File(description="영상에서 본인 얼굴을 고를 등록 사진 0~5장, 3장 권장"),
        ] = None,
    ) -> DeepfakeVideoAnalysisResponse:
        if not active_settings.accept_noncommercial_model_license:
            raise FaceGuardError(
                "MODEL_LICENSE_NOT_ACCEPTED",
                "InsightFace 비상업 연구용 얼굴 검출 가중치 조건 확인이 필요합니다.",
                503,
            )
        references = reference_images or []
        if len(references) > active_settings.maximum_reference_images:
            raise FaceGuardError(
                "TOO_MANY_REFERENCES",
                f"등록 사진은 최대 {active_settings.maximum_reference_images}장까지 사용할 수 있습니다.",
            )
        reference_payloads = [
            await _read_upload(upload, active_settings) for upload in references
        ]
        payload, suffix = await _read_video_upload(video, active_settings)
        result = await run_in_threadpool(
            active_video_deepfake_analyzer.analyze,
            payload,
            suffix=suffix,
            reference_payloads=reference_payloads,
        )
        calibration = (
            active_video_score_calibration.apply(result.video_score)
            if active_video_score_calibration is not None
            else unavailable_calibration_result()
        )
        return DeepfakeVideoAnalysisResponse(
            request_id=str(uuid4()),
            status=result.status,
            is_suspected_deepfake=result.is_suspected_deepfake,
            video_score=result.video_score,
            raw_score=result.video_score,
            calibrated_probability=calibration.calibrated_probability,
            calibration_status=calibration.calibration_status,
            calibration_version=calibration.calibration_version,
            risk_level=calibration.risk_level,
            threshold=result.threshold,
            decision_threshold=result.threshold,
            threshold_status=active_settings.deepfake_video_threshold_status,
            threshold_source=active_settings.deepfake_video_threshold_source,
            aggregation=result.aggregation,
            warning=f"{DEEPFAKE_VIDEO_WARNING} {calibration.warning}",
            duration_seconds=result.duration_seconds,
            fps=result.fps,
            total_frame_count=result.total_frame_count,
            requested_frame_count=result.requested_frame_count,
            decoded_frame_count=result.decoded_frame_count,
            analyzed_frame_count=result.analyzed_frame_count,
            skipped_frame_count=result.skipped_frame_count,
            reference_count=result.reference_count,
            frames=[
                VideoFrameAnalysisResponse(
                    frame_index=item.frame_index,
                    timestamp_seconds=item.timestamp_seconds,
                    status=item.status,
                    deepfake_score=item.deepfake_score,
                    is_suspected_deepfake=item.is_suspected_deepfake,
                    face_similarity=item.face_similarity,
                    quality_summary=(
                        _quality_response(item.quality) if item.quality else None
                    ),
                    error_code=item.error_code,
                    processing_ms=item.processing_ms,
                    inference_ms=item.inference_ms,
                )
                for item in result.frames
            ],
            suspicious_segments=[
                SuspiciousSegmentResponse(
                    start_seconds=item.start_seconds,
                    end_seconds=item.end_seconds,
                    peak_score=item.peak_score,
                    analyzed_frame_count=item.analyzed_frame_count,
                )
                for item in result.suspicious_segments
            ],
            processing_ms=result.processing_ms,
            inference_ms=result.inference_ms,
            model_name=result.model_name,
            execution_provider=result.execution_provider,
            model_fingerprint=result.model_fingerprint,
            config_version=(
                f"deepfake-video-{active_settings.deepfake_video_frame_count}-frame-mean-v1"
            ),
        )

    @application.post(
        "/v1/search/candidates",
        response_model=SearchCandidatesResponse,
        responses={
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["공개 후보 검색"],
        summary="Google Vision 또는 사용자가 전달한 공개 URL 후보 정규화·중복 제거",
    )
    async def search_candidates(
        payload: SearchCandidatesRequest,
    ) -> SearchCandidatesResponse:
        query = SearchQuery(
            privacy_mode=payload.privacy_mode,
            web_monitoring_consent=False,
            text_query=None,
            categories=("images",),
            language="ko-KR",
            safe_search=2,
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
            warning=SEARCH_WARNING,
        )

    return application


app = create_app()
