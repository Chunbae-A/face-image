"""얼굴가드 API의 공개 응답 형식."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    api_version: str
    model_name: str
    model_loaded: bool
    execution_provider: str | None = None
    model_fingerprint: str | None = None
    license_accepted: bool
    threshold_status: str
    search_providers: list[str]
    web_search_enabled: bool
    deepfake_model_name: str
    deepfake_model_loaded: bool
    deepfake_execution_provider: str | None = None
    deepfake_model_fingerprint: str | None = None
    deepfake_threshold_status: str
    deepfake_video_threshold_status: str
    deepfake_video_calibration_status: str
    deepfake_video_calibration_version: str | None = None


class ModelCapabilityResponse(BaseModel):
    """클라이언트 서버가 기능 노출 여부를 결정할 때 쓰는 모델 상태."""

    component_id: Literal["face_verification", "deepfake_image", "deepfake_video"]
    role: str
    model_name: str
    load_state: Literal["loaded", "lazy", "unavailable", "blocked"]
    decision_status: str
    score_semantics: Literal["cosine_similarity", "raw_model_score"]
    default_enabled: bool


class ApiCapabilitiesResponse(BaseModel):
    """딥소각 서버에 공개하는 안정적인 연구 API 기능 계약."""

    api_version: str
    deployment_mode: Literal["research_demo"]
    workflows: list[str]
    models: list[ModelCapabilityResponse]
    search_providers: list[str]
    web_search_enabled: bool
    scores_are_probabilities: Literal[False]
    automatic_enforcement_allowed: Literal[False]
    original_media_persisted: Literal[False]
    state_storage: Literal["process_memory_ttl"]
    warning: str


class ImageQualityResponse(BaseModel):
    detection_score: float = Field(ge=0.0, le=1.0)
    face_area_ratio: float = Field(gt=0.0, le=1.0)
    blur_score: float = Field(ge=0.0)
    brightness_mean: float = Field(ge=0.0, le=255.0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)


class VerificationResponse(BaseModel):
    request_id: str
    is_same_person: bool
    similarity: float = Field(ge=-1.0, le=1.0)
    raw_score: float = Field(ge=-1.0, le=1.0)
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_status: str
    calibration_version: str | None = None
    threshold: float = Field(ge=-1.0, le=1.0)
    decision_threshold: float = Field(ge=-1.0, le=1.0)
    threshold_status: str
    threshold_source: str
    warning: str
    reference_count: int
    recommended_reference_count: int
    reference_quality: list[ImageQualityResponse]
    query_quality: ImageQualityResponse
    processing_ms: float = Field(ge=0.0)
    model_name: str
    execution_provider: str | None = None
    model_fingerprint: str | None = None


class DeepfakeAnalysisResponse(BaseModel):
    request_id: str
    status: Literal["completed"]
    is_suspected_deepfake: bool
    deepfake_score: float = Field(ge=0.0, le=1.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_status: str
    calibration_version: str | None = None
    risk_level: Literal["low", "review", "high"] | None = None
    raw_logit: float
    threshold: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    threshold_status: str
    threshold_source: str
    warning: str
    quality_summary: ImageQualityResponse
    processing_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    model_name: str
    execution_provider: str
    model_fingerprint: str
    config_version: str


class VideoFrameAnalysisResponse(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    status: Literal["analyzed", "skipped"]
    deepfake_score: float | None = Field(default=None, ge=0.0, le=1.0)
    is_suspected_deepfake: bool | None = None
    face_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    quality_summary: ImageQualityResponse | None = None
    error_code: str | None = None
    processing_ms: float = Field(ge=0.0)
    inference_ms: float | None = Field(default=None, ge=0.0)


class SuspiciousSegmentResponse(BaseModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    peak_score: float = Field(ge=0.0, le=1.0)
    analyzed_frame_count: int = Field(gt=0)


class DeepfakeVideoAnalysisResponse(BaseModel):
    request_id: str
    status: Literal["completed", "partial_failed"]
    is_suspected_deepfake: bool
    video_score: float = Field(ge=0.0, le=1.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_status: str
    calibration_version: str | None = None
    risk_level: Literal["low", "review", "high"] | None = None
    threshold: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    threshold_status: str
    threshold_source: str
    aggregation: Literal["mean"]
    warning: str
    duration_seconds: float = Field(gt=0.0)
    fps: float = Field(gt=0.0)
    total_frame_count: int = Field(gt=0)
    requested_frame_count: int = Field(gt=0)
    decoded_frame_count: int = Field(gt=0)
    analyzed_frame_count: int = Field(gt=0)
    skipped_frame_count: int = Field(ge=0)
    reference_count: int = Field(ge=0)
    frames: list[VideoFrameAnalysisResponse]
    suspicious_segments: list[SuspiciousSegmentResponse]
    processing_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    model_name: str
    execution_provider: str
    model_fingerprint: str
    config_version: str


class SubmittedSearchCandidate(BaseModel):
    page_url: str = Field(min_length=1, max_length=2048)
    media_url: str | None = Field(default=None, min_length=1, max_length=2048)
    thumbnail_url: str | None = Field(default=None, min_length=1, max_length=2048)
    content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    perceptual_hash: str | None = Field(default=None, min_length=16, max_length=16)


class SearchCandidatesRequest(BaseModel):
    privacy_mode: Literal["privacy_strict"] = "privacy_strict"
    maximum_results: int = Field(default=20, ge=1, le=50)
    candidates: list[SubmittedSearchCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_search_source(self) -> SearchCandidatesRequest:
        if not self.candidates:
            raise ValueError("Google Vision 또는 사용자가 수집한 공개 후보 URL이 필요합니다.")
        return self


class SearchCandidateResponse(BaseModel):
    page_url: str
    media_url: str | None = None
    thumbnail_url: str | None = None
    provider: str
    providers: list[str]
    rank: int = Field(gt=0)
    retrieved_at: datetime
    source_engine: str | None = None


class SearchProviderResponse(BaseModel):
    provider: str
    status: Literal["completed", "failed"]
    candidate_count: int = Field(ge=0)
    processing_ms: float = Field(ge=0.0)
    error_code: str | None = None


class SearchCandidatesResponse(BaseModel):
    request_id: str
    status: Literal["completed", "partial_failed"]
    privacy_mode: Literal["privacy_strict", "web_monitoring"]
    candidates: list[SearchCandidateResponse]
    providers: list[SearchProviderResponse]
    raw_candidate_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    truncated_count: int = Field(ge=0)
    processing_ms: float = Field(ge=0.0)
    warning: str


class CandidateDeepfakeDecisionResponse(BaseModel):
    status: Literal["analyzed", "not_analyzed", "failed", "unavailable"]
    deepfake_score: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_score: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_status: str
    calibration_version: str | None = None
    risk_level: Literal["low", "review", "high"] | None = None
    is_suspected_deepfake: bool | None = None
    error_code: str | None = None
    processing_ms: float = Field(ge=0.0)
    inference_ms: float | None = Field(default=None, ge=0.0)
    execution_provider: str | None = None
    model_fingerprint: str | None = None


class CandidateFaceDecisionResponse(BaseModel):
    page_url: str
    media_url: str | None = None
    thumbnail_url: str | None = None
    provider: str
    source_engine: str | None = None
    status: Literal["identity_match", "retrieval_match", "not_matched", "skipped"]
    similarity_raw: float | None = Field(default=None, ge=-1.0, le=1.0)
    retrieval_match: bool | None = None
    identity_match: bool | None = None
    analyzed_url: str | None = None
    matched_frame_count: int = Field(ge=0, le=1)
    analyzed_frame_count: int = Field(ge=0, le=1)
    error_code: str | None = None
    quality_summary: ImageQualityResponse | None = None
    deepfake: CandidateDeepfakeDecisionResponse
    processing_ms: float = Field(ge=0.0)


class FaceEnrollmentResponse(BaseModel):
    enrollment_id: str
    status: Literal["active"]
    reference_count: int = Field(gt=0, le=5)
    recommended_reference_count: int = Field(gt=0)
    reference_quality: list[ImageQualityResponse]
    created_at: datetime
    expires_at: datetime
    storage: Literal["memory_only"]
    warning: str


class ExposureScanRequest(BaseModel):
    enrollment_id: str = Field(min_length=1, max_length=64)
    privacy_mode: Literal["privacy_strict"] = "privacy_strict"
    maximum_results: int = Field(default=5, ge=1, le=10)
    candidates: list[SubmittedSearchCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_search_source(self) -> ExposureScanRequest:
        if not self.candidates:
            raise ValueError("Google Vision 또는 사용자가 수집한 공개 후보 URL이 필요합니다.")
        return self


class ExposureScanCreatedResponse(BaseModel):
    scan_id: str
    status: Literal[
        "queued",
        "searching",
        "identity_filtering",
        "deepfake_analyzing",
        "completed",
        "partial_failed",
        "failed",
    ]
    reused: bool
    status_url: str
    candidates_url: str
    client_candidates_url: str
    created_at: datetime
    expires_at: datetime
    warning: str


class ExposureScanProgressResponse(BaseModel):
    searched_candidate_count: int = Field(ge=0)
    analyzed_candidate_count: int = Field(ge=0)
    skipped_candidate_count: int = Field(ge=0)
    identity_match_count: int = Field(ge=0)
    deepfake_completed_count: int = Field(ge=0)
    deepfake_failed_count: int = Field(ge=0)


class ExposureScanStatusResponse(BaseModel):
    scan_id: str
    status: Literal[
        "queued",
        "searching",
        "identity_filtering",
        "deepfake_analyzing",
        "completed",
        "partial_failed",
        "failed",
    ]
    progress_percent: int = Field(ge=0, le=100)
    progress: ExposureScanProgressResponse
    stage_durations_ms: dict[str, float]
    error_code: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    completed_at: datetime | None = None
    expires_at: datetime
    processing_ms: float = Field(ge=0.0)
    warning: str


class ExposureCandidateResponse(BaseModel):
    candidate_id: str
    scan_id: str
    result: CandidateFaceDecisionResponse
    warning: str


class ExposureCandidatesResponse(BaseModel):
    scan_id: str
    status: Literal[
        "queued",
        "searching",
        "identity_filtering",
        "deepfake_analyzing",
        "completed",
        "partial_failed",
        "failed",
    ]
    candidate_count: int = Field(ge=0)
    candidates: list[ExposureCandidateResponse]
    warning: str


class ClientExposureCandidateResponse(BaseModel):
    """딥소각 화면에서 모델 내부 구조를 몰라도 사용할 수 있는 후보 응답."""

    candidate_id: str
    source_url: str
    media_url: str | None = None
    thumbnail_url: str | None = None
    source_type: str
    source_engine: str | None = None
    face_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    face_match_level: Literal["matched", "review", "not_matched", "unavailable"]
    deepfake_score: float | None = Field(default=None, ge=0.0, le=1.0)
    deepfake_signal: Literal[
        "suspected", "not_suspected", "not_analyzed", "unavailable"
    ]
    recommended_action: Literal[
        "review_required",
        "identity_review_required",
        "monitor",
        "exclude_recommended",
        "analysis_unavailable",
    ]
    analysis_status: Literal["completed", "partial_failed", "unavailable"]
    warning: str


class ClientExposureCandidatesResponse(BaseModel):
    """화면 목록과 요약 카드에 필요한 최소 노출 후보 묶음."""

    scan_id: str
    status: Literal[
        "queued",
        "searching",
        "identity_filtering",
        "deepfake_analyzing",
        "completed",
        "partial_failed",
        "failed",
    ]
    candidate_count: int = Field(ge=0)
    identity_match_count: int = Field(ge=0)
    review_candidate_count: int = Field(ge=0)
    candidates: list[ClientExposureCandidateResponse]
    warning: str
