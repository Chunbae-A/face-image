"""얼굴가드 API의 공개 응답 형식."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
    threshold: float = Field(ge=-1.0, le=1.0)
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


class SubmittedSearchCandidate(BaseModel):
    page_url: str = Field(min_length=1, max_length=2048)
    media_url: str | None = Field(default=None, min_length=1, max_length=2048)
    thumbnail_url: str | None = Field(default=None, min_length=1, max_length=2048)
    content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    perceptual_hash: str | None = Field(default=None, min_length=16, max_length=16)


class SearchCandidatesRequest(BaseModel):
    privacy_mode: Literal["privacy_strict", "web_monitoring"] = "privacy_strict"
    web_monitoring_consent: bool = False
    candidates: list[SubmittedSearchCandidate] = Field(min_length=1)


class SearchCandidateResponse(BaseModel):
    page_url: str
    media_url: str | None = None
    thumbnail_url: str | None = None
    provider: str
    providers: list[str]
    rank: int = Field(gt=0)
    retrieved_at: datetime


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
