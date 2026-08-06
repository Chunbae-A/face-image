"""얼굴가드 API의 공개 응답 형식."""

from __future__ import annotations

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
