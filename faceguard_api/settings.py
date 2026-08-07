"""환경변수로 관리하는 얼굴가드 API 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

RESEARCH_THRESHOLD = 0.2823836207389832


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}은 true 또는 false여야 합니다.")


@dataclass(frozen=True)
class Settings:
    api_version: str = "0.3.0"
    model_name: str = "buffalo_l"
    model_root: Path = Path(".models/insightface")
    device: str = "auto"
    detection_size: int = 640
    similarity_threshold: float = RESEARCH_THRESHOLD
    threshold_status: str = "research_only_unapproved"
    threshold_source: str = (
        "Celeb-real 기준선 5프레임·등록 3개, FAR 0.001 목표의 seed별 기준값 최댓값"
    )
    accept_noncommercial_model_license: bool = False
    minimum_reference_images: int = 1
    recommended_reference_images: int = 3
    maximum_reference_images: int = 5
    maximum_image_bytes: int = 8 * 1024 * 1024
    maximum_image_pixels: int = 20_000_000
    minimum_detection_score: float = 0.60
    minimum_face_area_ratio: float = 0.01
    maximum_search_candidates: int = 100
    search_provider_timeout_seconds: float = 12.0
    searxng_base_url: str | None = None
    searxng_request_timeout_seconds: float = 4.0
    searxng_maximum_retries: int = 1
    searxng_retry_backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold는 -1과 1 사이여야 합니다.")
        if self.detection_size <= 0:
            raise ValueError("detection_size는 양수여야 합니다.")
        if not (
            1
            <= self.minimum_reference_images
            <= self.recommended_reference_images
            <= self.maximum_reference_images
        ):
            raise ValueError("등록 사진 수 설정의 순서가 올바르지 않습니다.")
        if self.maximum_image_bytes <= 0 or self.maximum_image_pixels <= 0:
            raise ValueError("이미지 크기 제한은 양수여야 합니다.")
        if not 0.0 <= self.minimum_detection_score <= 1.0:
            raise ValueError("minimum_detection_score는 0과 1 사이여야 합니다.")
        if not 0.0 < self.minimum_face_area_ratio <= 1.0:
            raise ValueError("minimum_face_area_ratio는 0보다 크고 1 이하여야 합니다.")
        if self.maximum_search_candidates <= 0:
            raise ValueError("maximum_search_candidates는 양수여야 합니다.")
        if self.search_provider_timeout_seconds <= 0:
            raise ValueError("search_provider_timeout_seconds는 양수여야 합니다.")
        if self.searxng_base_url:
            parsed = urlsplit(self.searxng_base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname is None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("searxng_base_url은 인증정보·쿼리 없는 HTTP(S) URL이어야 합니다.")
        if self.searxng_request_timeout_seconds <= 0:
            raise ValueError("searxng_request_timeout_seconds는 양수여야 합니다.")
        if self.searxng_maximum_retries < 0:
            raise ValueError("searxng_maximum_retries는 0 이상이어야 합니다.")
        if self.searxng_retry_backoff_seconds < 0:
            raise ValueError("searxng_retry_backoff_seconds는 0 이상이어야 합니다.")

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            model_name=os.environ.get("FACEGUARD_MODEL_NAME", "buffalo_l"),
            model_root=Path(
                os.environ.get("FACEGUARD_MODEL_ROOT", ".models/insightface")
            ).expanduser(),
            device=os.environ.get("FACEGUARD_DEVICE", "auto").strip().lower(),
            detection_size=int(os.environ.get("FACEGUARD_DETECTION_SIZE", "640")),
            similarity_threshold=float(
                os.environ.get("FACEGUARD_SIMILARITY_THRESHOLD", RESEARCH_THRESHOLD)
            ),
            threshold_status=os.environ.get(
                "FACEGUARD_THRESHOLD_STATUS", "research_only_unapproved"
            ),
            accept_noncommercial_model_license=_environment_bool(
                "FACEGUARD_ACCEPT_NONCOMMERCIAL_MODEL_LICENSE", False
            ),
            maximum_image_bytes=int(
                os.environ.get("FACEGUARD_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))
            ),
            maximum_image_pixels=int(
                os.environ.get("FACEGUARD_MAX_IMAGE_PIXELS", "20000000")
            ),
            minimum_detection_score=float(
                os.environ.get("FACEGUARD_MIN_DETECTION_SCORE", "0.60")
            ),
            minimum_face_area_ratio=float(
                os.environ.get("FACEGUARD_MIN_FACE_AREA_RATIO", "0.01")
            ),
            maximum_search_candidates=int(
                os.environ.get("FACEGUARD_MAX_SEARCH_CANDIDATES", "100")
            ),
            search_provider_timeout_seconds=float(
                os.environ.get("FACEGUARD_SEARCH_PROVIDER_TIMEOUT_SECONDS", "12.0")
            ),
            searxng_base_url=(
                os.environ.get("FACEGUARD_SEARXNG_BASE_URL", "").strip() or None
            ),
            searxng_request_timeout_seconds=float(
                os.environ.get("FACEGUARD_SEARXNG_REQUEST_TIMEOUT_SECONDS", "4.0")
            ),
            searxng_maximum_retries=int(
                os.environ.get("FACEGUARD_SEARXNG_MAXIMUM_RETRIES", "1")
            ),
            searxng_retry_backoff_seconds=float(
                os.environ.get("FACEGUARD_SEARXNG_RETRY_BACKOFF_SECONDS", "0.25")
            ),
        )
