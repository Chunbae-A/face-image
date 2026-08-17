"""환경변수로 관리하는 얼굴가드 API 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RESEARCH_THRESHOLD = 0.2823836207389832
RESEARCH_RETRIEVAL_THRESHOLD = 0.20
DEEPFAKE_RESEARCH_THRESHOLD = 0.7519882693886758
DEEPFAKE_MODEL_SHA256 = (
    "c32a8532e2e1bd275b833b16460946eb307207098e0c07e2247851b71c23a6f1"
)


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
    api_version: str = "0.9.0"
    model_name: str = "buffalo_l"
    model_root: Path = Path(".models/insightface")
    device: str = "auto"
    detection_size: int = 640
    similarity_threshold: float = RESEARCH_THRESHOLD
    retrieval_similarity_threshold: float = RESEARCH_RETRIEVAL_THRESHOLD
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
    maximum_video_bytes: int = 50 * 1024 * 1024
    maximum_video_seconds: float = 120.0
    maximum_video_request_bytes: int = 91 * 1024 * 1024
    minimum_detection_score: float = 0.60
    minimum_face_area_ratio: float = 0.01
    maximum_search_candidates: int = 100
    search_provider_timeout_seconds: float = 12.0
    maximum_pipeline_candidates: int = 10
    candidate_download_timeout_seconds: float = 4.0
    candidate_download_maximum_redirects: int = 2
    exposure_enrollment_ttl_seconds: int = 30 * 60
    exposure_scan_ttl_seconds: int = 60 * 60
    maximum_exposure_enrollments: int = 100
    maximum_exposure_scans: int = 100
    deepfake_model_name: str = "efficientnet_b4_celebdf_v2"
    deepfake_model_path: Path = Path(".models/deepfake/efficientnet_b4.onnx")
    deepfake_model_sha256: str = DEEPFAKE_MODEL_SHA256
    deepfake_calibration_path: Path = Path(
        ".models/deepfake/deepfake_video_calibration.json"
    )
    deepfake_device: str = "cpu"
    deepfake_input_size: int = 380
    deepfake_aligned_face_size: int = 224
    deepfake_threshold: float = DEEPFAKE_RESEARCH_THRESHOLD
    deepfake_threshold_status: str = "research_only_single_image_unvalidated"
    deepfake_threshold_source: str = "Celeb-DF-v2 영상 16프레임 평균 기준값 0.7519882694를 단일 이미지 연결 시험에 재사용"
    deepfake_video_frame_count: int = 16
    deepfake_video_minimum_valid_frames: int = 4
    deepfake_video_threshold_status: str = "research_only_unapproved"
    deepfake_video_threshold_source: str = (
        "Celeb-DF-v2 Validation에서 선택한 영상 16프레임 평균 연구 기준값"
    )

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold는 -1과 1 사이여야 합니다.")
        if not -1.0 <= self.retrieval_similarity_threshold <= 1.0:
            raise ValueError("retrieval_similarity_threshold는 -1과 1 사이여야 합니다.")
        if self.retrieval_similarity_threshold > self.similarity_threshold:
            raise ValueError(
                "후보 수집 기준값은 최종 동일인 기준값보다 높을 수 없습니다."
            )
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
        if self.maximum_video_bytes <= 0 or self.maximum_video_seconds <= 0:
            raise ValueError("영상 크기와 길이 제한은 양수여야 합니다.")
        minimum_video_request_bytes = (
            self.maximum_video_bytes
            + self.maximum_reference_images * self.maximum_image_bytes
        )
        if self.maximum_video_request_bytes < minimum_video_request_bytes:
            raise ValueError(
                "영상 요청 본문 제한은 영상과 최대 등록 사진 크기의 합 이상이어야 합니다."
            )
        if not 0.0 <= self.minimum_detection_score <= 1.0:
            raise ValueError("minimum_detection_score는 0과 1 사이여야 합니다.")
        if not 0.0 < self.minimum_face_area_ratio <= 1.0:
            raise ValueError("minimum_face_area_ratio는 0보다 크고 1 이하여야 합니다.")
        if self.maximum_search_candidates <= 0:
            raise ValueError("maximum_search_candidates는 양수여야 합니다.")
        if self.search_provider_timeout_seconds <= 0:
            raise ValueError("search_provider_timeout_seconds는 양수여야 합니다.")
        if self.maximum_pipeline_candidates <= 0:
            raise ValueError("maximum_pipeline_candidates는 양수여야 합니다.")
        if self.candidate_download_timeout_seconds <= 0:
            raise ValueError("candidate_download_timeout_seconds는 양수여야 합니다.")
        if self.candidate_download_maximum_redirects < 0:
            raise ValueError(
                "candidate_download_maximum_redirects는 0 이상이어야 합니다."
            )
        if self.exposure_enrollment_ttl_seconds <= 0:
            raise ValueError("exposure_enrollment_ttl_seconds는 양수여야 합니다.")
        if self.exposure_scan_ttl_seconds <= 0:
            raise ValueError("exposure_scan_ttl_seconds는 양수여야 합니다.")
        if self.maximum_exposure_enrollments <= 0 or self.maximum_exposure_scans <= 0:
            raise ValueError("비동기 저장소 항목 수는 양수여야 합니다.")
        if self.deepfake_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("deepfake_device는 auto, cpu, cuda 중 하나여야 합니다.")
        if self.deepfake_input_size <= 0 or self.deepfake_aligned_face_size <= 0:
            raise ValueError("딥페이크 모델 이미지 크기는 양수여야 합니다.")
        if not 0.0 <= self.deepfake_threshold <= 1.0:
            raise ValueError("deepfake_threshold는 0과 1 사이여야 합니다.")
        if not (
            1
            <= self.deepfake_video_minimum_valid_frames
            <= self.deepfake_video_frame_count
        ):
            raise ValueError("영상 유효 프레임 수 설정의 순서가 올바르지 않습니다.")
        if len(self.deepfake_model_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.deepfake_model_sha256.lower()
        ):
            raise ValueError("deepfake_model_sha256은 64자리 16진수여야 합니다.")

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
            retrieval_similarity_threshold=float(
                os.environ.get(
                    "FACEGUARD_RETRIEVAL_SIMILARITY_THRESHOLD",
                    RESEARCH_RETRIEVAL_THRESHOLD,
                )
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
            maximum_video_bytes=int(
                os.environ.get("FACEGUARD_MAX_VIDEO_BYTES", str(50 * 1024 * 1024))
            ),
            maximum_video_seconds=float(
                os.environ.get("FACEGUARD_MAX_VIDEO_SECONDS", "120")
            ),
            maximum_video_request_bytes=int(
                os.environ.get(
                    "FACEGUARD_MAX_VIDEO_REQUEST_BYTES", str(91 * 1024 * 1024)
                )
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
            maximum_pipeline_candidates=int(
                os.environ.get("FACEGUARD_MAXIMUM_PIPELINE_CANDIDATES", "10")
            ),
            candidate_download_timeout_seconds=float(
                os.environ.get("FACEGUARD_CANDIDATE_DOWNLOAD_TIMEOUT_SECONDS", "4.0")
            ),
            candidate_download_maximum_redirects=int(
                os.environ.get("FACEGUARD_CANDIDATE_DOWNLOAD_MAXIMUM_REDIRECTS", "2")
            ),
            exposure_enrollment_ttl_seconds=int(
                os.environ.get("FACEGUARD_EXPOSURE_ENROLLMENT_TTL_SECONDS", "1800")
            ),
            exposure_scan_ttl_seconds=int(
                os.environ.get("FACEGUARD_EXPOSURE_SCAN_TTL_SECONDS", "3600")
            ),
            maximum_exposure_enrollments=int(
                os.environ.get("FACEGUARD_MAXIMUM_EXPOSURE_ENROLLMENTS", "100")
            ),
            maximum_exposure_scans=int(
                os.environ.get("FACEGUARD_MAXIMUM_EXPOSURE_SCANS", "100")
            ),
            deepfake_model_name=os.environ.get(
                "FACEGUARD_DEEPFAKE_MODEL_NAME", "efficientnet_b4_celebdf_v2"
            ),
            deepfake_model_path=Path(
                os.environ.get(
                    "FACEGUARD_DEEPFAKE_MODEL_PATH",
                    ".models/deepfake/efficientnet_b4.onnx",
                )
            ).expanduser(),
            deepfake_model_sha256=os.environ.get(
                "FACEGUARD_DEEPFAKE_MODEL_SHA256", DEEPFAKE_MODEL_SHA256
            )
            .strip()
            .lower(),
            deepfake_calibration_path=Path(
                os.environ.get(
                    "FACEGUARD_DEEPFAKE_CALIBRATION_PATH",
                    ".models/deepfake/deepfake_video_calibration.json",
                )
            ).expanduser(),
            deepfake_device=os.environ.get("FACEGUARD_DEEPFAKE_DEVICE", "cpu")
            .strip()
            .lower(),
            deepfake_input_size=int(
                os.environ.get("FACEGUARD_DEEPFAKE_INPUT_SIZE", "380")
            ),
            deepfake_aligned_face_size=int(
                os.environ.get("FACEGUARD_DEEPFAKE_ALIGNED_FACE_SIZE", "224")
            ),
            deepfake_threshold=float(
                os.environ.get(
                    "FACEGUARD_DEEPFAKE_THRESHOLD", DEEPFAKE_RESEARCH_THRESHOLD
                )
            ),
            deepfake_threshold_status=os.environ.get(
                "FACEGUARD_DEEPFAKE_THRESHOLD_STATUS",
                "research_only_single_image_unvalidated",
            ),
            deepfake_video_frame_count=int(
                os.environ.get("FACEGUARD_DEEPFAKE_VIDEO_FRAME_COUNT", "16")
            ),
            deepfake_video_minimum_valid_frames=int(
                os.environ.get("FACEGUARD_DEEPFAKE_VIDEO_MINIMUM_VALID_FRAMES", "4")
            ),
            deepfake_video_threshold_status=os.environ.get(
                "FACEGUARD_DEEPFAKE_VIDEO_THRESHOLD_STATUS",
                "research_only_unapproved",
            ),
        )
