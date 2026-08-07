"""짧은 영상의 대표 프레임을 Celeb-DF 연구 모델로 분석한다."""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .deepfake import DeepfakeAnalysis
from .domain import (
    AlignedEncodedFace,
    EncodedFace,
    FaceQuality,
    cosine_similarity,
    pool_reference_embeddings,
)
from .errors import FaceGuardError, ModelUnavailableError
from .settings import Settings


@dataclass(frozen=True)
class DecodedVideoFrame:
    frame_index: int
    timestamp_seconds: float
    bgr: np.ndarray


@dataclass(frozen=True)
class DecodedVideo:
    duration_seconds: float
    fps: float
    total_frame_count: int
    requested_frame_count: int
    frames: tuple[DecodedVideoFrame, ...]


@dataclass(frozen=True)
class VideoFrameAnalysis:
    frame_index: int
    timestamp_seconds: float
    status: Literal["analyzed", "skipped"]
    deepfake_score: float | None
    is_suspected_deepfake: bool | None
    face_similarity: float | None
    quality: FaceQuality | None
    error_code: str | None
    processing_ms: float
    inference_ms: float | None


@dataclass(frozen=True)
class SuspiciousSegment:
    start_seconds: float
    end_seconds: float
    peak_score: float
    analyzed_frame_count: int


@dataclass(frozen=True)
class DeepfakeVideoAnalysis:
    status: Literal["completed", "partial_failed"]
    is_suspected_deepfake: bool
    video_score: float
    threshold: float
    aggregation: Literal["mean"]
    duration_seconds: float
    fps: float
    total_frame_count: int
    requested_frame_count: int
    decoded_frame_count: int
    analyzed_frame_count: int
    skipped_frame_count: int
    reference_count: int
    frames: tuple[VideoFrameAnalysis, ...]
    suspicious_segments: tuple[SuspiciousSegment, ...]
    processing_ms: float
    inference_ms: float
    model_name: str
    execution_provider: str
    model_fingerprint: str


class VideoFaceEncoder(Protocol):
    def encode(self, payload: bytes) -> EncodedFace: ...

    def encode_frame_faces(
        self,
        frame_bgr: np.ndarray,
        *,
        aligned_face_size: int,
    ) -> tuple[AlignedEncodedFace, ...]: ...


class AlignedDeepfakeAnalyzer(Protocol):
    def analyze_aligned(self, aligned: AlignedEncodedFace) -> DeepfakeAnalysis: ...


class VideoDecoder(Protocol):
    def __call__(self, payload: bytes, suffix: str) -> DecodedVideo: ...


def sample_video_frame_indices(frame_count: int, requested: int) -> list[int]:
    """학습 전처리와 같이 시작·끝 카드를 피해 균등한 프레임을 고른다."""

    if frame_count <= 0 or requested <= 0:
        return []
    if frame_count <= requested:
        return list(range(frame_count))
    first = min(frame_count - 1, max(0, round(frame_count * 0.08)))
    last = max(first, min(frame_count - 1, round(frame_count * 0.92) - 1))
    return sorted(
        {int(index) for index in np.linspace(first, last, num=requested, dtype=int)}
    )


class OpenCVVideoDecoder:
    """요청 영상을 임시 파일에서 읽고 컨텍스트 종료 시 즉시 삭제한다."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def __call__(self, payload: bytes, suffix: str) -> DecodedVideo:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - API 설치 오류
            raise ModelUnavailableError(
                "OpenCV 영상 라이브러리를 불러오지 못했습니다."
            ) from error

        with tempfile.TemporaryDirectory(prefix="faceguard-video-") as directory:
            video_path = Path(directory) / f"request{suffix}"
            video_path.write_bytes(payload)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FaceGuardError(
                    "INVALID_VIDEO", "손상되었거나 읽을 수 없는 영상입니다."
                )
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (
                    not math.isfinite(fps)
                    or fps <= 0.0
                    or frame_count <= 0
                    or width <= 0
                    or height <= 0
                ):
                    raise FaceGuardError(
                        "INVALID_VIDEO_METADATA",
                        "영상 길이·프레임 정보를 확인하지 못했습니다.",
                    )
                if width * height > self.settings.maximum_image_pixels:
                    raise FaceGuardError(
                        "VIDEO_FRAME_TOO_LARGE",
                        f"영상 프레임은 {self.settings.maximum_image_pixels} 픽셀 이하여야 합니다.",
                        413,
                    )
                duration_seconds = frame_count / fps
                if duration_seconds > self.settings.maximum_video_seconds:
                    raise FaceGuardError(
                        "VIDEO_TOO_LONG",
                        f"영상은 {self.settings.maximum_video_seconds:g}초 이하여야 합니다.",
                        413,
                    )

                indices = sample_video_frame_indices(
                    frame_count,
                    self.settings.deepfake_video_frame_count,
                )
                frames: list[DecodedVideoFrame] = []
                for frame_index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue
                    frames.append(
                        DecodedVideoFrame(
                            frame_index=frame_index,
                            timestamp_seconds=frame_index / fps,
                            bgr=np.ascontiguousarray(frame),
                        )
                    )
                if not frames:
                    raise FaceGuardError(
                        "VIDEO_DECODE_FAILED",
                        "영상에서 분석할 프레임을 읽지 못했습니다.",
                    )
                return DecodedVideo(
                    duration_seconds=duration_seconds,
                    fps=fps,
                    total_frame_count=frame_count,
                    requested_frame_count=len(indices),
                    frames=tuple(frames),
                )
            finally:
                capture.release()


def build_suspicious_segments(
    frames: Sequence[VideoFrameAnalysis],
    *,
    duration_seconds: float,
    requested_frame_count: int,
) -> tuple[SuspiciousSegment, ...]:
    """인접한 의심 프레임을 사용자가 확인할 대략적인 시간 구간으로 묶는다."""

    suspicious = [
        frame
        for frame in frames
        if frame.status == "analyzed"
        and frame.is_suspected_deepfake is True
        and frame.deepfake_score is not None
    ]
    if not suspicious:
        return ()
    half_window = duration_seconds / max(1, requested_frame_count) / 2.0
    windows = [
        (
            max(0.0, frame.timestamp_seconds - half_window),
            min(duration_seconds, frame.timestamp_seconds + half_window),
            float(frame.deepfake_score),
        )
        for frame in suspicious
    ]
    merged: list[list[float]] = []
    counts: list[int] = []
    for start, end, score in windows:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], score)
            counts[-1] += 1
        else:
            merged.append([start, end, score])
            counts.append(1)
    return tuple(
        SuspiciousSegment(
            start_seconds=start,
            end_seconds=end,
            peak_score=score,
            analyzed_frame_count=count,
        )
        for (start, end, score), count in zip(merged, counts, strict=True)
    )


class VideoDeepfakeAnalyzer:
    def __init__(
        self,
        settings: Settings,
        face_encoder: VideoFaceEncoder,
        deepfake_analyzer: AlignedDeepfakeAnalyzer,
        *,
        decoder: VideoDecoder | None = None,
    ) -> None:
        self.settings = settings
        self.face_encoder = face_encoder
        self.deepfake_analyzer = deepfake_analyzer
        self.decoder = decoder or OpenCVVideoDecoder(settings)

    def _reference_embedding(
        self, reference_payloads: Sequence[bytes]
    ) -> np.ndarray | None:
        if not reference_payloads:
            return None
        embeddings = [
            self.face_encoder.encode(payload).embedding
            for payload in reference_payloads
        ]
        return pool_reference_embeddings(embeddings)

    def _select_face(
        self,
        faces: Sequence[AlignedEncodedFace],
        template: np.ndarray | None,
    ) -> tuple[AlignedEncodedFace, float | None]:
        if not faces:
            raise FaceGuardError("NO_FACE", "영상 프레임에서 얼굴을 찾지 못했습니다.")
        if template is None:
            selected = max(
                faces,
                key=lambda item: item.face.quality.face_area_ratio,
            )
            return selected, None
        ranked = [
            (cosine_similarity(template, item.face.embedding), item) for item in faces
        ]
        similarity, selected = max(ranked, key=lambda item: item[0])
        if similarity < self.settings.retrieval_similarity_threshold:
            raise FaceGuardError(
                "REFERENCE_FACE_NOT_MATCHED",
                "영상 프레임에서 등록 얼굴과 비슷한 얼굴을 찾지 못했습니다.",
            )
        return selected, similarity

    def analyze(
        self,
        payload: bytes,
        *,
        suffix: str,
        reference_payloads: Sequence[bytes] = (),
    ) -> DeepfakeVideoAnalysis:
        started = time.perf_counter()
        decoded = self.decoder(payload, suffix)
        reference_template = self._reference_embedding(reference_payloads)
        tracking_embeddings: list[np.ndarray] = []
        frame_results: list[VideoFrameAnalysis] = []
        model_results: list[DeepfakeAnalysis] = []

        for decoded_frame in decoded.frames:
            frame_started = time.perf_counter()
            try:
                faces = self.face_encoder.encode_frame_faces(
                    decoded_frame.bgr,
                    aligned_face_size=self.settings.deepfake_aligned_face_size,
                )
                tracking_template = (
                    reference_template
                    if reference_template is not None
                    else (
                        pool_reference_embeddings(tracking_embeddings)
                        if tracking_embeddings
                        else None
                    )
                )
                selected, similarity = self._select_face(faces, tracking_template)
                result = self.deepfake_analyzer.analyze_aligned(selected)
                if reference_template is None:
                    tracking_embeddings.append(selected.face.embedding)
                model_results.append(result)
                frame_results.append(
                    VideoFrameAnalysis(
                        frame_index=decoded_frame.frame_index,
                        timestamp_seconds=decoded_frame.timestamp_seconds,
                        status="analyzed",
                        deepfake_score=result.deepfake_score,
                        is_suspected_deepfake=result.is_suspected_deepfake,
                        face_similarity=similarity,
                        quality=result.quality,
                        error_code=None,
                        processing_ms=(time.perf_counter() - frame_started) * 1000.0,
                        inference_ms=result.inference_ms,
                    )
                )
            except FaceGuardError as error:
                if error.code == "MODEL_UNAVAILABLE":
                    raise
                frame_results.append(
                    VideoFrameAnalysis(
                        frame_index=decoded_frame.frame_index,
                        timestamp_seconds=decoded_frame.timestamp_seconds,
                        status="skipped",
                        deepfake_score=None,
                        is_suspected_deepfake=None,
                        face_similarity=None,
                        quality=None,
                        error_code=error.code,
                        processing_ms=(time.perf_counter() - frame_started) * 1000.0,
                        inference_ms=None,
                    )
                )

        if len(model_results) < self.settings.deepfake_video_minimum_valid_frames:
            raise FaceGuardError(
                "INSUFFICIENT_VALID_VIDEO_FRAMES",
                "영상에서 분석 가능한 얼굴 프레임이 부족합니다. 더 밝고 얼굴이 크게 나온 영상을 사용하세요.",
            )
        video_score = float(
            np.mean([result.deepfake_score for result in model_results])
        )
        representative = model_results[0]
        suspicious_segments = build_suspicious_segments(
            frame_results,
            duration_seconds=decoded.duration_seconds,
            requested_frame_count=decoded.requested_frame_count,
        )
        skipped_count = decoded.requested_frame_count - len(model_results)
        return DeepfakeVideoAnalysis(
            status="partial_failed" if skipped_count else "completed",
            is_suspected_deepfake=video_score >= self.settings.deepfake_threshold,
            video_score=video_score,
            threshold=self.settings.deepfake_threshold,
            aggregation="mean",
            duration_seconds=decoded.duration_seconds,
            fps=decoded.fps,
            total_frame_count=decoded.total_frame_count,
            requested_frame_count=decoded.requested_frame_count,
            decoded_frame_count=len(decoded.frames),
            analyzed_frame_count=len(model_results),
            skipped_frame_count=skipped_count,
            reference_count=len(reference_payloads),
            frames=tuple(frame_results),
            suspicious_segments=suspicious_segments,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            inference_ms=float(sum(result.inference_ms for result in model_results)),
            model_name=representative.model_name,
            execution_provider=representative.execution_provider,
            model_fingerprint=representative.model_fingerprint,
        )
