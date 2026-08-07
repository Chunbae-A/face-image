"""웹 프레임워크와 모델 라이브러리에 독립적인 동일인 비교 로직."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceQuality:
    detection_score: float
    face_area_ratio: float
    blur_score: float
    brightness_mean: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class EncodedFace:
    embedding: np.ndarray
    quality: FaceQuality


@dataclass(frozen=True)
class AlignedEncodedFace:
    """동일인 임베딩과 딥페이크 모델용 정렬 얼굴을 함께 보관한다."""

    face: EncodedFace
    aligned_bgr: np.ndarray


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("임베딩은 유한한 값을 가진 1차원 벡터여야 합니다.")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("임베딩 크기는 0보다 커야 합니다.")
    return value / norm


def pool_reference_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("등록 임베딩이 하나 이상 필요합니다.")
    normalized = [l2_normalize(embedding) for embedding in embeddings]
    dimensions = {embedding.shape for embedding in normalized}
    if len(dimensions) != 1:
        raise ValueError("등록 임베딩 차원이 서로 다릅니다.")
    return l2_normalize(np.mean(np.stack(normalized), axis=0))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_normalized = l2_normalize(left)
    right_normalized = l2_normalize(right)
    if left_normalized.shape != right_normalized.shape:
        raise ValueError("비교할 임베딩 차원이 서로 다릅니다.")
    return float(np.clip(np.dot(left_normalized, right_normalized), -1.0, 1.0))
