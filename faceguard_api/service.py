"""얼굴 등록 사진 여러 장과 확인 사진 한 장을 비교하는 서비스."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .domain import EncodedFace, cosine_similarity, pool_reference_embeddings
from .errors import FaceGuardError
from .settings import Settings


class FaceEncoder(Protocol):
    @property
    def loaded(self) -> bool: ...

    @property
    def provider(self) -> str | None: ...

    @property
    def model_fingerprint(self) -> str | None: ...

    def encode(self, payload: bytes) -> EncodedFace: ...


@dataclass(frozen=True)
class Verification:
    is_same_person: bool
    similarity: float
    threshold: float
    threshold_status: str
    threshold_source: str
    reference_count: int
    reference_faces: Sequence[EncodedFace]
    query_face: EncodedFace
    processing_ms: float


class FaceGuardService:
    def __init__(self, settings: Settings, encoder: FaceEncoder) -> None:
        self.settings = settings
        self.encoder = encoder

    def verify(self, references: Sequence[bytes], query: bytes) -> Verification:
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

        started = time.perf_counter()
        reference_faces = [self.encoder.encode(payload) for payload in references]
        query_face = self.encoder.encode(query)
        reference_embedding = pool_reference_embeddings(
            [face.embedding for face in reference_faces]
        )
        similarity = cosine_similarity(reference_embedding, query_face.embedding)
        return Verification(
            is_same_person=similarity >= self.settings.similarity_threshold,
            similarity=similarity,
            threshold=self.settings.similarity_threshold,
            threshold_status=self.settings.threshold_status,
            threshold_source=self.settings.threshold_source,
            reference_count=len(reference_faces),
            reference_faces=reference_faces,
            query_face=query_face,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )
