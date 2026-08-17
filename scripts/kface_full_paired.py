#!/usr/bin/env python3
"""K-FACE 저·중화질 1:1 이미지 전체를 Mac에서 중단 재개 처리한다.

중화질에서 SCRFD 얼굴 위치를 한 번만 찾고, 정확히 같은 촬영의 저화질
이미지에는 크기 비율로 랜드마크를 옮긴다. 정렬된 저·중화질 얼굴은 ArcFace
인식 모델에 배치로 넣는다. 원본·정렬 얼굴은 저장하지 않고, 비공개 data/
아래에 임베딩·품질값·익명 인덱스만 구간별로 저장한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kface_pilot import (
    _image_infos,
    _normalized_member,
    _sha256_file,
    list_subject_archives,
    validate_archive_size,
)

PIPELINE_VERSION = "kface-full-paired-v2"
QUALITY_COLUMNS = 6
EMBEDDING_DIMENSIONS = 512


@dataclass(frozen=True)
class PairedChunkResult:
    image_indices: np.ndarray
    low_embeddings: np.ndarray
    medium_embeddings: np.ndarray
    low_quality: np.ndarray
    medium_quality: np.ndarray
    reject_reasons: dict[str, int]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (EMBEDDING_DIMENSIONS,):
        raise ValueError("ArcFace 출력은 (N, 512) 형식이어야 합니다.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if len(array) and (not np.all(np.isfinite(array)) or np.any(norms <= 0)):
        raise ValueError("ArcFace 출력에 유한하지 않거나 0인 벡터가 있습니다.")
    return array / norms if len(array) else array


def _empty_chunk(reject_reasons: Counter[str] | None = None) -> PairedChunkResult:
    return PairedChunkResult(
        image_indices=np.empty((0,), dtype=np.int32),
        low_embeddings=np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32),
        medium_embeddings=np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32),
        low_quality=np.empty((0, QUALITY_COLUMNS), dtype=np.float32),
        medium_quality=np.empty((0, QUALITY_COLUMNS), dtype=np.float32),
        reject_reasons=dict(sorted((reject_reasons or Counter()).items())),
    )


def _validate_chunk(result: PairedChunkResult) -> None:
    count = len(result.image_indices)
    if result.image_indices.shape != (count,):
        raise ValueError("이미지 인덱스 형식이 올바르지 않습니다.")
    if len(np.unique(result.image_indices)) != count:
        raise ValueError("이미지 인덱스가 중복됐습니다.")
    for label, values, shape in (
        ("저화질 임베딩", result.low_embeddings, (count, EMBEDDING_DIMENSIONS)),
        ("중화질 임베딩", result.medium_embeddings, (count, EMBEDDING_DIMENSIONS)),
        ("저화질 품질값", result.low_quality, (count, QUALITY_COLUMNS)),
        ("중화질 품질값", result.medium_quality, (count, QUALITY_COLUMNS)),
    ):
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{label} 형식이 올바르지 않습니다.")


def _atomic_chunk(path: Path, result: PairedChunkResult) -> None:
    _validate_chunk(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        # 임베딩은 난수에 가까워 압축 효과가 작다. 전체 처리 속도를 위해
        # 압축하지 않고 float32를 그대로 저장한다.
        np.savez(
            handle,
            image_indices=result.image_indices.astype(np.int32, copy=False),
            low_embeddings=result.low_embeddings.astype(np.float32, copy=False),
            medium_embeddings=result.medium_embeddings.astype(np.float32, copy=False),
            low_quality=result.low_quality.astype(np.float32, copy=False),
            medium_quality=result.medium_quality.astype(np.float32, copy=False),
        )
    os.replace(temporary, path)


def _completed_checkpoint(
    path: Path, *, config_fingerprint: str, chunk_path: Path
) -> dict[str, Any] | None:
    if not path.is_file() or not chunk_path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if (
        payload.get("complete") is True
        and payload.get("config_fingerprint") == config_fingerprint
    ):
        return payload
    return None


def _face_quality(image: np.ndarray, bbox: np.ndarray, detection_score: float) -> list[float]:
    import cv2

    height, width = image.shape[:2]
    x1 = max(0, min(width, int(np.floor(bbox[0]))))
    y1 = max(0, min(height, int(np.floor(bbox[1]))))
    x2 = max(0, min(width, int(np.ceil(bbox[2]))))
    y2 = max(0, min(height, int(np.ceil(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("얼굴 영역이 올바르지 않습니다.")
    region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return [
        detection_score,
        float(((x2 - x1) * (y2 - y1)) / (width * height)),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        float(gray.mean()),
        float(width),
        float(height),
    ]


class PairedInsightFaceEncoder:
    """중화질 검출을 저화질에 공유하고 ArcFace를 배치 실행한다."""

    def __init__(
        self,
        *,
        model_root: Path,
        model_name: str,
        detection_size: int,
        minimum_detection_score: float,
        minimum_face_area_ratio: float,
        recognition_batch_size: int | None,
        recognition_provider: str,
        fast_detection_size: int | None,
        fast_detection_score: float,
    ) -> None:
        if detection_size <= 0:
            raise ValueError("검출 크기는 양수여야 합니다.")
        if recognition_batch_size is not None and recognition_batch_size <= 0:
            raise ValueError("인식 배치 크기는 양수여야 합니다.")
        if recognition_provider not in {"auto", "cpu", "coreml"}:
            raise ValueError("인식 실행 장치는 auto, cpu, coreml 중 하나여야 합니다.")
        if fast_detection_size is not None and (
            fast_detection_size <= 0 or fast_detection_size % 32
        ):
            raise ValueError("빠른 검출 크기는 32의 배수여야 합니다.")
        if not 0 <= fast_detection_score <= 1:
            raise ValueError("빠른 검출 점수는 0과 1 사이여야 합니다.")
        if not 0 <= minimum_detection_score <= 1:
            raise ValueError("최소 검출 점수는 0과 1 사이여야 합니다.")
        if not 0 < minimum_face_area_ratio <= 1:
            raise ValueError("최소 얼굴 면적 비율은 0보다 크고 1 이하여야 합니다.")
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        application = FaceAnalysis(
            name=model_name,
            root=str(model_root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        application.prepare(ctx_id=-1, det_size=(detection_size, detection_size))
        from faceguard_api.engine import InsightFaceEncoder

        self.application = application
        recognition = application.models["recognition"]
        use_coreml = recognition_provider == "coreml" or (
            recognition_provider == "auto"
            and "CoreMLExecutionProvider" in available
        )
        if recognition_provider == "coreml" and "CoreMLExecutionProvider" not in available:
            raise RuntimeError("CoreMLExecutionProvider를 사용할 수 없습니다.")
        if use_coreml:
            recognition.session = ort.InferenceSession(
                recognition.model_file,
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
            self.recognition_provider = "CoreMLExecutionProvider"
            default_recognition_batch_size = 1
        else:
            self.recognition_provider = "CPUExecutionProvider"
            default_recognition_batch_size = 4
        self.provider = (
            f"CPUDetection+{self.recognition_provider.removesuffix('ExecutionProvider')}Recognition"
        )
        self.available_providers = tuple(available)
        self.model_fingerprint = InsightFaceEncoder._fingerprint_loaded_models(
            application
        )
        self.model_name = model_name
        self.detection_size = detection_size
        self.minimum_detection_score = minimum_detection_score
        self.minimum_face_area_ratio = minimum_face_area_ratio
        self.fast_detection_size = fast_detection_size
        self.fast_detection_score = fast_detection_score
        self.recognition_batch_size = (
            recognition_batch_size
            if recognition_batch_size is not None
            else default_recognition_batch_size
        )

    @staticmethod
    def _decode(payload: bytes) -> np.ndarray:
        import cv2

        if not payload:
            raise ValueError("빈 이미지입니다.")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("이미지를 읽지 못했습니다.")
        return image

    def _recognize_aligned_faces(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        if not faces:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        recognition = self.application.models["recognition"]
        batches = []
        for start in range(0, len(faces), self.recognition_batch_size):
            batches.append(
                recognition.get_feat(
                    list(faces[start : start + self.recognition_batch_size])
                )
            )
        return np.concatenate(batches, axis=0)

    def encode_payload_pairs(
        self,
        payload_pairs: Sequence[tuple[bytes, bytes]],
        image_indices: Sequence[int],
    ) -> PairedChunkResult:
        if len(payload_pairs) != len(image_indices):
            raise ValueError("이미지 쌍과 인덱스 수가 다릅니다.")
        if not payload_pairs:
            return _empty_chunk()
        from insightface.utils import face_align

        accepted_indices: list[int] = []
        low_quality: list[list[float]] = []
        medium_quality: list[list[float]] = []
        aligned_faces: list[np.ndarray] = []
        recognition_jobs: list[Future[np.ndarray]] = []
        executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="coreml-arcface")
            if self.recognition_provider == "CoreMLExecutionProvider"
            else None
        )
        reject_reasons: Counter[str] = Counter()
        try:
            for image_index, (low_payload, medium_payload) in zip(
                image_indices, payload_pairs
            ):
                try:
                    low = self._decode(low_payload)
                    medium = self._decode(medium_payload)
                    if self.fast_detection_size is None:
                        bboxes, landmarks = self.application.det_model.detect(
                            medium, max_num=1
                        )
                    else:
                        fast_size = (self.fast_detection_size,) * 2
                        bboxes, landmarks = self.application.det_model.detect(
                            medium, input_size=fast_size, max_num=1
                        )
                        fast_score = float(bboxes[0, 4]) if len(bboxes) else 0.0
                        if landmarks is None or fast_score < self.fast_detection_score:
                            bboxes, landmarks = self.application.det_model.detect(
                                medium, max_num=1
                            )
                    if len(bboxes) != 1 or landmarks is None:
                        reject_reasons["NO_FACE"] += 1
                        continue
                    bbox = np.asarray(bboxes[0, :4], dtype=np.float32)
                    detection_score = float(bboxes[0, 4])
                    if detection_score < self.minimum_detection_score:
                        reject_reasons["LOW_DETECTION_SCORE"] += 1
                        continue
                    medium_item_quality = _face_quality(
                        medium, bbox, detection_score
                    )
                    if medium_item_quality[1] < self.minimum_face_area_ratio:
                        reject_reasons["FACE_TOO_SMALL"] += 1
                        continue
                    medium_landmarks = np.asarray(landmarks[0], dtype=np.float32)
                    if medium_landmarks.shape != (5, 2) or not np.all(
                        np.isfinite(medium_landmarks)
                    ):
                        reject_reasons["INVALID_FACE_LANDMARKS"] += 1
                        continue
                    scale = np.asarray(
                        [
                            low.shape[1] / medium.shape[1],
                            low.shape[0] / medium.shape[0],
                        ],
                        dtype=np.float32,
                    )
                    low_landmarks = medium_landmarks * scale
                    low_bbox = bbox * np.asarray(
                        [scale[0], scale[1], scale[0], scale[1]], dtype=np.float32
                    )
                    low_item_quality = _face_quality(
                        low, low_bbox, detection_score
                    )
                    aligned_low = face_align.norm_crop(
                        low, landmark=low_landmarks, image_size=112
                    )
                    aligned_medium = face_align.norm_crop(
                        medium, landmark=medium_landmarks, image_size=112
                    )
                    accepted_indices.append(int(image_index))
                    low_quality.append(low_item_quality)
                    medium_quality.append(medium_item_quality)
                    aligned_faces.extend([aligned_low, aligned_medium])
                    if executor is not None and len(aligned_faces) >= 64:
                        recognition_jobs.append(
                            executor.submit(
                                self._recognize_aligned_faces, aligned_faces[:64]
                            )
                        )
                        del aligned_faces[:64]
                except (OSError, TypeError, ValueError):
                    reject_reasons["INVALID_IMAGE_OR_ALIGNMENT"] += 1

            if not accepted_indices:
                return _empty_chunk(reject_reasons)
            if executor is not None:
                if aligned_faces:
                    recognition_jobs.append(
                        executor.submit(
                            self._recognize_aligned_faces, aligned_faces.copy()
                        )
                    )
                    aligned_faces.clear()
                embeddings = _unit_rows(
                    np.concatenate([job.result() for job in recognition_jobs], axis=0)
                )
            else:
                embeddings = _unit_rows(
                    self._recognize_aligned_faces(aligned_faces)
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        if len(embeddings) != len(accepted_indices) * 2:
            raise RuntimeError("ArcFace 배치 출력 수가 얼굴 쌍 수와 다릅니다.")
        return PairedChunkResult(
            image_indices=np.asarray(accepted_indices, dtype=np.int32),
            low_embeddings=embeddings[0::2],
            medium_embeddings=embeddings[1::2],
            low_quality=np.asarray(low_quality, dtype=np.float32),
            medium_quality=np.asarray(medium_quality, dtype=np.float32),
            reject_reasons=dict(sorted(reject_reasons.items())),
        )


def _subject_output_paths(
    output_dir: Path, pseudonym: str, chunk_index: int
) -> tuple[Path, Path]:
    root = output_dir / "subjects" / pseudonym
    stem = f"chunk_{chunk_index:05d}"
    return root / "chunks" / f"{stem}.npz", root / "checkpoints" / f"{stem}.json"


def _validate_subject_pair(low_subject: Any, medium_subject: Any) -> None:
    if low_subject.pseudonym != medium_subject.pseudonym:
        raise ValueError("저·중화질 인물 ZIP 순서가 일치하지 않습니다.")


def process_paired_archives(
    low_archive: Path,
    medium_archive: Path,
    *,
    output_dir: Path,
    subject_start: int,
    subject_end: int,
    chunk_size: int,
    encoder: Any,
    maximum_pairs_per_subject: int | None = None,
    low_archive_sha256: str | None = None,
    medium_archive_sha256: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if subject_start <= 0 or subject_end < subject_start or chunk_size <= 0:
        raise ValueError("인물 범위와 chunk 크기를 확인하세요.")
    if maximum_pairs_per_subject is not None and maximum_pairs_per_subject <= 0:
        raise ValueError("인물별 최대 이미지 쌍 수는 양수여야 합니다.")
    low_subjects = list_subject_archives(low_archive)
    medium_subjects = list_subject_archives(medium_archive)
    if len(low_subjects) != len(medium_subjects):
        raise ValueError("저·중화질 인물 수가 다릅니다.")
    if subject_end > len(low_subjects):
        raise ValueError("요청한 마지막 인물이 데이터 범위를 벗어났습니다.")
    for low_subject, medium_subject in zip(low_subjects, medium_subjects):
        _validate_subject_pair(low_subject, medium_subject)

    low_sha256 = low_archive_sha256 or _sha256_file(low_archive)
    medium_sha256 = medium_archive_sha256 or _sha256_file(medium_archive)
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "low_archive_sha256": low_sha256,
        "medium_archive_sha256": medium_sha256,
        "chunk_size": chunk_size,
        "model_name": getattr(encoder, "model_name", None),
        "model_fingerprint": getattr(encoder, "model_fingerprint", None),
        "detection_size": getattr(encoder, "detection_size", None),
        "fast_detection_size": getattr(encoder, "fast_detection_size", None),
        "fast_detection_score": getattr(encoder, "fast_detection_score", None),
        "minimum_detection_score": getattr(
            encoder, "minimum_detection_score", None
        ),
        "minimum_face_area_ratio": getattr(
            encoder, "minimum_face_area_ratio", None
        ),
        "recognition_batch_size": getattr(
            encoder, "recognition_batch_size", None
        ),
        "recognition_provider": getattr(encoder, "recognition_provider", None),
        "maximum_pairs_per_subject": maximum_pairs_per_subject,
    }
    config_fingerprint = _config_fingerprint(config)
    totals = Counter()
    reject_reasons: Counter[str] = Counter()
    started = time.perf_counter()
    selected_low_subjects = low_subjects[subject_start - 1 : subject_end]
    selected_medium_subjects = medium_subjects[subject_start - 1 : subject_end]

    with zipfile.ZipFile(low_archive) as low_outer, zipfile.ZipFile(
        medium_archive
    ) as medium_outer:
        for local_position, (low_subject, medium_subject) in enumerate(
            zip(selected_low_subjects, selected_medium_subjects), start=subject_start
        ):
            subject_started = time.perf_counter()
            low_nested = low_outer.read(low_subject.outer_member)
            medium_nested = medium_outer.read(medium_subject.outer_member)
            with zipfile.ZipFile(BytesIO(low_nested)) as low_inner, zipfile.ZipFile(
                BytesIO(medium_nested)
            ) as medium_inner:
                low_infos = _image_infos(low_inner)
                medium_infos = _image_infos(medium_inner)
                low_names = [_normalized_member(item.filename) for item in low_infos]
                medium_names = [
                    _normalized_member(item.filename) for item in medium_infos
                ]
                if low_names != medium_names:
                    raise ValueError(
                        "저·중화질 내부 이미지 경로가 1:1로 일치하지 않습니다."
                    )
                if maximum_pairs_per_subject is not None:
                    low_infos = low_infos[:maximum_pairs_per_subject]
                    medium_infos = medium_infos[:maximum_pairs_per_subject]
                totals["selected_images"] += len(low_infos) * 2
                chunks = math.ceil(len(low_infos) / chunk_size)
                for chunk_index in range(chunks):
                    start = chunk_index * chunk_size
                    end = min(len(low_infos), start + chunk_size)
                    chunk_path, checkpoint_path = _subject_output_paths(
                        output_dir, low_subject.pseudonym, chunk_index
                    )
                    completed = _completed_checkpoint(
                        checkpoint_path,
                        config_fingerprint=config_fingerprint,
                        chunk_path=chunk_path,
                    )
                    if completed is not None:
                        totals["skipped_chunks"] += 1
                        totals["accepted_pairs"] += int(
                            completed["accepted_pairs"]
                        )
                        totals["rejected_pairs"] += int(
                            completed["rejected_pairs"]
                        )
                        reject_reasons.update(completed.get("reject_reasons", {}))
                        if progress:
                            progress(
                                {
                                    "subject": local_position,
                                    "subject_end": subject_end,
                                    "chunk": chunk_index + 1,
                                    "chunks": chunks,
                                    "status": "skipped",
                                }
                            )
                        continue
                    chunk_started = time.perf_counter()
                    payload_pairs = [
                        (
                            low_inner.read(low_infos[index]),
                            medium_inner.read(medium_infos[index]),
                        )
                        for index in range(start, end)
                    ]
                    result = encoder.encode_payload_pairs(
                        payload_pairs, list(range(start, end))
                    )
                    _atomic_chunk(chunk_path, result)
                    accepted_pairs = len(result.image_indices)
                    rejected_pairs = end - start - accepted_pairs
                    checkpoint = {
                        "complete": True,
                        "config_fingerprint": config_fingerprint,
                        "image_pair_start": start,
                        "image_pair_end": end,
                        "selected_pairs": end - start,
                        "accepted_pairs": accepted_pairs,
                        "rejected_pairs": rejected_pairs,
                        "reject_reasons": result.reject_reasons,
                        "elapsed_seconds": time.perf_counter() - chunk_started,
                        "contains_raw_path": False,
                        "contains_face_image": False,
                    }
                    _atomic_json(checkpoint_path, checkpoint)
                    totals["processed_chunks"] += 1
                    totals["accepted_pairs"] += accepted_pairs
                    totals["rejected_pairs"] += rejected_pairs
                    reject_reasons.update(result.reject_reasons)
                    if progress:
                        elapsed = time.perf_counter() - chunk_started
                        progress(
                            {
                                "subject": local_position,
                                "subject_end": subject_end,
                                "chunk": chunk_index + 1,
                                "chunks": chunks,
                                "status": "processed",
                                "accepted_pairs": accepted_pairs,
                                "rejected_pairs": rejected_pairs,
                                "pairs_per_second": round((end - start) / elapsed, 3),
                            }
                        )
            if progress:
                progress(
                    {
                        "subject": local_position,
                        "subject_end": subject_end,
                        "status": "subject_complete",
                        "elapsed_seconds": round(
                            time.perf_counter() - subject_started, 3
                        ),
                    }
                )

    summary = {
        "dataset": "K-FACE",
        "pipeline_version": PIPELINE_VERSION,
        "config_fingerprint": config_fingerprint,
        "subject_start": subject_start,
        "subject_end": subject_end,
        "subject_count": subject_end - subject_start + 1,
        "maximum_pairs_per_subject": maximum_pairs_per_subject,
        "full_subject_images_selected": maximum_pairs_per_subject is None,
        "selected_images": int(totals["selected_images"]),
        "accepted_pairs": int(totals["accepted_pairs"]),
        "accepted_images": int(totals["accepted_pairs"] * 2),
        "rejected_pairs": int(totals["rejected_pairs"]),
        "processed_chunks": int(totals["processed_chunks"]),
        "skipped_chunks": int(totals["skipped_chunks"]),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "processing_seconds": time.perf_counter() - started,
        "model_provider": getattr(encoder, "provider", None),
        "model_fingerprint": getattr(encoder, "model_fingerprint", None),
        "contains_raw_paths": False,
        "contains_subject_identifiers": False,
        "contains_face_images": False,
        "embeddings_are_private": True,
    }
    shard_path = output_dir / "shards" / f"subjects_{subject_start:03d}_{subject_end:03d}.json"
    _atomic_json(shard_path, summary)
    return summary


def _progress_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-archive", type=Path, required=True)
    parser.add_argument("--medium-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-start", type=int, required=True)
    parser.add_argument("--subject-end", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--maximum-pairs-per-subject",
        type=int,
        help="연결 시험 전용 제한입니다. 전체 실행에서는 지정하지 마세요.",
    )
    parser.add_argument("--model-root", type=Path, default=Path(".models/insightface"))
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--detection-size", type=int, default=160)
    parser.add_argument(
        "--fast-detection-size",
        type=int,
        help="이 크기로 먼저 검출하고 실패하면 --detection-size로 재시도합니다.",
    )
    parser.add_argument("--fast-detection-score", type=float, default=0.50)
    parser.add_argument("--minimum-detection-score", type=float, default=0.50)
    parser.add_argument("--minimum-face-area-ratio", type=float, default=0.01)
    parser.add_argument("--recognition-batch-size", type=int)
    parser.add_argument(
        "--recognition-provider",
        choices=("auto", "cpu", "coreml"),
        default="auto",
        help="auto는 Mac에서 Core ML, 그 외 환경에서 CPU를 선택합니다.",
    )
    parser.add_argument("--expected-low-bytes", type=int)
    parser.add_argument("--expected-medium-bytes", type=int)
    parser.add_argument("--low-archive-sha256")
    parser.add_argument("--medium-archive-sha256")
    parser.add_argument("--accept-noncommercial-model-license", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.low_archive, args.medium_archive):
        if not path.is_file():
            raise FileNotFoundError(f"K-FACE ZIP을 찾지 못했습니다: {path}")
    validate_archive_size(args.low_archive, args.expected_low_bytes)
    validate_archive_size(args.medium_archive, args.expected_medium_bytes)
    if not args.accept_noncommercial_model_license:
        raise PermissionError(
            "InsightFace 제공 가중치의 비상업 연구 조건을 확인한 뒤 "
            "--accept-noncommercial-model-license를 지정하세요."
        )
    encoder = PairedInsightFaceEncoder(
        model_root=args.model_root,
        model_name=args.model_name,
        detection_size=args.detection_size,
        minimum_detection_score=args.minimum_detection_score,
        minimum_face_area_ratio=args.minimum_face_area_ratio,
        recognition_batch_size=args.recognition_batch_size,
        recognition_provider=args.recognition_provider,
        fast_detection_size=args.fast_detection_size,
        fast_detection_score=args.fast_detection_score,
    )
    result = process_paired_archives(
        args.low_archive,
        args.medium_archive,
        output_dir=args.output_dir,
        subject_start=args.subject_start,
        subject_end=args.subject_end,
        chunk_size=args.chunk_size,
        encoder=encoder,
        maximum_pairs_per_subject=args.maximum_pairs_per_subject,
        low_archive_sha256=args.low_archive_sha256,
        medium_archive_sha256=args.medium_archive_sha256,
        progress=_progress_line,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
