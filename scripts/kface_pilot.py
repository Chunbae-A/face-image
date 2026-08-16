#!/usr/bin/env python3
"""K-FACE 저·중화질 ZIP을 작은 배치로 검증하고 ArcFace 특징을 추출한다.

원본 ZIP 안에 인물별 ZIP이 들어 있는 구조를 전제로 한다. 한 번에 인물 한 명의
ZIP만 메모리에 올리고, 정렬 얼굴이나 원본 이미지는 저장하지 않는다. 인물별 NPZ와
완료 JSON을 따로 남겨 중간에 종료돼도 완료한 인물부터 다시 시작하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_RESOLUTIONS = {"low", "medium"}
PIPELINE_VERSION = "kface-pilot-v1"


@dataclass(frozen=True)
class SubjectArchive:
    index: int
    outer_member: str
    pseudonym: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int


@dataclass(frozen=True)
class SubjectResult:
    pseudonym: str
    selected_images: int
    accepted_images: int
    rejected_images: int
    reject_reasons: dict[str, int]
    elapsed_seconds: float
    checkpoint: str | None


def _normalized_member(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _assert_safe_member(name: str) -> str:
    normalized = _normalized_member(name)
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("ZIP 내부에 안전하지 않은 경로가 있습니다.")
    return normalized


def _subject_key(member: str) -> str:
    normalized = _assert_safe_member(member)
    return PurePosixPath(normalized).stem.casefold()


def subject_pseudonym(member: str) -> str:
    digest = hashlib.sha256(
        f"{PIPELINE_VERSION}:{_subject_key(member)}".encode()
    ).hexdigest()
    return f"subject_{digest[:16]}"


def list_subject_archives(archive_path: Path) -> list[SubjectArchive]:
    rows: list[SubjectArchive] = []
    with zipfile.ZipFile(archive_path) as outer:
        infos = sorted(
            (
                info
                for info in outer.infolist()
                if not info.is_dir()
                and PurePosixPath(_normalized_member(info.filename)).suffix.casefold()
                == ".zip"
            ),
            key=lambda info: _normalized_member(info.filename).casefold(),
        )
        if not infos:
            raise ValueError("바깥 ZIP에서 인물별 내부 ZIP을 찾지 못했습니다.")
        seen_members: set[str] = set()
        seen_pseudonyms: set[str] = set()
        for index, info in enumerate(infos, start=1):
            normalized = _assert_safe_member(info.filename)
            if info.flag_bits & 0x1:
                raise ValueError("암호화된 내부 ZIP은 처리할 수 없습니다.")
            pseudonym = subject_pseudonym(normalized)
            if normalized in seen_members or pseudonym in seen_pseudonyms:
                raise ValueError("중복된 인물별 ZIP이 있습니다.")
            seen_members.add(normalized)
            seen_pseudonyms.add(pseudonym)
            rows.append(
                SubjectArchive(
                    index=index,
                    outer_member=info.filename,
                    pseudonym=pseudonym,
                    compressed_bytes=int(info.compress_size),
                    uncompressed_bytes=int(info.file_size),
                    crc32=int(info.CRC),
                )
            )
    return rows


def evenly_spaced(items: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0:
        raise ValueError("이미지 선택 수는 양수여야 합니다.")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def _image_infos(inner: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos: list[zipfile.ZipInfo] = []
    seen: set[str] = set()
    for info in inner.infolist():
        if info.is_dir():
            continue
        normalized = _assert_safe_member(info.filename)
        if PurePosixPath(normalized).suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        if info.flag_bits & 0x1:
            raise ValueError("암호화된 이미지 파일은 처리할 수 없습니다.")
        if normalized in seen:
            raise ValueError("내부 ZIP에 중복된 이미지 경로가 있습니다.")
        seen.add(normalized)
        infos.append(info)
    return sorted(infos, key=lambda info: _normalized_member(info.filename).casefold())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive_size(path: Path, expected_bytes: int | None) -> None:
    """중단된 브라우저 다운로드를 완성본으로 오인하지 않게 막는다."""

    if expected_bytes is None:
        return
    if expected_bytes <= 0:
        raise ValueError("예상 ZIP 크기는 양수여야 합니다.")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise OSError(
            f"ZIP 크기가 다릅니다: {actual_bytes:,} != {expected_bytes:,} bytes"
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def inventory_archive(
    archive_path: Path,
    *,
    resolution: str,
    inspect_subjects: int | None = None,
) -> dict[str, Any]:
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError("해상도는 low 또는 medium이어야 합니다.")
    subjects = list_subject_archives(archive_path)
    selected = subjects if inspect_subjects is None else subjects[:inspect_subjects]
    image_counts: list[int] = []
    nested_uncompressed = 0
    with zipfile.ZipFile(archive_path) as outer:
        for subject in selected:
            nested_payload = outer.read(subject.outer_member)
            nested_uncompressed += len(nested_payload)
            with zipfile.ZipFile(BytesIO(nested_payload)) as inner:
                image_counts.append(len(_image_infos(inner)))
    return {
        "dataset": "K-FACE",
        "resolution": resolution,
        "pipeline_version": PIPELINE_VERSION,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "subject_archive_count": len(subjects),
        "inspected_subject_count": len(selected),
        "inspected_image_count": sum(image_counts),
        "minimum_images_per_inspected_subject": min(image_counts) if image_counts else 0,
        "maximum_images_per_inspected_subject": max(image_counts) if image_counts else 0,
        "nested_zip_bytes_inspected": nested_uncompressed,
        "contains_raw_paths": False,
        "contains_face_images": False,
        "contains_embeddings": False,
    }


def _config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _subject_paths(output_dir: Path, pseudonym: str) -> tuple[Path, Path]:
    return (
        output_dir / "embeddings" / f"{pseudonym}.npz",
        output_dir / "checkpoints" / f"{pseudonym}.json",
    )


def _completed_checkpoint(path: Path, config_fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        payload.get("complete") is True
        and payload.get("config_fingerprint") == config_fingerprint
    ):
        return payload
    return None


def _save_embeddings(
    path: Path,
    *,
    embeddings: Sequence[np.ndarray],
    quality: Sequence[Sequence[float]],
    selected_indices: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if embeddings:
        embedding_array = np.stack(embeddings).astype(np.float32, copy=False)
    else:
        embedding_array = np.empty((0, 512), dtype=np.float32)
    quality_array = np.asarray(quality, dtype=np.float32).reshape((-1, 6))
    index_array = np.asarray(selected_indices, dtype=np.int32)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=embedding_array,
            quality=quality_array,
            selected_indices=index_array,
        )
    os.replace(temporary, path)


def _load_embeddings(path: Path) -> tuple[list[np.ndarray], list[list[float]], list[int]]:
    """비공개 NPZ를 검증해 재시도 파이프라인의 시작점으로 읽는다."""

    if not path.is_file():
        raise FileNotFoundError(f"기준 임베딩을 찾지 못했습니다: {path}")
    with np.load(path, allow_pickle=False) as payload:
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        quality = np.asarray(payload["quality"], dtype=np.float32)
        selected_indices = np.asarray(payload["selected_indices"], dtype=np.int32)
    if embeddings.ndim != 2 or embeddings.shape[1:] != (512,):
        raise ValueError(f"기준 임베딩 형식이 다릅니다: {path.name}")
    if quality.shape != (len(embeddings), 6):
        raise ValueError(f"기준 품질값 형식이 다릅니다: {path.name}")
    if selected_indices.shape != (len(embeddings),):
        raise ValueError(f"기준 선택 인덱스 형식이 다릅니다: {path.name}")
    if len(np.unique(selected_indices)) != len(selected_indices):
        raise ValueError(f"기준 선택 인덱스가 중복됐습니다: {path.name}")
    if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(quality)):
        raise ValueError(f"기준 NPZ에 유한하지 않은 값이 있습니다: {path.name}")
    return (
        [row.copy() for row in embeddings],
        quality.astype(float).tolist(),
        selected_indices.astype(int).tolist(),
    )


def process_adaptive_retry(
    archive_path: Path,
    *,
    resolution: str,
    baseline_dir: Path,
    output_dir: Path,
    max_subjects: int,
    images_per_subject: int,
    retry_encoders: Sequence[tuple[str, Any]],
    archive_sha256: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """기준 처리에서 실패한 이미지만 더 큰 검출 입력으로 다시 시도한다.

    기준 NPZ는 수정하지 않고 새 디렉터리에 병합 결과를 만든다. 인물별 완료
    체크포인트를 남기므로 Mac이 잠들거나 명령이 중단돼도 완료 인물은 건너뛴다.
    """

    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError("해상도는 low 또는 medium이어야 합니다.")
    if max_subjects <= 0 or images_per_subject <= 0:
        raise ValueError("인물 수와 인물당 이미지 수는 양수여야 합니다.")
    if not retry_encoders or any(not label.strip() for label, _ in retry_encoders):
        raise ValueError("이름이 있는 재시도 인코더가 하나 이상 필요합니다.")
    labels = [label for label, _ in retry_encoders]
    if len(set(labels)) != len(labels):
        raise ValueError("재시도 단계 이름은 중복될 수 없습니다.")

    baseline_summary_path = baseline_dir / "summary.json"
    if not baseline_summary_path.is_file():
        raise FileNotFoundError(
            f"기준 처리 요약을 찾지 못했습니다: {baseline_summary_path}"
        )
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    if baseline_summary.get("resolution") != resolution:
        raise ValueError("기준 처리 해상도와 재시도 해상도가 다릅니다.")
    if int(baseline_summary.get("subject_count", 0)) < max_subjects:
        raise ValueError("기준 처리 인물 수가 요청한 재시도 인물 수보다 적습니다.")

    subjects = list_subject_archives(archive_path)[:max_subjects]
    source_sha256 = archive_sha256 or _sha256_file(archive_path)
    if baseline_summary.get("archive_sha256") not in {None, source_sha256}:
        raise ValueError("기준 처리와 재시도 원본 ZIP의 SHA-256이 다릅니다.")
    retry_stage_configs = []
    for label, encoder in retry_encoders:
        settings = getattr(encoder, "settings", None)
        retry_stage_configs.append(
            {
                "label": label,
                "detection_size": getattr(settings, "detection_size", None),
                "minimum_detection_score": getattr(
                    settings, "minimum_detection_score", None
                ),
                "model_name": getattr(settings, "model_name", None),
            }
        )
    config = {
        "pipeline_version": "kface-adaptive-retry-v1",
        "resolution": resolution,
        "archive_sha256": source_sha256,
        "baseline_config_fingerprint": baseline_summary.get("config_fingerprint"),
        "max_subjects": max_subjects,
        "images_per_subject": images_per_subject,
        "retry_stages": retry_stage_configs,
    }
    config_fingerprint = _config_fingerprint(config)
    stage_recoveries: Counter[str] = Counter()
    remaining_reasons: Counter[str] = Counter()
    initial_accepted = 0
    final_accepted = 0
    skipped_subjects = 0
    started = time.perf_counter()

    with zipfile.ZipFile(archive_path) as outer:
        for position, subject in enumerate(subjects, start=1):
            output_embedding, output_checkpoint = _subject_paths(
                output_dir, subject.pseudonym
            )
            completed = _completed_checkpoint(output_checkpoint, config_fingerprint)
            if completed is not None and output_embedding.is_file():
                skipped_subjects += 1
                initial_accepted += int(completed["initial_accepted_images"])
                final_accepted += int(completed["accepted_images"])
                stage_recoveries.update(completed.get("retry_recoveries", {}))
                remaining_reasons.update(completed.get("remaining_reject_reasons", {}))
                if progress:
                    progress(
                        {
                            "subject": position,
                            "subjects": len(subjects),
                            "status": "skipped",
                            "accepted": int(completed["accepted_images"]),
                        }
                    )
                continue

            baseline_embedding, _ = _subject_paths(baseline_dir, subject.pseudonym)
            embeddings, qualities, accepted_indices = _load_embeddings(
                baseline_embedding
            )
            if any(index < 0 or index >= images_per_subject for index in accepted_indices):
                raise ValueError(
                    f"기준 선택 인덱스가 요청 범위를 벗어났습니다: {baseline_embedding.name}"
                )
            subject_initial = len(embeddings)
            missing = sorted(set(range(images_per_subject)) - set(accepted_indices))
            subject_recoveries: Counter[str] = Counter()
            subject_reasons: Counter[str] = Counter()
            subject_started = time.perf_counter()

            if missing:
                nested_payload = outer.read(subject.outer_member)
                if len(nested_payload) != subject.uncompressed_bytes:
                    raise OSError("인물별 ZIP 크기가 바깥 ZIP 목록과 다릅니다.")
                with zipfile.ZipFile(BytesIO(nested_payload)) as inner:
                    selected = evenly_spaced(_image_infos(inner), images_per_subject)
                    if len(selected) != images_per_subject:
                        raise ValueError("기준 처리와 같은 이미지 수를 선택하지 못했습니다.")
                    for image_index in missing:
                        image_payload = inner.read(selected[image_index])
                        final_error: Exception | None = None
                        for stage_label, encoder in retry_encoders:
                            try:
                                encoded = encoder.encode(image_payload)
                                embedding = np.asarray(
                                    encoded.embedding, dtype=np.float32
                                ).reshape(-1)
                                if embedding.shape != (512,) or not np.all(
                                    np.isfinite(embedding)
                                ):
                                    raise ValueError(
                                        "얼굴 임베딩이 512차원 유한값이 아닙니다."
                                    )
                                item_quality = encoded.quality
                                embeddings.append(embedding)
                                qualities.append(
                                    [
                                        float(item_quality.detection_score),
                                        float(item_quality.face_area_ratio),
                                        float(item_quality.blur_score),
                                        float(item_quality.brightness_mean),
                                        float(item_quality.image_width),
                                        float(item_quality.image_height),
                                    ]
                                )
                                accepted_indices.append(image_index)
                                subject_recoveries[stage_label] += 1
                                final_error = None
                                break
                            except Exception as error:  # noqa: BLE001 - 다음 단계로 재시도
                                final_error = error
                        if final_error is not None:
                            code = getattr(final_error, "code", None) or type(
                                final_error
                            ).__name__
                            subject_reasons[str(code)] += 1

            order = np.argsort(np.asarray(accepted_indices, dtype=np.int32))
            sorted_embeddings = [embeddings[int(index)] for index in order]
            sorted_qualities = [qualities[int(index)] for index in order]
            sorted_indices = [accepted_indices[int(index)] for index in order]
            _save_embeddings(
                output_embedding,
                embeddings=sorted_embeddings,
                quality=sorted_qualities,
                selected_indices=sorted_indices,
            )
            subject_final = len(sorted_embeddings)
            checkpoint = {
                "complete": True,
                "config_fingerprint": config_fingerprint,
                "selected_images": images_per_subject,
                "initial_accepted_images": subject_initial,
                "accepted_images": subject_final,
                "rejected_images": images_per_subject - subject_final,
                "retry_recoveries": dict(sorted(subject_recoveries.items())),
                "remaining_reject_reasons": dict(sorted(subject_reasons.items())),
                "elapsed_seconds": time.perf_counter() - subject_started,
                "contains_raw_path": False,
                "contains_face_image": False,
            }
            _atomic_json(output_checkpoint, checkpoint)
            initial_accepted += subject_initial
            final_accepted += subject_final
            stage_recoveries.update(subject_recoveries)
            remaining_reasons.update(subject_reasons)
            if progress:
                progress(
                    {
                        "subject": position,
                        "subjects": len(subjects),
                        "status": "processed",
                        "initial_accepted": subject_initial,
                        "recovered": subject_final - subject_initial,
                        "remaining_rejected": images_per_subject - subject_final,
                        "elapsed_seconds": round(time.perf_counter() - subject_started, 3),
                    }
                )

    summary = {
        "dataset": "K-FACE",
        "resolution": resolution,
        "pipeline_version": config["pipeline_version"],
        "config_fingerprint": config_fingerprint,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": source_sha256,
        "subject_count": len(subjects),
        "skipped_completed_subjects": skipped_subjects,
        "selected_images": len(subjects) * images_per_subject,
        "initial_accepted_images": initial_accepted,
        "accepted_images": final_accepted,
        "recovered_images": final_accepted - initial_accepted,
        "rejected_images": len(subjects) * images_per_subject - final_accepted,
        "retry_recoveries": dict(sorted(stage_recoveries.items())),
        "retry_stages": retry_stage_configs,
        "remaining_reject_reasons": dict(sorted(remaining_reasons.items())),
        "processing_seconds": time.perf_counter() - started,
        "model_providers": sorted(
            {
                getattr(encoder, "provider", None)
                for _, encoder in retry_encoders
                if getattr(encoder, "provider", None)
            }
        ),
        "model_fingerprints": sorted(
            {
                getattr(encoder, "model_fingerprint", None)
                for _, encoder in retry_encoders
                if getattr(encoder, "model_fingerprint", None)
            }
        ),
        "contains_raw_paths": False,
        "contains_face_images": False,
        "embeddings_are_private": True,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def process_archive(
    archive_path: Path,
    *,
    resolution: str,
    output_dir: Path,
    max_subjects: int,
    images_per_subject: int,
    encoder: Any,
    archive_sha256: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError("해상도는 low 또는 medium이어야 합니다.")
    if max_subjects <= 0 or images_per_subject <= 0:
        raise ValueError("인물 수와 인물당 이미지 수는 양수여야 합니다.")
    subjects = list_subject_archives(archive_path)[:max_subjects]
    source_sha256 = archive_sha256 or _sha256_file(archive_path)
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "resolution": resolution,
        "archive_sha256": source_sha256,
        "max_subjects": max_subjects,
        "images_per_subject": images_per_subject,
    }
    config_fingerprint = _config_fingerprint(config)
    results: list[SubjectResult] = []
    skipped_subjects = 0
    started = time.perf_counter()

    with zipfile.ZipFile(archive_path) as outer:
        for position, subject in enumerate(subjects, start=1):
            embedding_path, checkpoint_path = _subject_paths(
                output_dir, subject.pseudonym
            )
            completed = _completed_checkpoint(checkpoint_path, config_fingerprint)
            if completed is not None and embedding_path.is_file():
                skipped_subjects += 1
                result = SubjectResult(
                    pseudonym=subject.pseudonym,
                    selected_images=int(completed["selected_images"]),
                    accepted_images=int(completed["accepted_images"]),
                    rejected_images=int(completed["rejected_images"]),
                    reject_reasons=dict(completed.get("reject_reasons", {})),
                    elapsed_seconds=float(completed.get("elapsed_seconds", 0.0)),
                    checkpoint=str(embedding_path),
                )
                results.append(result)
                if progress:
                    progress(
                        {
                            "subject": position,
                            "subjects": len(subjects),
                            "status": "skipped",
                            "accepted": result.accepted_images,
                        }
                    )
                continue

            subject_started = time.perf_counter()
            nested_payload = outer.read(subject.outer_member)
            if len(nested_payload) != subject.uncompressed_bytes:
                raise OSError("인물별 ZIP 크기가 바깥 ZIP 목록과 다릅니다.")
            embeddings: list[np.ndarray] = []
            qualities: list[list[float]] = []
            accepted_indices: list[int] = []
            reject_reasons: Counter[str] = Counter()
            with zipfile.ZipFile(BytesIO(nested_payload)) as inner:
                images = _image_infos(inner)
                selected = evenly_spaced(images, images_per_subject)
                for image_index, info in enumerate(selected):
                    try:
                        encoded = encoder.encode(inner.read(info))
                        embedding = np.asarray(encoded.embedding, dtype=np.float32).reshape(-1)
                        if embedding.shape != (512,) or not np.all(np.isfinite(embedding)):
                            raise ValueError("얼굴 임베딩이 512차원 유한값이 아닙니다.")
                        item_quality = encoded.quality
                        embeddings.append(embedding)
                        qualities.append(
                            [
                                float(item_quality.detection_score),
                                float(item_quality.face_area_ratio),
                                float(item_quality.blur_score),
                                float(item_quality.brightness_mean),
                                float(item_quality.image_width),
                                float(item_quality.image_height),
                            ]
                        )
                        accepted_indices.append(image_index)
                    except Exception as error:  # noqa: BLE001 - 한 장 실패로 배치를 멈추지 않는다.
                        code = getattr(error, "code", None) or type(error).__name__
                        reject_reasons[str(code)] += 1

            _save_embeddings(
                embedding_path,
                embeddings=embeddings,
                quality=qualities,
                selected_indices=accepted_indices,
            )
            elapsed = time.perf_counter() - subject_started
            result = SubjectResult(
                pseudonym=subject.pseudonym,
                selected_images=len(selected),
                accepted_images=len(embeddings),
                rejected_images=len(selected) - len(embeddings),
                reject_reasons=dict(sorted(reject_reasons.items())),
                elapsed_seconds=elapsed,
                checkpoint=str(embedding_path),
            )
            checkpoint = {
                **asdict(result),
                "complete": True,
                "config_fingerprint": config_fingerprint,
                "source_member_crc32": subject.crc32,
                "contains_raw_path": False,
                "contains_face_image": False,
            }
            _atomic_json(checkpoint_path, checkpoint)
            results.append(result)
            if progress:
                progress(
                    {
                        "subject": position,
                        "subjects": len(subjects),
                        "status": "processed",
                        "accepted": result.accepted_images,
                        "rejected": result.rejected_images,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )

    aggregate_reasons: Counter[str] = Counter()
    for result in results:
        aggregate_reasons.update(result.reject_reasons)
    summary = {
        "dataset": "K-FACE",
        "resolution": resolution,
        "pipeline_version": PIPELINE_VERSION,
        "config_fingerprint": config_fingerprint,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": source_sha256,
        "subject_count": len(results),
        "skipped_completed_subjects": skipped_subjects,
        "selected_images": sum(result.selected_images for result in results),
        "accepted_images": sum(result.accepted_images for result in results),
        "rejected_images": sum(result.rejected_images for result in results),
        "reject_reasons": dict(sorted(aggregate_reasons.items())),
        "processing_seconds": time.perf_counter() - started,
        "model_provider": getattr(encoder, "provider", None),
        "model_fingerprint": getattr(encoder, "model_fingerprint", None),
        "contains_raw_paths": False,
        "contains_face_images": False,
        "embeddings_are_private": True,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _progress_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _build_encoder(args: argparse.Namespace) -> Any:
    if not args.accept_noncommercial_model_license:
        raise PermissionError(
            "InsightFace 제공 가중치의 비상업 연구 조건을 확인한 뒤 "
            "--accept-noncommercial-model-license를 지정하세요."
        )
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from faceguard_api.engine import InsightFaceEncoder
    from faceguard_api.settings import Settings

    return InsightFaceEncoder(
        Settings(
            model_root=args.model_root,
            device="cpu",
            accept_noncommercial_model_license=True,
        )
    )


def _build_retry_encoders(args: argparse.Namespace) -> list[tuple[str, Any]]:
    if not args.accept_noncommercial_model_license:
        raise PermissionError(
            "InsightFace 제공 가중치의 비상업 연구 조건을 확인한 뒤 "
            "--accept-noncommercial-model-license를 지정하세요."
        )
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from faceguard_api.engine import InsightFaceEncoder
    from faceguard_api.settings import Settings

    encoders: list[tuple[str, Any]] = []
    for detection_size in args.retry_detection_sizes:
        label = (
            f"det{detection_size}_score"
            f"{int(round(args.retry_minimum_detection_score * 100)):02d}"
        )
        encoders.append(
            (
                label,
                InsightFaceEncoder(
                    Settings(
                        model_root=args.model_root,
                        device="cpu",
                        detection_size=detection_size,
                        minimum_detection_score=args.retry_minimum_detection_score,
                        accept_noncommercial_model_license=True,
                    )
                ),
            )
        )
    return encoders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="중첩 ZIP 구조를 검사")
    inventory.add_argument("archive", type=Path)
    inventory.add_argument("--resolution", choices=sorted(SUPPORTED_RESOLUTIONS), required=True)
    inventory.add_argument("--inspect-subjects", type=int)
    inventory.add_argument("--expected-bytes", type=int)
    inventory.add_argument("--output", type=Path, required=True)

    process = subparsers.add_parser("process", help="소규모 ArcFace 배치를 처리")
    process.add_argument("archive", type=Path)
    process.add_argument("--resolution", choices=sorted(SUPPORTED_RESOLUTIONS), required=True)
    process.add_argument("--output-dir", type=Path, required=True)
    process.add_argument("--max-subjects", type=int, default=30)
    process.add_argument("--images-per-subject", type=int, default=15)
    process.add_argument("--model-root", type=Path, default=Path(".models/insightface"))
    process.add_argument("--archive-sha256")
    process.add_argument("--expected-bytes", type=int)
    process.add_argument("--accept-noncommercial-model-license", action="store_true")

    retry = subparsers.add_parser(
        "retry", help="기준 처리에서 실패한 이미지만 더 큰 검출 입력으로 재시도"
    )
    retry.add_argument("archive", type=Path)
    retry.add_argument("--resolution", choices=sorted(SUPPORTED_RESOLUTIONS), required=True)
    retry.add_argument("--baseline-dir", type=Path, required=True)
    retry.add_argument("--output-dir", type=Path, required=True)
    retry.add_argument("--max-subjects", type=int, default=400)
    retry.add_argument("--images-per-subject", type=int, default=15)
    retry.add_argument("--model-root", type=Path, default=Path(".models/insightface"))
    retry.add_argument("--archive-sha256")
    retry.add_argument("--expected-bytes", type=int)
    retry.add_argument(
        "--retry-detection-sizes", type=int, nargs="+", default=[960, 1280]
    )
    retry.add_argument("--retry-minimum-detection-score", type=float, default=0.50)
    retry.add_argument("--accept-noncommercial-model-license", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.archive.is_file():
        raise FileNotFoundError(f"ZIP 파일을 찾지 못했습니다: {args.archive}")
    validate_archive_size(args.archive, args.expected_bytes)
    if args.command == "inventory":
        payload = inventory_archive(
            args.archive,
            resolution=args.resolution,
            inspect_subjects=args.inspect_subjects,
        )
        _atomic_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "retry":
        if any(size <= 0 for size in args.retry_detection_sizes):
            raise ValueError("재시도 검출 입력 크기는 양수여야 합니다.")
        if not 0.0 <= args.retry_minimum_detection_score <= 1.0:
            raise ValueError("재시도 최소 검출 점수는 0과 1 사이여야 합니다.")
        payload = process_adaptive_retry(
            args.archive,
            resolution=args.resolution,
            baseline_dir=args.baseline_dir,
            output_dir=args.output_dir,
            max_subjects=args.max_subjects,
            images_per_subject=args.images_per_subject,
            retry_encoders=_build_retry_encoders(args),
            archive_sha256=args.archive_sha256,
            progress=_progress_line,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    encoder = _build_encoder(args)
    payload = process_archive(
        args.archive,
        resolution=args.resolution,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects,
        images_per_subject=args.images_per_subject,
        encoder=encoder,
        archive_sha256=args.archive_sha256,
        progress=_progress_line,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
