#!/usr/bin/env python3
"""K-FACE 비공개 임베딩을 추가 복사 없이 Kaggle 업로드 폴더로 준비한다.

원본 얼굴 이미지나 실명 식별자는 다루지 않는다. 처리 완료된 익명화 임베딩
NPZ만 업로드 폴더에 하드링크하고, Kaggle CLI가 하위 디렉터리 압축본을 만들지
않도록 파일명을 평탄화한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SUBJECT_PATTERN = re.compile(r"^subject_[0-9a-f]{16}$")
CHUNK_PATTERN = re.compile(r"^chunk_(\d{5})\.npz$")
DEFAULT_DATASET_ID = "hywznn/deepsogak-kface-arcface-private-2026-08-17"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _discover(source: Path) -> dict[str, list[Path]]:
    subjects: dict[str, list[Path]] = {}
    for subject_dir in sorted((source / "subjects").glob("subject_*")):
        if not subject_dir.is_dir() or not SUBJECT_PATTERN.fullmatch(subject_dir.name):
            continue
        chunks = sorted((subject_dir / "chunks").glob("chunk_*.npz"))
        if not chunks:
            raise ValueError(f"임베딩 chunk가 없습니다: {subject_dir.name}")
        indices: list[int] = []
        for chunk in chunks:
            match = CHUNK_PATTERN.fullmatch(chunk.name)
            if match is None:
                raise ValueError(f"chunk 파일명이 올바르지 않습니다: {chunk.name}")
            indices.append(int(match.group(1)))
        if indices != list(range(len(indices))):
            raise ValueError(f"chunk 번호가 연속되지 않습니다: {subject_dir.name}")
        subjects[subject_dir.name] = chunks
    return subjects


def prepare(
    source: Path,
    destination: Path,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    expected_subjects: int = 400,
    expected_chunks: int = 8_800,
    subjects_per_batch: int = 0,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("원본과 업로드 폴더는 달라야 합니다.")
    if subjects_per_batch < 0:
        raise ValueError("subjects_per_batch는 0 이상이어야 합니다.")
    subjects = _discover(source)
    chunk_count = sum(len(items) for items in subjects.values())
    if len(subjects) != expected_subjects or chunk_count != expected_chunks:
        raise ValueError(
            "전체 처리본 수가 예상과 다릅니다: "
            f"subjects={len(subjects)}/{expected_subjects}, "
            f"chunks={chunk_count}/{expected_chunks}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    linked_bytes = 0
    linked_files = 0
    expected_names: set[str] = set()
    subject_items = list(subjects.items())
    for subject_position, (subject, chunks) in enumerate(subject_items, start=1):
        target_directory = destination
        if subjects_per_batch:
            batch_start = (
                (subject_position - 1) // subjects_per_batch * subjects_per_batch + 1
            )
            batch_end = min(
                len(subject_items), batch_start + subjects_per_batch - 1
            )
            target_directory = destination / f"subjects_{batch_start:03d}_{batch_end:03d}"
            target_directory.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            target_name = f"{subject}__{chunk.name}"
            relative_name = str(target_directory.relative_to(destination) / target_name)
            expected_names.add(relative_name)
            target = target_directory / target_name
            source_stat = chunk.stat()
            if target.exists():
                target_stat = target.stat()
                if (
                    target_stat.st_ino != source_stat.st_ino
                    or target_stat.st_dev != source_stat.st_dev
                ):
                    raise FileExistsError(f"다른 파일이 이미 존재합니다: {target}")
            else:
                os.link(chunk, target)
            linked_bytes += source_stat.st_size
            linked_files += 1

    stale = [
        path
        for path in destination.rglob("subject_*__chunk_*.npz")
        if str(path.relative_to(destination)) not in expected_names
    ]
    if stale:
        raise ValueError(f"예상하지 못한 기존 파일이 있습니다: {stale[0].name}")

    manifest = {
        "dataset": "K-FACE",
        "purpose": "private ArcFace verification calibration",
        "pipeline_version": "kface-full-paired-v2",
        "config_fingerprint": (
            "312d3afc15b2541a4bf1a47b9ef837f77d027bef020b81626486d166ffad9252"
        ),
        "subject_count": len(subjects),
        "chunk_count": linked_files,
        "embedding_bytes": linked_bytes,
        "upload_layout": (
            "batched_directories" if subjects_per_batch else "flat_hardlinks"
        ),
        "subjects_per_batch": subjects_per_batch or None,
        "contains_raw_paths": False,
        "contains_subject_identifiers": False,
        "contains_face_images": False,
        "contains_embeddings": True,
        "privacy": "private_dataset_only",
    }
    _atomic_json(destination / "kface_private_manifest.json", manifest)
    _atomic_json(
        destination / "dataset-metadata.json",
        {
            "title": "DeepSogak K-FACE ArcFace Private Embeddings",
            "subtitle": "Private research embeddings for Korean face verification calibration",
            "description": (
                "AI-Hub K-FACE 승인 데이터에서 생성한 익명화 ArcFace 연구용 "
                "임베딩입니다. 원본 얼굴 이미지와 원본 식별자는 포함하지 않습니다. "
                "외부 공개 및 재배포를 금지합니다."
            ),
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        },
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--expected-subjects", type=int, default=400)
    parser.add_argument("--expected-chunks", type=int, default=8_800)
    parser.add_argument(
        "--subjects-per-batch",
        type=int,
        default=0,
        help="0이면 평탄화 파일, 양수이면 해당 인물 수마다 TAR 업로드 디렉터리 생성",
    )
    args = parser.parse_args(argv)
    result = prepare(
        args.source,
        args.destination,
        dataset_id=args.dataset_id,
        expected_subjects=args.expected_subjects,
        expected_chunks=args.expected_chunks,
        subjects_per_batch=args.subjects_per_batch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
