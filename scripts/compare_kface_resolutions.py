#!/usr/bin/env python3
"""K-FACE 저화질·중화질 파일럿 결과를 개인정보 없이 비교한다."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

QUALITY_NAMES = (
    "detection_score",
    "face_area_ratio",
    "blur_score",
    "brightness_mean",
    "image_width",
    "image_height",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("0이거나 유한하지 않은 임베딩은 비교할 수 없습니다.")
    return vector / norm


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p05": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _load_private_results(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"결과 요약을 찾지 못했습니다: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    subjects: dict[str, Any] = {}
    for path in sorted((directory / "embeddings").glob("subject_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            quality = np.asarray(payload["quality"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1:] != (512,):
            raise ValueError(f"512차원 임베딩 형식이 아닙니다: {path.name}")
        if quality.shape != (len(embeddings), 6):
            raise ValueError(f"품질 지표 형식이 아닙니다: {path.name}")
        if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(quality)):
            raise ValueError(f"유한하지 않은 결과값이 있습니다: {path.name}")
        centroid = _unit(np.mean(embeddings, axis=0)) if len(embeddings) else None
        within = (
            [float(_unit(row) @ centroid) for row in embeddings]
            if centroid is not None
            else []
        )
        subjects[path.stem] = {
            "centroid": centroid,
            "within": within,
            "quality": quality,
        }
    return summary, subjects


def _resolution_metrics(summary: dict[str, Any], subjects: dict[str, Any]) -> dict[str, Any]:
    selected = int(summary.get("selected_images", 0))
    accepted = int(summary.get("accepted_images", 0))
    quality_rows = [
        row
        for subject in subjects.values()
        for row in subject["quality"]
    ]
    quality = np.asarray(quality_rows, dtype=np.float32).reshape((-1, 6))
    within = [
        score
        for subject in subjects.values()
        for score in subject["within"]
    ]
    return {
        "subject_count": len(subjects),
        "selected_images": selected,
        "accepted_images": accepted,
        "rejected_images": int(summary.get("rejected_images", selected - accepted)),
        "face_acceptance_rate": float(accepted / selected) if selected else None,
        "within_subject_similarity": _distribution(within),
        "quality_means": {
            name: (float(np.mean(quality[:, index])) if len(quality) else None)
            for index, name in enumerate(QUALITY_NAMES)
        },
    }


def compare_results(low_dir: Path, medium_dir: Path) -> dict[str, Any]:
    low_summary, low_subjects = _load_private_results(low_dir)
    medium_summary, medium_subjects = _load_private_results(medium_dir)
    paired = sorted(set(low_subjects) & set(medium_subjects))
    cross_resolution = []
    for pseudonym in paired:
        low_centroid = low_subjects[pseudonym]["centroid"]
        medium_centroid = medium_subjects[pseudonym]["centroid"]
        if low_centroid is not None and medium_centroid is not None:
            cross_resolution.append(float(low_centroid @ medium_centroid))
    return {
        "dataset": "K-FACE",
        "comparison": "low_vs_medium",
        "low": _resolution_metrics(low_summary, low_subjects),
        "medium": _resolution_metrics(medium_summary, medium_subjects),
        "paired_subject_count": len(paired),
        "paired_subjects_with_faces": len(cross_resolution),
        "cross_resolution_same_subject_similarity": _distribution(cross_resolution),
        "contains_raw_paths": False,
        "contains_face_images": False,
        "contains_embeddings": False,
        "note": "파일럿 표본의 얼굴 검출·특징 안정성 비교이며 운영 정확도 승인이 아닙니다.",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--medium-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_results(args.low_dir, args.medium_dir)
    _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
