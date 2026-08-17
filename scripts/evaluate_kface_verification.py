#!/usr/bin/env python3
"""K-FACE 저·중화질 임베딩으로 3장·5장 등록 검증을 수행한다.

중화질 등록 사진의 평균 임베딩을 만든 뒤 등록에 쓰지 않은
저·중화질 질의를 본인 및 타인 등록 중심과 비교한다. 기준값은 validation
인물에서만 고르고 subject-disjoint test 인물에 고정 적용한다.
원본 경로, 인물 식별자, 개별 임베딩과 점수는 공개 결과에 기록하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (512,):
        raise ValueError("임베딩은 (N, 512) 형식이어야 합니다.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if len(array) and (not np.all(np.isfinite(array)) or np.any(norms <= 0)):
        raise ValueError("유한하지 않거나 0인 임베딩은 비교할 수 없습니다.")
    return array / norms if len(array) else array


def _unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (512,) or not np.all(np.isfinite(vector)) or norm <= 0:
        raise ValueError("유한한 512차원 중심 벡터가 필요합니다.")
    return vector / norm


def _load_subjects(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"결과 요약을 찾지 못했습니다: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    subjects: dict[str, Any] = {}
    for path in sorted((directory / "embeddings").glob("subject_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            embeddings = _unit_rows(payload["embeddings"])
            selected_indices = np.asarray(payload["selected_indices"], dtype=np.int32)
        if selected_indices.shape != (len(embeddings),):
            raise ValueError(f"선택 인덱스 형식이 다릅니다: {path.name}")
        if len(np.unique(selected_indices)) != len(selected_indices):
            raise ValueError(f"중복된 선택 인덱스가 있습니다: {path.name}")
        subjects[path.stem] = {
            "embeddings": embeddings,
            "selected_indices": selected_indices,
        }
    return summary, subjects


def _even_positions(length: int, count: int) -> np.ndarray:
    if count <= 0 or length < count:
        raise ValueError("등록 사진 수보다 성공 임베딩이 적습니다.")
    if count == 1:
        return np.asarray([length // 2], dtype=np.int32)
    return np.asarray(
        [round(index * (length - 1) / (count - 1)) for index in range(count)],
        dtype=np.int32,
    )


def _enrollment(subject: dict[str, Any], references: int) -> tuple[np.ndarray, set[int]]:
    embeddings = subject["embeddings"]
    selected_indices = subject["selected_indices"]
    positions = _even_positions(len(embeddings), references)
    center = _unit_vector(np.mean(embeddings[positions], axis=0))
    used = {int(selected_indices[position]) for position in positions}
    return center, used


def _query_rows(subject: dict[str, Any], excluded_indices: set[int]) -> np.ndarray:
    mask = np.asarray(
        [int(index) not in excluded_indices for index in subject["selected_indices"]],
        dtype=bool,
    )
    return subject["embeddings"][mask]


def _score_split(
    subject_ids: Sequence[str],
    *,
    medium_subjects: dict[str, Any],
    query_subjects: dict[str, Any],
    references: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    centers = []
    excluded: dict[str, set[int]] = {}
    eligible = []
    for subject_id in subject_ids:
        medium = medium_subjects[subject_id]
        if len(medium["embeddings"]) < references + 1:
            continue
        center, used = _enrollment(medium, references)
        queries = _query_rows(query_subjects[subject_id], used)
        if not len(queries):
            continue
        eligible.append(subject_id)
        centers.append(center)
        excluded[subject_id] = used
    if len(eligible) < 2:
        raise ValueError("본인·타인 비교에 필요한 인물이 부족합니다.")

    # Apple Accelerate의 일부 float32 SGEMM 경로에서 유한 입력에도
    # overflow 경고가 발생한 사례가 있어 검증 점수 행렬곱은 float64로 고정한다.
    center_matrix = np.stack(centers).astype(np.float64, copy=False)
    genuine_chunks = []
    impostor_chunks = []
    query_count = 0
    for position, subject_id in enumerate(eligible):
        queries = _query_rows(
            query_subjects[subject_id], excluded[subject_id]
        ).astype(np.float64, copy=False)
        scores = np.einsum("qd,cd->qc", queries, center_matrix, optimize=False)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("유한하지 않은 유사도 점수가 발생했습니다.")
        genuine_chunks.append(scores[:, position])
        impostor_chunks.append(np.delete(scores, position, axis=1).reshape(-1))
        query_count += len(queries)
    return (
        np.concatenate(genuine_chunks).astype(np.float64, copy=False),
        np.concatenate(impostor_chunks).astype(np.float64, copy=False),
        query_count,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("빈 점수 분포는 집계할 수 없습니다.")
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _histogram(values: np.ndarray, *, bins: int = 60) -> dict[str, Any]:
    counts, edges = np.histogram(
        np.asarray(values, dtype=np.float64), bins=bins, range=(-1.0, 1.0)
    )
    return {
        "range": [-1.0, 1.0],
        "bins": bins,
        "counts": counts.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
    }


def _roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    values = np.concatenate([impostor, genuine]).astype(np.float64, copy=False)
    labels = np.concatenate(
        [np.zeros(len(impostor), dtype=np.int8), np.ones(len(genuine), dtype=np.int8)]
    )
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[:-1] != sorted_values[1:]) + 1]
    ends = np.r_[starts[1:], len(sorted_values)]
    counts = ends - starts
    average_ranks = (starts + 1 + ends) / 2.0
    sorted_ranks = np.repeat(average_ranks, counts)
    sorted_labels = labels[order]
    positives = sorted_labels == 1
    n_positive = int(np.sum(positives))
    n_negative = len(labels) - n_positive
    rank_sum = float(np.sum(sorted_ranks[positives]))
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([genuine, impostor]).astype(np.float64, copy=False)
    labels = np.concatenate(
        [np.ones(len(genuine), dtype=np.int8), np.zeros(len(impostor), dtype=np.int8)]
    )
    order = np.argsort(-values, kind="mergesort")
    scores = values[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1 - sorted_labels)
    group_ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    fpr = false_positive[group_ends] / len(impostor)
    fnr = 1.0 - true_positive[group_ends] / len(genuine)
    best = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[best] + fnr[best]) / 2.0), float(scores[group_ends[best]])


def _target_far_threshold(impostor: np.ndarray, target_far: float) -> float:
    if not 0 < target_far < 1:
        raise ValueError("target FAR은 0과 1 사이여야 합니다.")
    descending = np.sort(np.asarray(impostor, dtype=np.float64))[::-1]
    allowed = int(math.floor(target_far * len(descending)))
    if allowed <= 0:
        return float(
            np.nextafter(
                np.float32(descending[0]), np.float32(math.inf), dtype=np.float32
            )
        )
    if allowed >= len(descending):
        return float(-math.inf)
    return float(
        np.nextafter(
            np.float32(descending[allowed]), np.float32(math.inf), dtype=np.float32
        )
    )


def _metrics(
    genuine: np.ndarray,
    impostor: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    tar = float(np.mean(np.asarray(genuine, dtype=np.float64) >= threshold))
    far = float(np.mean(np.asarray(impostor, dtype=np.float64) >= threshold))
    eer, eer_threshold = _eer(genuine, impostor)
    return {
        "threshold": float(threshold),
        "roc_auc": _roc_auc(genuine, impostor),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "tar": tar,
        "frr": 1.0 - tar,
        "far": far,
        "genuine": _distribution(genuine),
        "impostor": _distribution(impostor),
        "genuine_histogram": _histogram(genuine),
        "impostor_histogram": _histogram(impostor),
    }


def _subject_split(subject_ids: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    ids = np.asarray(sorted(subject_ids), dtype=object)
    rng = np.random.default_rng(seed)
    ids = ids[rng.permutation(len(ids))]
    midpoint = len(ids) // 2
    if midpoint < 2 or len(ids) - midpoint < 2:
        raise ValueError("validation/test 인물 분리에 필요한 표본이 부족합니다.")
    return ids[:midpoint].tolist(), ids[midpoint:].tolist()


def evaluate(
    low_dir: Path,
    medium_dir: Path,
    *,
    references: Sequence[int],
    seed: int,
    target_far: float,
) -> dict[str, Any]:
    low_summary, low_subjects = _load_subjects(low_dir)
    medium_summary, medium_subjects = _load_subjects(medium_dir)
    paired = sorted(set(low_subjects) & set(medium_subjects))
    maximum_references = max(references)
    eligible = [
        subject_id
        for subject_id in paired
        if len(medium_subjects[subject_id]["embeddings"]) >= maximum_references + 1
        and len(low_subjects[subject_id]["embeddings"]) >= 1
    ]
    validation_ids, test_ids = _subject_split(eligible, seed)

    protocols: dict[str, Any] = {}
    for reference_count in references:
        scored: dict[str, Any] = {}
        private_scores: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for resolution, query_subjects in (
            ("low", low_subjects),
            ("medium", medium_subjects),
        ):
            private_scores[resolution] = {}
            for split_name, split_ids in (
                ("validation", validation_ids),
                ("test", test_ids),
            ):
                genuine, impostor, query_count = _score_split(
                    split_ids,
                    medium_subjects=medium_subjects,
                    query_subjects=query_subjects,
                    references=reference_count,
                )
                private_scores[resolution][split_name] = (genuine, impostor)
                scored.setdefault(resolution, {})[split_name] = {
                    "query_count": query_count,
                    "genuine_count": int(len(genuine)),
                    "impostor_count": int(len(impostor)),
                }

        candidate_thresholds = {
            resolution: _target_far_threshold(
                private_scores[resolution]["validation"][1], target_far
            )
            for resolution in ("low", "medium")
        }
        operating_threshold = max(candidate_thresholds.values())
        for resolution in ("low", "medium"):
            for split_name in ("validation", "test"):
                genuine, impostor = private_scores[resolution][split_name]
                scored[resolution][split_name].update(
                    _metrics(genuine, impostor, threshold=operating_threshold)
                )
        test_tars = [scored[item]["test"]["tar"] for item in ("low", "medium")]
        test_fars = [scored[item]["test"]["far"] for item in ("low", "medium")]
        protocols[f"references_{reference_count}"] = {
            "reference_count": reference_count,
            "enrollment_resolution": "medium",
            "target_far": target_far,
            "validation_threshold_candidates": candidate_thresholds,
            "operating_threshold": operating_threshold,
            "conditions": scored,
            "research_gate": {
                "target_minimum_tar": 0.90,
                "target_maximum_far": target_far,
                "observed_minimum_test_tar": min(test_tars),
                "observed_maximum_test_far": max(test_fars),
                "passed": min(test_tars) >= 0.90 and max(test_fars) <= target_far,
            },
        }

    return {
        "dataset": "K-FACE",
        "protocol": "medium_enrollment_subject_disjoint_low_medium_query_v1",
        "seed": seed,
        "target_far": target_far,
        "processed_subjects": {
            "low": int(low_summary.get("subject_count", len(low_subjects))),
            "medium": int(medium_summary.get("subject_count", len(medium_subjects))),
        },
        "paired_subjects": len(paired),
        "eligible_subjects": len(eligible),
        "validation_subjects": len(validation_ids),
        "test_subjects": len(test_ids),
        "protocols": protocols,
        "contains_raw_paths": False,
        "contains_subject_identifiers": False,
        "contains_face_images": False,
        "contains_embeddings": False,
        "threshold_status": "research_only_unapproved",
        "note": "K-FACE 저·중화질 연구 검증이며 운영·본인인증 승인이 아닙니다.",
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
    parser.add_argument("--references", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--target-far", type=float, default=0.001)
    args = parser.parse_args(argv)
    if not args.references or min(args.references) <= 0:
        parser.error("references는 양의 정수여야 합니다.")
    result = evaluate(
        args.low_dir,
        args.medium_dir,
        references=tuple(sorted(set(args.references))),
        seed=args.seed,
        target_far=args.target_far,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
