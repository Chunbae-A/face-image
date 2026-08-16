#!/usr/bin/env python3
"""K-FACE 400명 전체 임베딩으로 반복 얼굴 검증을 수행한다.

Kaggle GPU에서 400명 전체 저·중화질 임베딩을 스트리밍으로 읽는다. 인물
단위 validation/test 분리, 등록 3·5·9장, 반복 seed, FAR/TAR/EER/ROC-AUC를
평가한다. 수십억 개 타인 점수는 저장하지 않고 고해상도 histogram으로
누적하므로 메모리를 제한하면서 전체 비교를 사용할 수 있다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FLAT_PATTERN = re.compile(r"^(subject_[0-9a-f]{16})__(chunk_\d{5}\.npz)$")
NESTED_SUBJECT_PATTERN = re.compile(r"^subject_[0-9a-f]{16}$")
EMBEDDING_DIMENSIONS = 512
HISTOGRAM_MINIMUM = -1.0
HISTOGRAM_MAXIMUM = 1.0


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (EMBEDDING_DIMENSIONS,):
        raise ValueError("임베딩은 (N, 512) 형식이어야 합니다.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if len(array) and (not np.all(np.isfinite(array)) or np.any(norms <= 0)):
        raise ValueError("유한하지 않거나 0인 임베딩은 비교할 수 없습니다.")
    return array / norms if len(array) else array


def _unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (EMBEDDING_DIMENSIONS,) or not math.isfinite(norm) or norm <= 0:
        raise ValueError("유한한 512차원 중심 벡터가 필요합니다.")
    return vector / norm


def discover_subject_files(root: Path) -> dict[str, list[Path]]:
    """평탄화 Kaggle 입력 또는 로컬 중첩 구조에서 인물별 chunk를 찾는다."""

    root = root.resolve()
    subjects: dict[str, list[Path]] = defaultdict(list)
    # Kaggle의 ``--dir-mode tar`` 업로드는 묶음 tar를 Dataset 내부의
    # ``subjects_001_020/`` 같은 폴더로 자동 확장한다. 로컬 평탄 구조와
    # Kaggle 묶음 폴더를 같은 평가 코드로 읽기 위해 재귀 탐색한다.
    for path in sorted(root.rglob("subject_*__chunk_*.npz")):
        match = FLAT_PATTERN.fullmatch(path.name)
        if match is not None:
            subjects[match.group(1)].append(path)
    if subjects:
        return dict(subjects)

    nested_root = root / "subjects"
    for directory in sorted(nested_root.glob("subject_*")):
        if directory.is_dir() and NESTED_SUBJECT_PATTERN.fullmatch(directory.name):
            subjects[directory.name].extend(
                sorted((directory / "chunks").glob("chunk_*.npz"))
            )
    return {key: value for key, value in subjects.items() if value}


def _load_subject(paths: Sequence[Path]) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    required = (
        "image_indices",
        "low_embeddings",
        "medium_embeddings",
        "low_quality",
        "medium_quality",
    )
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(set(required) - set(payload.files))
            if missing:
                raise ValueError(f"필수 배열이 없습니다: {path.name}: {missing}")
            for key in required:
                chunks[key].append(np.asarray(payload[key]))
    result = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    count = len(result["image_indices"])
    if result["image_indices"].shape != (count,):
        raise ValueError("image_indices 형식이 올바르지 않습니다.")
    if len(np.unique(result["image_indices"])) != count:
        raise ValueError("한 인물 안에 중복 image_indices가 있습니다.")
    for key in ("low_embeddings", "medium_embeddings"):
        if result[key].shape != (count, EMBEDDING_DIMENSIONS):
            raise ValueError(f"{key} 형식이 올바르지 않습니다.")
        result[key] = _unit_rows(result[key])
    for key in ("low_quality", "medium_quality"):
        if result[key].shape != (count, 6) or not np.all(np.isfinite(result[key])):
            raise ValueError(f"{key} 형식이 올바르지 않습니다.")
        result[key] = np.asarray(result[key], dtype=np.float32)
    order = np.argsort(result["image_indices"], kind="mergesort")
    return {key: np.asarray(value)[order] for key, value in result.items()}


def _even_positions(length: int, count: int) -> np.ndarray:
    if count <= 0 or length < count:
        raise ValueError("등록 사진 수보다 품질 통과 임베딩이 적습니다.")
    if count == 1:
        return np.asarray([length // 2], dtype=np.int32)
    return np.asarray(
        [round(index * (length - 1) / (count - 1)) for index in range(count)],
        dtype=np.int32,
    )


def _subject_split(subject_ids: Sequence[str], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subject_ids))
    midpoint = len(order) // 2
    if midpoint < 2 or len(order) - midpoint < 2:
        raise ValueError("validation/test 인물 분리에 필요한 인물이 부족합니다.")
    return order[:midpoint].tolist(), order[midpoint:].tolist()


@dataclass
class ScoreHistogram:
    genuine: np.ndarray
    impostor: np.ndarray

    @classmethod
    def empty(cls, bins: int) -> ScoreHistogram:
        return cls(
            genuine=np.zeros(bins, dtype=np.int64),
            impostor=np.zeros(bins, dtype=np.int64),
        )


def _histogram_numpy(values: np.ndarray, bins: int) -> np.ndarray:
    counts, _ = np.histogram(
        np.asarray(values, dtype=np.float32),
        bins=bins,
        range=(HISTOGRAM_MINIMUM, HISTOGRAM_MAXIMUM),
    )
    return counts.astype(np.int64, copy=False)


class ScoreEngine:
    def __init__(self, device: str, bins: int) -> None:
        self.bins = bins
        self.torch: Any | None = None
        self.device = "cpu"
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
        if device != "cpu":
            try:
                import torch

                if torch.cuda.is_available():
                    self.torch = torch
                    self.device = "cuda"
                elif device == "cuda":
                    raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
            except ImportError:
                if device == "cuda":
                    raise RuntimeError("CUDA 실행에는 PyTorch가 필요합니다.") from None

    def centers(self, values: np.ndarray) -> Any:
        array = np.asarray(values, dtype=np.float32)
        if self.device == "cuda":
            return self.torch.as_tensor(array, device="cuda")
        return array

    def scores(self, queries: np.ndarray, centers: Any) -> Any:
        query_rows = np.asarray(queries, dtype=np.float32)
        if self.device == "cuda":
            tensor = self.torch.as_tensor(query_rows, device="cuda")
            return tensor @ centers.T
        return query_rows @ np.asarray(centers, dtype=np.float32).T

    def histogram(self, values: Any) -> np.ndarray:
        if self.device == "cuda":
            counts = self.torch.histc(
                values.float(),
                bins=self.bins,
                min=HISTOGRAM_MINIMUM,
                max=HISTOGRAM_MAXIMUM,
            )
            return counts.to(dtype=self.torch.int64, device="cpu").numpy()
        return _histogram_numpy(np.asarray(values), self.bins)

    def select_columns(self, scores: Any, columns: Sequence[int]) -> Any:
        if self.device == "cuda":
            index = self.torch.as_tensor(columns, dtype=self.torch.long, device="cuda")
            return scores.index_select(1, index)
        return np.asarray(scores)[:, np.asarray(columns, dtype=np.int64)]

    def select_column(self, scores: Any, column: int) -> Any:
        return scores[:, column]


def _histogram_edges(bins: int) -> np.ndarray:
    return np.linspace(HISTOGRAM_MINIMUM, HISTOGRAM_MAXIMUM, bins + 1)


def _threshold_for_far(impostor: np.ndarray, target_far: float) -> float:
    total = int(np.sum(impostor))
    if total <= 0:
        raise ValueError("타인 점수 histogram이 비어 있습니다.")
    allowed = math.floor(target_far * total)
    high_to_low = np.cumsum(impostor[::-1], dtype=np.int64)
    valid = np.flatnonzero(high_to_low <= allowed)
    if not len(valid):
        return HISTOGRAM_MAXIMUM
    reverse_index = int(valid[-1])
    bin_index = len(impostor) - 1 - reverse_index
    return float(_histogram_edges(len(impostor))[bin_index])


def _accepted(histogram: np.ndarray, threshold: float) -> int:
    edges = _histogram_edges(len(histogram))
    index = int(np.searchsorted(edges, threshold, side="left"))
    index = max(0, min(len(histogram), index))
    return int(np.sum(histogram[index:], dtype=np.int64))


def _percentile_from_histogram(histogram: np.ndarray, percentile: float) -> float:
    total = int(np.sum(histogram))
    if total <= 0:
        raise ValueError("빈 histogram의 분위수를 계산할 수 없습니다.")
    target = percentile / 100.0 * max(total - 1, 0)
    index = int(np.searchsorted(np.cumsum(histogram), target, side="right"))
    index = min(index, len(histogram) - 1)
    edges = _histogram_edges(len(histogram))
    return float((edges[index] + edges[index + 1]) / 2.0)


def _distribution(histogram: np.ndarray) -> dict[str, float | int]:
    count = int(np.sum(histogram))
    nonzero = np.flatnonzero(histogram)
    if count <= 0 or not len(nonzero):
        raise ValueError("빈 histogram은 집계할 수 없습니다.")
    edges = _histogram_edges(len(histogram))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "count": count,
        "minimum_approx": float(centers[int(nonzero[0])]),
        "p05_approx": _percentile_from_histogram(histogram, 5),
        "median_approx": _percentile_from_histogram(histogram, 50),
        "mean_approx": float(np.sum(histogram * centers) / count),
        "p95_approx": _percentile_from_histogram(histogram, 95),
        "maximum_approx": float(centers[int(nonzero[-1])]),
    }


def _roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    positives = int(np.sum(genuine))
    negatives = int(np.sum(impostor))
    if positives <= 0 or negatives <= 0:
        raise ValueError("ROC-AUC 계산에 본인·타인 점수가 모두 필요합니다.")
    negatives_below = np.cumsum(impostor, dtype=np.int64) - impostor
    wins = np.sum(genuine * (negatives_below + 0.5 * impostor), dtype=np.float64)
    return float(wins / (positives * negatives))


def _eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    positives = int(np.sum(genuine))
    negatives = int(np.sum(impostor))
    true_positive = np.cumsum(genuine[::-1], dtype=np.int64)[::-1]
    false_positive = np.cumsum(impostor[::-1], dtype=np.int64)[::-1]
    fpr = false_positive / negatives
    fnr = 1.0 - true_positive / positives
    index = int(np.argmin(np.abs(fpr - fnr)))
    threshold = float(_histogram_edges(len(genuine))[index])
    return float((fpr[index] + fnr[index]) / 2.0), threshold


def _preview(histogram: np.ndarray, output_bins: int = 200) -> dict[str, Any]:
    groups = np.array_split(np.arange(len(histogram)), output_bins)
    counts = [int(np.sum(histogram[group])) for group in groups]
    edges = _histogram_edges(len(histogram))
    preview_edges = [float(edges[int(group[0])]) for group in groups]
    preview_edges.append(HISTOGRAM_MAXIMUM)
    return {"range": [-1.0, 1.0], "bins": output_bins, "counts": counts, "edges": preview_edges}


def _metrics(scores: ScoreHistogram, threshold: float) -> dict[str, Any]:
    genuine_count = int(np.sum(scores.genuine))
    impostor_count = int(np.sum(scores.impostor))
    tar = _accepted(scores.genuine, threshold) / genuine_count
    far = _accepted(scores.impostor, threshold) / impostor_count
    eer, eer_threshold = _eer(scores.genuine, scores.impostor)
    return {
        "threshold": threshold,
        "roc_auc_approx": _roc_auc(scores.genuine, scores.impostor),
        "eer_approx": eer,
        "eer_threshold_approx": eer_threshold,
        "tar": tar,
        "frr": 1.0 - tar,
        "far": far,
        "genuine": _distribution(scores.genuine),
        "impostor": _distribution(scores.impostor),
        "histogram_method": "streaming_uniform_40000_bins_by_default",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def evaluate_full(
    input_dir: Path,
    *,
    references: Sequence[int] = (3, 5, 9),
    seeds: Sequence[int] = (20260815, 20260816, 20260817, 20260818, 20260819),
    target_far: float = 0.001,
    calibration_far: float = 0.0009,
    minimum_detection_score: float = 0.60,
    bins: int = 40_000,
    device: str = "auto",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    references = tuple(sorted({int(item) for item in references}))
    seeds = tuple(dict.fromkeys(int(item) for item in seeds))
    if not references or min(references) <= 0:
        raise ValueError("references는 양의 정수여야 합니다.")
    if not seeds:
        raise ValueError("seed가 하나 이상 필요합니다.")
    if not 0 < calibration_far <= target_far < 1:
        raise ValueError("calibration FAR은 0보다 크고 target FAR 이하여야 합니다.")
    if not 0 <= minimum_detection_score <= 1:
        raise ValueError("최소 검출점수는 0과 1 사이여야 합니다.")
    if bins < 1_000:
        raise ValueError("정밀한 FAR 평가를 위해 histogram bin은 1,000 이상이어야 합니다.")

    started = time.perf_counter()
    subject_files = discover_subject_files(input_dir)
    subject_ids = sorted(subject_files)
    if len(subject_ids) < 4:
        raise ValueError("본인·타인 검증에 필요한 인물이 부족합니다.")
    maximum_references = max(references)
    eligible: list[str] = []
    centers_by_reference: dict[int, list[np.ndarray]] = {item: [] for item in references}
    used_indices: dict[int, dict[str, set[int]]] = {
        item: {} for item in references
    }

    for position, subject_id in enumerate(subject_ids, start=1):
        subject = _load_subject(subject_files[subject_id])
        mask = subject["medium_quality"][:, 0] >= minimum_detection_score
        eligible_positions = np.flatnonzero(mask)
        if len(eligible_positions) < maximum_references + 1:
            continue
        eligible.append(subject_id)
        for reference_count in references:
            selected_positions = eligible_positions[
                _even_positions(len(eligible_positions), reference_count)
            ]
            center = _unit_vector(
                np.mean(subject["medium_embeddings"][selected_positions], axis=0)
            )
            centers_by_reference[reference_count].append(center)
            used_indices[reference_count][subject_id] = {
                int(item) for item in subject["image_indices"][selected_positions]
            }
        if progress and (position == 1 or position % 20 == 0 or position == len(subject_ids)):
            progress(
                {
                    "stage": "enrollment",
                    "processed_subjects": position,
                    "total_subjects": len(subject_ids),
                }
            )

    subject_ids = eligible
    if len(subject_ids) < 4:
        raise ValueError("품질 Gate 이후 평가 가능한 인물이 부족합니다.")
    subject_position = {item: index for index, item in enumerate(subject_ids)}
    split_indices: dict[int, dict[str, list[int]]] = {}
    split_membership: dict[int, dict[str, str]] = {}
    for seed in seeds:
        validation, test = _subject_split(subject_ids, seed)
        split_indices[seed] = {"validation": validation, "test": test}
        membership: dict[str, str] = {}
        for index in validation:
            membership[subject_ids[index]] = "validation"
        for index in test:
            membership[subject_ids[index]] = "test"
        split_membership[seed] = membership

    engine = ScoreEngine(device, bins)
    center_tensors = {
        reference_count: engine.centers(np.stack(centers))
        for reference_count, centers in centers_by_reference.items()
    }
    histograms: dict[tuple[int, int, str, str], ScoreHistogram] = {}

    def accumulator(seed: int, reference_count: int, resolution: str, split: str) -> ScoreHistogram:
        key = (seed, reference_count, resolution, split)
        if key not in histograms:
            histograms[key] = ScoreHistogram.empty(bins)
        return histograms[key]

    for completed, subject_id in enumerate(subject_ids, start=1):
        subject = _load_subject(subject_files[subject_id])
        own_position = subject_position[subject_id]
        for resolution in ("low", "medium"):
            quality = subject[f"{resolution}_quality"]
            quality_mask = quality[:, 0] >= minimum_detection_score
            for reference_count in references:
                excluded = used_indices[reference_count][subject_id]
                query_mask = quality_mask & np.asarray(
                    [int(item) not in excluded for item in subject["image_indices"]],
                    dtype=bool,
                )
                queries = subject[f"{resolution}_embeddings"][query_mask]
                if not len(queries):
                    raise ValueError(f"품질 Gate 이후 질의가 없습니다: {subject_id}")
                scores = engine.scores(queries, center_tensors[reference_count])
                genuine = engine.select_column(scores, own_position)
                for seed in seeds:
                    split = split_membership[seed][subject_id]
                    columns = [
                        item
                        for item in split_indices[seed][split]
                        if item != own_position
                    ]
                    score_histogram = accumulator(
                        seed, reference_count, resolution, split
                    )
                    score_histogram.genuine += engine.histogram(genuine)
                    impostor = engine.select_columns(scores, columns)
                    score_histogram.impostor += engine.histogram(impostor)
        if progress and (completed == 1 or completed % 10 == 0 or completed == len(subject_ids)):
            progress(
                {
                    "stage": "scoring",
                    "processed_subjects": completed,
                    "total_subjects": len(subject_ids),
                    "device": engine.device,
                }
            )

    runs: dict[str, Any] = {}
    aggregate_inputs: dict[int, dict[str, list[float]]] = {
        item: defaultdict(list) for item in references
    }
    for seed in seeds:
        seed_result: dict[str, Any] = {}
        for reference_count in references:
            candidates = {
                resolution: _threshold_for_far(
                    accumulator(seed, reference_count, resolution, "validation").impostor,
                    calibration_far,
                )
                for resolution in ("low", "medium")
            }
            operating_threshold = max(candidates.values())
            conditions: dict[str, Any] = {}
            for resolution in ("low", "medium"):
                conditions[resolution] = {}
                for split in ("validation", "test"):
                    item = accumulator(seed, reference_count, resolution, split)
                    conditions[resolution][split] = _metrics(item, operating_threshold)
                    if split == "test":
                        conditions[resolution][split]["histogram_preview"] = {
                            "genuine": _preview(item.genuine),
                            "impostor": _preview(item.impostor),
                        }
            test_tars = [conditions[item]["test"]["tar"] for item in ("low", "medium")]
            test_fars = [conditions[item]["test"]["far"] for item in ("low", "medium")]
            gate_passed = min(test_tars) >= 0.90 and max(test_fars) <= target_far
            seed_result[f"references_{reference_count}"] = {
                "reference_count": reference_count,
                "validation_threshold_candidates": candidates,
                "operating_threshold": operating_threshold,
                "conditions": conditions,
                "research_gate": {
                    "target_minimum_tar": 0.90,
                    "target_maximum_far": target_far,
                    "observed_minimum_test_tar": min(test_tars),
                    "observed_maximum_test_far": max(test_fars),
                    "passed": gate_passed,
                },
            }
            inputs = aggregate_inputs[reference_count]
            inputs["threshold"].append(operating_threshold)
            inputs["minimum_test_tar"].append(min(test_tars))
            inputs["maximum_test_far"].append(max(test_fars))
            inputs["gate"].append(float(gate_passed))
            for resolution in ("low", "medium"):
                inputs[f"{resolution}_tar"].append(conditions[resolution]["test"]["tar"])
                inputs[f"{resolution}_far"].append(conditions[resolution]["test"]["far"])
        runs[str(seed)] = seed_result

    aggregates: dict[str, Any] = {}
    for reference_count in references:
        values = aggregate_inputs[reference_count]
        metrics: dict[str, Any] = {}
        for name, rows in values.items():
            array = np.asarray(rows, dtype=np.float64)
            metrics[name] = {
                "minimum": float(np.min(array)),
                "median": float(np.median(array)),
                "maximum": float(np.max(array)),
            }
        aggregates[f"references_{reference_count}"] = {
            "reference_count": reference_count,
            "seed_count": len(seeds),
            "all_seeds_passed": all(item == 1.0 for item in values["gate"]),
            "conservative_candidate_threshold": float(max(values["threshold"])),
            "metrics_across_seeds": metrics,
        }

    passed = [
        item for item in references if aggregates[f"references_{item}"]["all_seeds_passed"]
    ]
    recommended = max(passed) if passed else max(
        references,
        key=lambda item: (
            aggregates[f"references_{item}"]["metrics_across_seeds"]["minimum_test_tar"]["minimum"],
            -aggregates[f"references_{item}"]["metrics_across_seeds"]["maximum_test_far"]["maximum"],
        ),
    )
    return {
        "dataset": "K-FACE",
        "protocol": "full_400_subject_streaming_histogram_v1",
        "pipeline_version": "kface-full-paired-v2",
        "input_subjects": len(subject_files),
        "eligible_subjects": len(subject_ids),
        "reference_counts": list(references),
        "seeds": list(seeds),
        "target_far": target_far,
        "calibration_far": calibration_far,
        "minimum_detection_score": minimum_detection_score,
        "histogram_bins": bins,
        "execution_device": engine.device,
        "runs": runs,
        "aggregates": aggregates,
        "recommendation": {
            "reference_count": recommended,
            "candidate_threshold": aggregates[f"references_{recommended}"]["conservative_candidate_threshold"],
            "all_seeds_passed": aggregates[f"references_{recommended}"]["all_seeds_passed"],
            "status": "research_only_unapproved",
        },
        "processing_seconds": time.perf_counter() - started,
        "contains_raw_paths": False,
        "contains_subject_identifiers": False,
        "contains_face_images": False,
        "contains_embeddings": False,
        "individual_scores_persisted": False,
        "threshold_status": "research_only_unapproved",
        "note": (
            "K-FACE 통제 촬영 데이터의 반복 연구 검증이다. 실제 웹·모바일 "
            "외부 검증 전에는 API 운영 기준값을 자동 교체하지 않는다."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=int, nargs="+", default=[3, 5, 9])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260815, 20260816, 20260817, 20260818, 20260819],
    )
    parser.add_argument("--target-far", type=float, default=0.001)
    parser.add_argument("--calibration-far", type=float, default=0.0009)
    parser.add_argument("--minimum-detection-score", type=float, default=0.60)
    parser.add_argument("--bins", type=int, default=40_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    def progress(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    result = evaluate_full(
        args.input_dir,
        references=args.references,
        seeds=args.seeds,
        target_far=args.target_far,
        calibration_far=args.calibration_far,
        minimum_detection_score=args.minimum_detection_score,
        bins=args.bins,
        device=args.device,
        progress=progress,
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "recommendation": result["recommendation"],
                "processing_seconds": result["processing_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
