#!/usr/bin/env python3
"""K-FACE 전체 특짓값에서 등록 5장 결합 방식을 반복 비교한다.

단순 평균, 품질 가중 평균, 두 개 등록 중심과 두 개 품질 가중
등록 중심을 같은 질의·인물 분할에서 비교한다. 질의 품질 Gate는
추가하지 않아 자동 처리 coverage를 100%로 고정한다. 개별 인물,
임베딩과 비교 점수는 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_kface_full_embeddings import (
    ScoreEngine,
    ScoreHistogram,
    _even_positions,
    _load_subject,
    _metrics,
    _subject_split,
    _threshold_for_far,
    _unit_vector,
    discover_subject_files,
)


@dataclass(frozen=True)
class EnrollmentStrategy:
    """등록 사진에서 비교용 템플릿을 만드는 고정 전략."""

    name: str
    prototype_count: int
    quality_weighted: bool
    description: str

    def __post_init__(self) -> None:
        if not self.name or self.prototype_count not in {1, 2}:
            raise ValueError("전략 이름과 1개 또는 2개 중심이 필요합니다.")


DEFAULT_STRATEGIES = (
    EnrollmentStrategy(
        "mean_5",
        prototype_count=1,
        quality_weighted=False,
        description="현재 API와 같은 등록 5장 단순 평균",
    ),
    EnrollmentStrategy(
        "quality_weighted_mean_5",
        prototype_count=1,
        quality_weighted=True,
        description="검출점수·얼굴 픽셀 크기·밝기 품질 가중 평균",
    ),
    EnrollmentStrategy(
        "dual_prototype_5",
        prototype_count=2,
        quality_weighted=False,
        description="가장 다른 두 등록 사진을 시작점으로 두 중심 보존",
    ),
    EnrollmentStrategy(
        "dual_quality_weighted_5",
        prototype_count=2,
        quality_weighted=True,
        description="두 등록 중심 안에서 품질 가중 평균",
    ),
)


def face_pixel_side(quality: np.ndarray) -> np.ndarray:
    """원본 이미지에서 얼굴 면적과 같은 정사각형의 한 변 크기."""

    values = np.asarray(quality, dtype=np.float32)
    if values.ndim != 2 or values.shape[1:] != (6,):
        raise ValueError("품질값은 (N, 6) 형식이어야 합니다.")
    return np.sqrt(values[:, 1] * values[:, 4] * values[:, 5])


def enrollment_quality_weights(quality: np.ndarray) -> np.ndarray:
    """등록 품질 3개 축을 평등하게 결합한 양수 가중치를 만든다.

    모든 축은 0.25~1.0으로 제한하고 기하평균을 사용해 특정
    지표 하나가 전체 가중치를 과도하게 지배하지 않게 한다.
    """

    values = np.asarray(quality, dtype=np.float32)
    sides = face_pixel_side(values)
    detection = np.clip((values[:, 0] - 0.50) / 0.40, 0.25, 1.0)
    size = np.clip(sides / 96.0, 0.25, 1.0)
    exposure = np.clip(
        1.0 - np.abs(values[:, 3] - 127.5) / 127.5,
        0.25,
        1.0,
    )
    weights = np.cbrt(detection * size * exposure).astype(np.float32)
    if not len(weights) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("유한한 양수 등록 가중치가 필요합니다.")
    return weights


def _weighted_center(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rows = np.asarray(embeddings, dtype=np.float32)
    values = np.asarray(weights, dtype=np.float32).reshape(-1)
    if rows.ndim != 2 or rows.shape[1:] != (512,) or len(rows) != len(values):
        raise ValueError("임베딩과 가중치 크기가 일치해야 합니다.")
    return _unit_vector(np.average(rows, axis=0, weights=values))


def _dual_assignments(embeddings: np.ndarray) -> np.ndarray:
    """가장 다른 두 사진을 초기점으로 고정한 2-means 할당."""

    rows = np.asarray(embeddings, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1:] != (512,) or len(rows) < 2:
        raise ValueError("두 중심에는 512차원 임베딩이 2개 이상 필요합니다.")
    similarities = rows @ rows.T
    np.fill_diagonal(similarities, np.inf)
    first, second = np.unravel_index(np.argmin(similarities), similarities.shape)
    seeds = rows[[first, second]]
    assignments = np.argmax(rows @ seeds.T, axis=1).astype(np.int32)
    assignments[first] = 0
    assignments[second] = 1
    return assignments


def build_templates(
    embeddings: np.ndarray,
    quality: np.ndarray,
    strategy: EnrollmentStrategy,
) -> np.ndarray:
    """전략에 따라 한 인물의 1개 또는 2개 등록 중심을 만든다."""

    rows = np.asarray(embeddings, dtype=np.float32)
    values = np.asarray(quality, dtype=np.float32)
    if rows.shape != (len(values), 512) or len(rows) < strategy.prototype_count:
        raise ValueError("등록 임베딩과 품질값 형식이 올바르지 않습니다.")
    weights = (
        enrollment_quality_weights(values)
        if strategy.quality_weighted
        else np.ones(len(rows), dtype=np.float32)
    )
    if strategy.prototype_count == 1:
        return _weighted_center(rows, weights)[None, :]
    assignments = _dual_assignments(rows)
    centers = [
        _weighted_center(rows[assignments == cluster], weights[assignments == cluster])
        for cluster in (0, 1)
    ]
    return np.stack(centers).astype(np.float32, copy=False)


def _template_scores(
    engine: ScoreEngine,
    queries: np.ndarray,
    flattened_templates: Any,
    *,
    subject_count: int,
    prototype_count: int,
) -> Any:
    raw = engine.scores(queries, flattened_templates)
    if prototype_count == 1:
        return raw
    if engine.device == "cuda":
        return raw.reshape(len(queries), subject_count, prototype_count).amax(dim=2)
    return np.max(
        np.asarray(raw).reshape(len(queries), subject_count, prototype_count),
        axis=2,
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("집계할 유한한 값이 필요합니다.")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def analyze_enrollment_strategies(
    input_dir: Path,
    *,
    strategies: Sequence[EnrollmentStrategy] = DEFAULT_STRATEGIES,
    reference_count: int = 5,
    seeds: Sequence[int] = (20260815, 20260816, 20260817, 20260818, 20260819),
    calibration_fars: Sequence[float] = (0.0009, 0.0008, 0.0007),
    target_far: float = 0.001,
    minimum_detection_score: float = 0.60,
    bins: int = 40_000,
    device: str = "auto",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """등록 결합 전략과 validation FAR 안전 여유를 반복 평가한다."""

    strategies = tuple(strategies)
    seeds = tuple(dict.fromkeys(int(item) for item in seeds))
    calibration_fars = tuple(
        sorted({float(item) for item in calibration_fars}, reverse=True)
    )
    if not strategies or len({item.name for item in strategies}) != len(strategies):
        raise ValueError("서로 다른 등록 전략이 하나 이상 필요합니다.")
    if reference_count < 2 or not seeds:
        raise ValueError("등록 사진은 2장 이상이고 seed가 필요합니다.")
    if (
        not calibration_fars
        or min(calibration_fars) <= 0
        or max(calibration_fars) > target_far
        or target_far >= 1
    ):
        raise ValueError("calibration FAR은 0보다 크고 target FAR 이하여야 합니다.")
    if not 0 <= minimum_detection_score <= 1 or bins < 1_000:
        raise ValueError("검출점수와 histogram bin 설정을 확인하세요.")

    started = time.perf_counter()
    subject_files = discover_subject_files(input_dir)
    subject_ids = sorted(subject_files)
    templates_by_strategy: dict[str, list[np.ndarray]] = {
        item.name: [] for item in strategies
    }
    used_indices: dict[str, set[int]] = {}
    eligible: list[str] = []
    enrollment_weights: list[np.ndarray] = []
    dual_cluster_sizes: dict[str, list[int]] = defaultdict(list)

    for position, subject_id in enumerate(subject_ids, start=1):
        subject = _load_subject(subject_files[subject_id])
        available = np.flatnonzero(
            subject["medium_quality"][:, 0] >= minimum_detection_score
        )
        if len(available) < reference_count + 1:
            continue
        selected = available[_even_positions(len(available), reference_count)]
        embeddings = subject["medium_embeddings"][selected]
        quality = subject["medium_quality"][selected]
        weights = enrollment_quality_weights(quality)
        enrollment_weights.append(weights)
        for strategy in strategies:
            templates = build_templates(embeddings, quality, strategy)
            templates_by_strategy[strategy.name].append(templates)
            if strategy.prototype_count == 2:
                assignments = _dual_assignments(embeddings)
                dual_cluster_sizes[strategy.name].extend(
                    [int(np.sum(assignments == item)) for item in (0, 1)]
                )
        used_indices[subject_id] = {
            int(item) for item in subject["image_indices"][selected]
        }
        eligible.append(subject_id)
        if progress and (
            position == 1 or position % 20 == 0 or position == len(subject_ids)
        ):
            progress(
                {
                    "stage": "enrollment",
                    "processed_subjects": position,
                    "total_subjects": len(subject_ids),
                }
            )

    subject_ids = eligible
    if len(subject_ids) < 4:
        raise ValueError("등록 전략 비교에 필요한 인물이 부족합니다.")
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
    template_tensors: dict[str, Any] = {}
    for strategy in strategies:
        stacked = np.stack(templates_by_strategy[strategy.name])
        expected = (len(subject_ids), strategy.prototype_count, 512)
        if stacked.shape != expected:
            raise ValueError(f"등록 템플릿 형식이 다릅니다: {strategy.name}")
        template_tensors[strategy.name] = engine.centers(stacked.reshape(-1, 512))

    histograms: dict[tuple[str, int, str, str], ScoreHistogram] = {}

    def accumulator(
        strategy: str, seed: int, resolution: str, split: str
    ) -> ScoreHistogram:
        key = (strategy, seed, resolution, split)
        if key not in histograms:
            histograms[key] = ScoreHistogram.empty(bins)
        return histograms[key]

    for completed, subject_id in enumerate(subject_ids, start=1):
        subject = _load_subject(subject_files[subject_id])
        own_position = subject_position[subject_id]
        excluded = used_indices[subject_id]
        for resolution in ("low", "medium"):
            quality = subject[f"{resolution}_quality"]
            query_mask = quality[:, 0] >= minimum_detection_score
            query_mask &= np.asarray(
                [int(item) not in excluded for item in subject["image_indices"]],
                dtype=bool,
            )
            queries = subject[f"{resolution}_embeddings"][query_mask]
            if not len(queries):
                raise ValueError(f"평가 질의가 없습니다: {subject_id}")
            for strategy in strategies:
                scores = _template_scores(
                    engine,
                    queries,
                    template_tensors[strategy.name],
                    subject_count=len(subject_ids),
                    prototype_count=strategy.prototype_count,
                )
                genuine = engine.select_column(scores, own_position)
                for seed in seeds:
                    split = split_membership[seed][subject_id]
                    columns = [
                        item
                        for item in split_indices[seed][split]
                        if item != own_position
                    ]
                    item = accumulator(strategy.name, seed, resolution, split)
                    item.genuine += engine.histogram(genuine)
                    item.impostor += engine.histogram(
                        engine.select_columns(scores, columns)
                    )
        if progress and (
            completed == 1 or completed % 10 == 0 or completed == len(subject_ids)
        ):
            progress(
                {
                    "stage": "strategy_scoring",
                    "processed_subjects": completed,
                    "total_subjects": len(subject_ids),
                    "device": engine.device,
                }
            )

    runs: dict[str, Any] = {}
    aggregate_inputs: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed in seeds:
        seed_result: dict[str, Any] = {}
        for strategy in strategies:
            strategy_result: dict[str, Any] = {}
            for calibration_far in calibration_fars:
                key = f"calibration_far_{calibration_far:.4f}"
                candidates = {
                    resolution: _threshold_for_far(
                        accumulator(
                            strategy.name,
                            seed,
                            resolution,
                            "validation",
                        ).impostor,
                        calibration_far,
                    )
                    for resolution in ("low", "medium")
                }
                threshold = max(candidates.values())
                conditions: dict[str, Any] = {}
                for resolution in ("low", "medium"):
                    conditions[resolution] = {
                        split: _metrics(
                            accumulator(
                                strategy.name,
                                seed,
                                resolution,
                                split,
                            ),
                            threshold,
                        )
                        for split in ("validation", "test")
                    }
                    for split in ("validation", "test"):
                        conditions[resolution][split]["query_coverage"] = 1.0
                test_tars = [
                    conditions[item]["test"]["tar"] for item in ("low", "medium")
                ]
                test_fars = [
                    conditions[item]["test"]["far"] for item in ("low", "medium")
                ]
                passed = min(test_tars) >= 0.90 and max(test_fars) <= target_far
                strategy_result[key] = {
                    "calibration_far": calibration_far,
                    "validation_threshold_candidates": candidates,
                    "operating_threshold": threshold,
                    "conditions": conditions,
                    "research_identity_gate": {
                        "target_minimum_tar": 0.90,
                        "target_maximum_far": target_far,
                        "observed_minimum_test_tar": min(test_tars),
                        "observed_maximum_test_far": max(test_fars),
                        "query_coverage": 1.0,
                        "passed": passed,
                    },
                }
                aggregate_key = f"{strategy.name}__{key}"
                inputs = aggregate_inputs[aggregate_key]
                inputs["threshold"].append(threshold)
                inputs["minimum_test_tar"].append(min(test_tars))
                inputs["maximum_test_far"].append(max(test_fars))
                inputs["low_test_tar"].append(conditions["low"]["test"]["tar"])
                inputs["medium_test_tar"].append(conditions["medium"]["test"]["tar"])
                inputs["low_test_far"].append(conditions["low"]["test"]["far"])
                inputs["medium_test_far"].append(conditions["medium"]["test"]["far"])
                inputs["gate"].append(float(passed))
            seed_result[strategy.name] = strategy_result
        runs[str(seed)] = seed_result

    strategy_lookup = {item.name: item for item in strategies}
    aggregates: dict[str, Any] = {}
    for aggregate_key, values in aggregate_inputs.items():
        strategy_name, margin_key = aggregate_key.split("__", 1)
        calibration_far = float(margin_key.removeprefix("calibration_far_"))
        metrics = {name: _summary(rows) for name, rows in values.items()}
        aggregates[aggregate_key] = {
            "strategy": asdict(strategy_lookup[strategy_name]),
            "calibration_far": calibration_far,
            "seed_count": len(seeds),
            "query_coverage": 1.0,
            "all_seeds_passed": all(item == 1.0 for item in values["gate"]),
            "metrics_across_seeds": metrics,
        }

    passed_keys = [key for key, item in aggregates.items() if item["all_seeds_passed"]]
    if passed_keys:
        chosen = max(
            passed_keys,
            key=lambda key: (
                aggregates[key]["metrics_across_seeds"]["minimum_test_tar"]["minimum"],
                -aggregates[key]["metrics_across_seeds"]["maximum_test_far"]["maximum"],
                aggregates[key]["calibration_far"],
            ),
        )
        recommendation = {
            "candidate": chosen,
            "status": "research_candidate_external_validation_required",
        }
    else:
        recommendation = {
            "candidate": None,
            "status": "no_enrollment_strategy_passed",
        }

    all_weights = np.concatenate(enrollment_weights)
    return {
        "dataset": "K-FACE",
        "protocol": "full_400_subject_enrollment_strategy_benchmark_v1",
        "pipeline_version": "kface-full-paired-v2",
        "input_subjects": len(subject_files),
        "eligible_subjects": len(subject_ids),
        "reference_count": reference_count,
        "seeds": list(seeds),
        "target_far": target_far,
        "calibration_fars": list(calibration_fars),
        "minimum_detection_score": minimum_detection_score,
        "histogram_bins": bins,
        "execution_device": engine.device,
        "query_coverage": 1.0,
        "strategies": [asdict(item) for item in strategies],
        "quality_weight_formula": {
            "detection": "clip((score - 0.50) / 0.40, 0.25, 1.0)",
            "face_size": "clip(face_pixel_side / 96, 0.25, 1.0)",
            "exposure": "clip(1 - abs(brightness - 127.5) / 127.5, 0.25, 1.0)",
            "combination": "geometric_mean(detection, face_size, exposure)",
            "observed_weights": _summary(all_weights.tolist()),
        },
        "dual_cluster_size": {
            name: _summary([float(item) for item in sizes])
            for name, sizes in dual_cluster_sizes.items()
        },
        "split_protocol": {
            "validation_subjects_per_seed": len(subject_ids) // 2,
            "test_subjects_per_seed": len(subject_ids) - len(subject_ids) // 2,
            "reference_query_image_overlap": 0,
            "benchmark_candidate_ranking_uses_repeated_test_metrics": True,
            "external_locked_test_required_before_api_change": True,
        },
        "runs": runs,
        "aggregates": aggregates,
        "recommendation": recommendation,
        "processing_seconds": time.perf_counter() - started,
        "contains_raw_paths": False,
        "contains_subject_identifiers": False,
        "contains_face_images": False,
        "contains_embeddings": False,
        "individual_scores_persisted": False,
        "threshold_status": "research_only_unapproved",
        "note": (
            "K-FACE 통제 촬영 데이터의 등록 결합 연구다. 동일 데이터에서 "
            "전략을 탐색했으므로 실제 웹·모바일 외부 검증 전에는 API 기본 "
            "등록 방식과 판정 기준값을 변경하지 않는다."
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-count", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260815, 20260816, 20260817, 20260818, 20260819],
    )
    parser.add_argument(
        "--calibration-fars",
        type=float,
        nargs="+",
        default=[0.0009, 0.0008, 0.0007],
    )
    parser.add_argument("--target-far", type=float, default=0.001)
    parser.add_argument("--minimum-detection-score", type=float, default=0.60)
    parser.add_argument("--bins", type=int, default=40_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    result = analyze_enrollment_strategies(
        args.input_dir,
        reference_count=args.reference_count,
        seeds=args.seeds,
        calibration_fars=args.calibration_fars,
        target_far=args.target_far,
        minimum_detection_score=args.minimum_detection_score,
        bins=args.bins,
        device=args.device,
        progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recommendation": result["recommendation"],
                "processing_minutes": round(result["processing_seconds"] / 60, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
