#!/usr/bin/env python3
"""K-FACE 전체 특징값에서 얼굴가드 품질 Gate 후보를 반복 검증한다.

등록 5장 기준으로 검출점수, 실제 얼굴 픽셀 크기와 밝기 조합을 비교한다.
기준값과 Gate는 validation에서 선택하고 인물 단위 test에서 TAR/FAR와 자동
처리 coverage를 측정한다. 개별 인물, 임베딩과 비교 점수는 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
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
class QualityGateRule:
    """API에서 재현 가능한 얼굴 품질 하한 조합."""

    name: str
    minimum_detection_score: float = 0.60
    minimum_face_pixel_side: float = 0.0
    minimum_brightness: float = 0.0
    maximum_brightness: float = 255.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("품질 Gate 이름이 필요합니다.")
        if not 0 <= self.minimum_detection_score <= 1:
            raise ValueError("검출점수 하한은 0과 1 사이여야 합니다.")
        if self.minimum_face_pixel_side < 0:
            raise ValueError("얼굴 픽셀 크기 하한은 0 이상이어야 합니다.")
        if not 0 <= self.minimum_brightness <= self.maximum_brightness <= 255:
            raise ValueError("밝기 범위는 0~255 안에서 순서대로 지정해야 합니다.")

    def mask(self, quality: np.ndarray) -> np.ndarray:
        values = np.asarray(quality, dtype=np.float32)
        if values.ndim != 2 or values.shape[1:] != (6,):
            raise ValueError("품질값은 (N, 6) 형식이어야 합니다.")
        face_pixel_side = np.sqrt(values[:, 1] * values[:, 4] * values[:, 5])
        return (
            (values[:, 0] >= self.minimum_detection_score)
            & (face_pixel_side >= self.minimum_face_pixel_side)
            & (values[:, 3] >= self.minimum_brightness)
            & (values[:, 3] <= self.maximum_brightness)
        )


DEFAULT_RULES = (
    QualityGateRule("baseline_det060"),
    QualityGateRule("det070", minimum_detection_score=0.70),
    QualityGateRule("brightness20", minimum_brightness=20.0),
    QualityGateRule("brightness35", minimum_brightness=35.0),
    QualityGateRule("face_side38", minimum_face_pixel_side=38.0),
    QualityGateRule("face_side42", minimum_face_pixel_side=42.0),
    QualityGateRule("face_side46", minimum_face_pixel_side=46.0),
    QualityGateRule(
        "side38_brightness20",
        minimum_face_pixel_side=38.0,
        minimum_brightness=20.0,
    ),
    QualityGateRule(
        "side42_brightness35",
        minimum_face_pixel_side=42.0,
        minimum_brightness=35.0,
    ),
    QualityGateRule(
        "side46_brightness50",
        minimum_face_pixel_side=46.0,
        minimum_brightness=50.0,
    ),
    QualityGateRule(
        "det070_side42_brightness35",
        minimum_detection_score=0.70,
        minimum_face_pixel_side=42.0,
        minimum_brightness=35.0,
    ),
)

QUALITY_BINS: dict[str, tuple[float, ...]] = {
    "detection_score": (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.000001),
    "face_pixel_side": (
        0.0,
        32.0,
        36.0,
        38.0,
        40.0,
        42.0,
        44.0,
        46.0,
        48.0,
        52.0,
        56.0,
        64.0,
        80.0,
        96.0,
        128.0,
        float("inf"),
    ),
    "brightness_mean": (
        0.0,
        10.0,
        20.0,
        35.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
        256.0,
    ),
    "blur_score": (
        0.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1_000.0,
        3_000.0,
        10_000.0,
        float("inf"),
    ),
}


def _quality_column(quality: np.ndarray, name: str) -> np.ndarray:
    if name == "detection_score":
        return quality[:, 0]
    if name == "face_pixel_side":
        return np.sqrt(quality[:, 1] * quality[:, 4] * quality[:, 5])
    if name == "brightness_mean":
        return quality[:, 3]
    if name == "blur_score":
        return quality[:, 2]
    raise KeyError(name)


def _score_numpy(values: Any, engine: ScoreEngine) -> np.ndarray:
    if engine.device == "cuda":
        return values.detach().to(device="cpu").numpy()
    return np.asarray(values)


def _mask_rows(values: Any, mask: np.ndarray, engine: ScoreEngine) -> Any:
    if engine.device == "cuda":
        index = engine.torch.as_tensor(mask, dtype=engine.torch.bool, device="cuda")
        return values[index]
    return np.asarray(values)[mask]


def _quality_diagnostics(
    quality: np.ndarray,
    genuine_scores: np.ndarray,
    *,
    diagnostic_threshold: float,
) -> dict[str, Any]:
    quality = np.asarray(quality, dtype=np.float32)
    scores = np.asarray(genuine_scores, dtype=np.float32).reshape(-1)
    if quality.shape != (len(scores), 6) or not len(scores):
        raise ValueError("품질 진단에는 같은 수의 품질값과 본인 점수가 필요합니다.")
    result: dict[str, Any] = {}
    for name, raw_edges in QUALITY_BINS.items():
        values = _quality_column(quality, name)
        edges = np.asarray(raw_edges, dtype=np.float64)
        bins: list[dict[str, Any]] = []
        for lower, upper in pairwise(edges):
            mask = (values >= lower) & (values < upper)
            selected = scores[mask]
            if not len(selected):
                continue
            bins.append(
                {
                    "minimum": float(lower),
                    "maximum": None if math.isinf(upper) else float(upper),
                    "count": len(selected),
                    "share": float(len(selected) / len(scores)),
                    "mean_similarity": float(np.mean(selected)),
                    "p05_similarity": float(np.quantile(selected, 0.05)),
                    "median_similarity": float(np.median(selected)),
                    "p95_similarity": float(np.quantile(selected, 0.95)),
                    "match_rate_at_diagnostic_threshold": float(
                        np.mean(selected >= diagnostic_threshold)
                    ),
                }
            )
        result[name] = {
            "bins": bins,
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }
    return result


def _empty_histogram(bins: int) -> ScoreHistogram:
    return ScoreHistogram.empty(bins)


def analyze_quality_gates(
    input_dir: Path,
    *,
    rules: Sequence[QualityGateRule] = DEFAULT_RULES,
    reference_count: int = 5,
    seeds: Sequence[int] = (20260815, 20260816, 20260817, 20260818, 20260819),
    target_far: float = 0.001,
    calibration_far: float = 0.0009,
    baseline_detection_score: float = 0.60,
    bins: int = 40_000,
    device: str = "auto",
    diagnostic_threshold: float = 0.3784,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """고정 품질 Gate 후보를 validation/test에서 반복 평가한다."""

    rules = tuple(rules)
    seeds = tuple(dict.fromkeys(int(item) for item in seeds))
    if not rules or len({item.name for item in rules}) != len(rules):
        raise ValueError("서로 다른 이름의 품질 Gate가 하나 이상 필요합니다.")
    if reference_count <= 0 or not seeds:
        raise ValueError("등록 수와 seed가 필요합니다.")
    if not 0 < calibration_far <= target_far < 1:
        raise ValueError("calibration FAR은 0보다 크고 target FAR 이하여야 합니다.")
    if not 0 <= baseline_detection_score <= 1:
        raise ValueError("기준 검출점수는 0과 1 사이여야 합니다.")
    if bins < 1_000:
        raise ValueError("histogram bin은 1,000 이상이어야 합니다.")

    started = time.perf_counter()
    subject_files = discover_subject_files(input_dir)
    subject_ids = sorted(subject_files)
    centers: list[np.ndarray] = []
    used_indices: dict[str, set[int]] = {}
    eligible: list[str] = []

    for position, subject_id in enumerate(subject_ids, start=1):
        subject = _load_subject(subject_files[subject_id])
        available = np.flatnonzero(
            subject["medium_quality"][:, 0] >= baseline_detection_score
        )
        if len(available) < reference_count + 1:
            continue
        selected = available[_even_positions(len(available), reference_count)]
        centers.append(
            _unit_vector(np.mean(subject["medium_embeddings"][selected], axis=0))
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
        raise ValueError("품질 Gate 반복 검증에 필요한 인물이 부족합니다.")
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
    center_tensor = engine.centers(np.stack(centers))
    histograms: dict[tuple[str, int, str, str], ScoreHistogram] = {}
    coverage: dict[tuple[str, int, str, str], list[int]] = defaultdict(lambda: [0, 0])
    diagnostic_quality: dict[str, list[np.ndarray]] = defaultdict(list)
    diagnostic_scores: dict[str, list[np.ndarray]] = defaultdict(list)

    def accumulator(
        rule: str, seed: int, resolution: str, split: str
    ) -> ScoreHistogram:
        key = (rule, seed, resolution, split)
        if key not in histograms:
            histograms[key] = _empty_histogram(bins)
        return histograms[key]

    for completed, subject_id in enumerate(subject_ids, start=1):
        subject = _load_subject(subject_files[subject_id])
        own_position = subject_position[subject_id]
        excluded = used_indices[subject_id]
        for resolution in ("low", "medium"):
            quality = subject[f"{resolution}_quality"]
            baseline_mask = quality[:, 0] >= baseline_detection_score
            baseline_mask &= np.asarray(
                [int(item) not in excluded for item in subject["image_indices"]],
                dtype=bool,
            )
            baseline_positions = np.flatnonzero(baseline_mask)
            if not len(baseline_positions):
                raise ValueError(f"기준 품질 통과 질의가 없습니다: {subject_id}")
            baseline_quality = quality[baseline_positions]
            queries = subject[f"{resolution}_embeddings"][baseline_positions]
            scores = engine.scores(queries, center_tensor)
            genuine = engine.select_column(scores, own_position)
            diagnostic_quality[resolution].append(baseline_quality)
            diagnostic_scores[resolution].append(_score_numpy(genuine, engine))

            for seed in seeds:
                split = split_membership[seed][subject_id]
                for rule in rules:
                    coverage[(rule.name, seed, resolution, split)][1] += len(
                        baseline_positions
                    )

            for rule in rules:
                keep = rule.mask(baseline_quality)
                if not np.any(keep):
                    continue
                rule_scores = _mask_rows(scores, keep, engine)
                rule_genuine = _mask_rows(genuine, keep, engine)
                for seed in seeds:
                    split = split_membership[seed][subject_id]
                    coverage[(rule.name, seed, resolution, split)][0] += int(
                        np.sum(keep)
                    )
                    columns = [
                        item
                        for item in split_indices[seed][split]
                        if item != own_position
                    ]
                    item = accumulator(rule.name, seed, resolution, split)
                    item.genuine += engine.histogram(rule_genuine)
                    item.impostor += engine.histogram(
                        engine.select_columns(rule_scores, columns)
                    )
        if progress and (
            completed == 1 or completed % 10 == 0 or completed == len(subject_ids)
        ):
            progress(
                {
                    "stage": "quality_scoring",
                    "processed_subjects": completed,
                    "total_subjects": len(subject_ids),
                    "device": engine.device,
                }
            )

    diagnostics = {
        resolution: _quality_diagnostics(
            np.concatenate(diagnostic_quality[resolution]),
            np.concatenate(diagnostic_scores[resolution]),
            diagnostic_threshold=diagnostic_threshold,
        )
        for resolution in ("low", "medium")
    }

    runs: dict[str, Any] = {}
    aggregate_inputs: dict[str, dict[str, list[float]]] = {
        rule.name: defaultdict(list) for rule in rules
    }
    for seed in seeds:
        seed_result: dict[str, Any] = {}
        for rule in rules:
            candidates = {
                resolution: _threshold_for_far(
                    accumulator(rule.name, seed, resolution, "validation").impostor,
                    calibration_far,
                )
                for resolution in ("low", "medium")
            }
            threshold = max(candidates.values())
            conditions: dict[str, Any] = {}
            for resolution in ("low", "medium"):
                conditions[resolution] = {}
                for split in ("validation", "test"):
                    metrics = _metrics(
                        accumulator(rule.name, seed, resolution, split), threshold
                    )
                    kept, baseline = coverage[(rule.name, seed, resolution, split)]
                    metrics["query_coverage"] = kept / baseline
                    metrics["kept_queries"] = kept
                    metrics["baseline_queries"] = baseline
                    conditions[resolution][split] = metrics
            test_tars = [conditions[item]["test"]["tar"] for item in ("low", "medium")]
            test_fars = [conditions[item]["test"]["far"] for item in ("low", "medium")]
            test_coverage = [
                conditions[item]["test"]["query_coverage"] for item in ("low", "medium")
            ]
            passed = min(test_tars) >= 0.90 and max(test_fars) <= target_far
            seed_result[rule.name] = {
                "rule": asdict(rule),
                "validation_threshold_candidates": candidates,
                "operating_threshold": threshold,
                "conditions": conditions,
                "research_identity_gate": {
                    "target_minimum_tar": 0.90,
                    "target_maximum_far": target_far,
                    "observed_minimum_test_tar": min(test_tars),
                    "observed_maximum_test_far": max(test_fars),
                    "observed_minimum_test_coverage": min(test_coverage),
                    "passed": passed,
                },
            }
            inputs = aggregate_inputs[rule.name]
            inputs["threshold"].append(threshold)
            inputs["minimum_test_tar"].append(min(test_tars))
            inputs["maximum_test_far"].append(max(test_fars))
            inputs["minimum_test_coverage"].append(min(test_coverage))
            inputs["low_test_coverage"].append(test_coverage[0])
            inputs["medium_test_coverage"].append(test_coverage[1])
            inputs["gate"].append(float(passed))
        runs[str(seed)] = seed_result

    aggregates: dict[str, Any] = {}
    for rule in rules:
        values = aggregate_inputs[rule.name]
        metrics: dict[str, Any] = {}
        for name, rows in values.items():
            array = np.asarray(rows, dtype=np.float64)
            metrics[name] = {
                "minimum": float(np.min(array)),
                "median": float(np.median(array)),
                "maximum": float(np.max(array)),
            }
        aggregates[rule.name] = {
            "rule": asdict(rule),
            "seed_count": len(seeds),
            "all_seeds_passed": all(item == 1.0 for item in values["gate"]),
            "metrics_across_seeds": metrics,
        }

    passed_rules = [rule for rule in rules if aggregates[rule.name]["all_seeds_passed"]]
    recommendation: dict[str, Any]
    if passed_rules:
        chosen = max(
            passed_rules,
            key=lambda item: (
                aggregates[item.name]["metrics_across_seeds"]["minimum_test_coverage"][
                    "minimum"
                ],
                aggregates[item.name]["metrics_across_seeds"]["minimum_test_tar"][
                    "minimum"
                ],
                -aggregates[item.name]["metrics_across_seeds"]["maximum_test_far"][
                    "maximum"
                ],
            ),
        )
        recommendation = {
            "rule": chosen.name,
            "configuration": asdict(chosen),
            "status": "research_candidate_external_validation_required",
        }
    else:
        recommendation = {
            "rule": None,
            "configuration": None,
            "status": "no_quality_gate_passed",
        }

    return {
        "dataset": "K-FACE",
        "protocol": "full_400_subject_quality_gate_sweep_v1",
        "input_subjects": len(subject_files),
        "eligible_subjects": len(subject_ids),
        "reference_count": reference_count,
        "seeds": list(seeds),
        "target_far": target_far,
        "calibration_far": calibration_far,
        "baseline_detection_score": baseline_detection_score,
        "diagnostic_threshold": diagnostic_threshold,
        "histogram_bins": bins,
        "execution_device": engine.device,
        "rules": [asdict(item) for item in rules],
        "quality_diagnostics": diagnostics,
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
            "K-FACE 내부 품질 Gate 탐색이다. 실제 웹·모바일 외부 검증과 "
            "제품 coverage 기준 합의 전에는 API 기본 동작을 변경하지 않는다."
        ),
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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-count", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260815, 20260816, 20260817, 20260818, 20260819],
    )
    parser.add_argument("--target-far", type=float, default=0.001)
    parser.add_argument("--calibration-far", type=float, default=0.0009)
    parser.add_argument("--baseline-detection-score", type=float, default=0.60)
    parser.add_argument("--bins", type=int, default=40_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--diagnostic-threshold", type=float, default=0.3784)
    args = parser.parse_args(argv)

    result = analyze_quality_gates(
        args.input_dir,
        reference_count=args.reference_count,
        seeds=args.seeds,
        target_far=args.target_far,
        calibration_far=args.calibration_far,
        baseline_detection_score=args.baseline_detection_score,
        bins=args.bins,
        device=args.device,
        diagnostic_threshold=args.diagnostic_threshold,
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
