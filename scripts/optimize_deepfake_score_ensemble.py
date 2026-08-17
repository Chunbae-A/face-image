#!/usr/bin/env python3
"""Validation 점수만으로 EfficientNet-B4와 Xception 결합 정책을 고른다.

입력 CSV에는 영상 ID와 프레임별 점수가 있어 비공개 런타임에만 둔다.
출력 JSON에는 조건별 집계 지표만 기록하며 영상·프레임 식별자는 남기지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from celebdf_deepfake import (
        ScoreRecord,
        aggregate_video_scores,
        classification_metrics,
        latency_summary,
        operating_point_at_recall,
        read_score_records,
    )
except ModuleNotFoundError:  # importlib로 불러오는 테스트 환경
    from scripts.celebdf_deepfake import (
        ScoreRecord,
        aggregate_video_scores,
        classification_metrics,
        latency_summary,
        operating_point_at_recall,
        read_score_records,
    )


EPSILON = 1e-7


class EnsembleError(ValueError):
    """안전한 비교 규칙을 만족하지 못한 입력에 사용한다."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise EnsembleError("ensemble config must be a JSON object")
    return payload


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_key(record: ScoreRecord) -> tuple[str, str, str, int]:
    return (record.split, record.condition, record.video_id, record.frame_index)


def _index_records(
    model_id: str,
    records: Sequence[ScoreRecord],
) -> dict[tuple[str, str, str, int], ScoreRecord]:
    if not records:
        raise EnsembleError(f"{model_id}: score CSV is empty")
    indexed: dict[tuple[str, str, str, int], ScoreRecord] = {}
    for record in records:
        if record.split != "validation":
            raise EnsembleError(
                f"{model_id}: only validation scores are allowed; official test is locked"
            )
        if record.label not in {0, 1}:
            raise EnsembleError(f"{model_id}: labels must use 0=real and 1=fake")
        if not math.isfinite(record.score) or not 0.0 <= record.score <= 1.0:
            raise EnsembleError(f"{model_id}: scores must be finite values in [0, 1]")
        if not math.isfinite(record.latency_ms) or record.latency_ms < 0.0:
            raise EnsembleError(f"{model_id}: latency must be a finite non-negative value")
        key = _score_key(record)
        if key in indexed:
            raise EnsembleError(f"{model_id}: duplicate validation frame key")
        indexed[key] = record
    return indexed


def align_score_records(
    primary_model: str,
    primary: Sequence[ScoreRecord],
    specialist_model: str,
    specialist: Sequence[ScoreRecord],
    *,
    expected_conditions: Sequence[str],
) -> list[tuple[ScoreRecord, ScoreRecord]]:
    """두 모델의 입력 단위가 정확히 같을 때만 정렬된 점수를 반환한다."""

    primary_index = _index_records(primary_model, primary)
    specialist_index = _index_records(specialist_model, specialist)
    if set(primary_index) != set(specialist_index):
        missing_primary = len(set(specialist_index).difference(primary_index))
        missing_specialist = len(set(primary_index).difference(specialist_index))
        raise EnsembleError(
            "candidate frame keys do not match: "
            f"missing_primary={missing_primary}, missing_specialist={missing_specialist}"
        )
    found_conditions = {key[1] for key in primary_index}
    if found_conditions != set(expected_conditions):
        raise EnsembleError(
            "validation conditions differ from config: "
            f"found={sorted(found_conditions)}"
        )
    aligned: list[tuple[ScoreRecord, ScoreRecord]] = []
    for key in sorted(primary_index):
        left = primary_index[key]
        right = specialist_index[key]
        if left.label != right.label:
            raise EnsembleError("candidate labels do not match for aligned frames")
        aligned.append((left, right))
    return aligned


def _logit(score: float) -> float:
    clipped = min(max(float(score), EPSILON), 1.0 - EPSILON)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def fuse_aligned_records(
    aligned: Sequence[tuple[ScoreRecord, ScoreRecord]],
    *,
    primary_weight: float,
    specialist_conditions: Sequence[str] | None,
) -> list[ScoreRecord]:
    """프레임별 확률형 점수를 logit으로 바꾼 뒤 가중 결합한다.

    ``specialist_conditions=None``이면 모든 조건에서 두 모델을 실행한다.
    조건 목록을 주면 해당 조건에서만 specialist를 실행하고 나머지는 primary와
    완전히 같은 점수·지연시간을 유지한다.
    """

    if not 0.0 <= primary_weight <= 1.0:
        raise EnsembleError("primary_weight must be in [0, 1]")
    specialist_set = None if specialist_conditions is None else set(specialist_conditions)
    fused: list[ScoreRecord] = []
    for primary, specialist in aligned:
        use_specialist = specialist_set is None or primary.condition in specialist_set
        if use_specialist and primary_weight < 1.0:
            score = _sigmoid(
                primary_weight * _logit(primary.score)
                + (1.0 - primary_weight) * _logit(specialist.score)
            )
            latency_ms = primary.latency_ms + specialist.latency_ms
        else:
            score = primary.score
            latency_ms = primary.latency_ms
        fused.append(replace(primary, score=score, latency_ms=latency_ms))
    return fused


def _arrays(records: Sequence[ScoreRecord]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([record.label for record in records], dtype=np.int8),
        np.asarray([record.score for record in records], dtype=np.float64),
    )


def evaluate_policy(
    policy_id: str,
    records: Sequence[ScoreRecord],
    *,
    aggregation: str,
    target_recall: float,
    policy_type: str,
    primary_weight: float,
    specialist_conditions: Sequence[str],
) -> dict[str, Any]:
    clean_frames = [record for record in records if record.condition == "clean"]
    if not clean_frames:
        raise EnsembleError("clean validation scores are required")
    clean_videos = aggregate_video_scores(clean_frames, method=aggregation)
    clean_labels, clean_scores = _arrays(clean_videos)
    operating_point = operating_point_at_recall(
        clean_labels,
        clean_scores,
        target_recall,
    )
    threshold = float(operating_point["threshold"])
    conditions: dict[str, dict[str, Any]] = {}
    for condition in sorted({record.condition for record in records}):
        frames = [record for record in records if record.condition == condition]
        videos = aggregate_video_scores(frames, method=aggregation)
        labels, scores = _arrays(videos)
        conditions[condition] = {
            "video": classification_metrics(labels, scores, threshold=threshold),
            "operating_point_at_target_recall": operating_point_at_recall(
                labels,
                scores,
                target_recall,
            ),
            "latency": latency_summary(videos),
        }
    active_conditions = (
        sorted(conditions)
        if policy_type == "static_logit_fusion" and primary_weight < 1.0
        else sorted(specialist_conditions)
        if policy_type == "condition_aware_logit_fusion" and primary_weight < 1.0
        else []
    )
    return {
        "policy_id": policy_id,
        "policy_type": policy_type,
        "fusion_space": "logit",
        "primary_weight": float(primary_weight),
        "specialist_active_conditions": active_conditions,
        "specialist_route_fraction_in_validation_grid": (
            len(active_conditions) / len(conditions) if conditions else 0.0
        ),
        "route_fraction_is_live_traffic_estimate": False,
        "selected_threshold_on_clean_validation": threshold,
        "clean_operating_point_at_target_recall": operating_point,
        "condition_validation": conditions,
    }


def _candidate_policies(config: dict[str, Any]) -> list[tuple[str, str, float, list[str] | None]]:
    weights = sorted({float(value) for value in config["primary_weight_grid"]})
    if not weights or weights[0] < 0.0 or weights[-1] > 1.0 or 1.0 not in weights:
        raise EnsembleError("primary_weight_grid must contain 1.0 and stay within [0, 1]")
    specialist_conditions = list(config["specialist_conditions"])
    policies: list[tuple[str, str, float, list[str] | None]] = [
        ("primary_only", "primary_only", 1.0, specialist_conditions)
    ]
    for weight in weights:
        if weight >= 1.0:
            continue
        suffix = str(weight).replace(".", "_")
        policies.append(
            (f"static_primary_weight_{suffix}", "static_logit_fusion", weight, None)
        )
        policies.append(
            (
                f"conditional_primary_weight_{suffix}",
                "condition_aware_logit_fusion",
                weight,
                specialist_conditions,
            )
        )
    return policies


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("selection_split") != "validation":
        raise EnsembleError("selection_split must be validation")
    if config.get("official_test_used_for_selection") is not False:
        raise EnsembleError("official test cannot be used for ensemble selection")
    conditions = list(config.get("conditions", []))
    specialist = list(config.get("specialist_conditions", []))
    if "clean" not in conditions or not specialist:
        raise EnsembleError("clean and at least one specialist condition are required")
    if "clean" in specialist or not set(specialist).issubset(conditions):
        raise EnsembleError("specialist conditions must be non-clean configured conditions")
    if config.get("aggregation") != "mean":
        raise EnsembleError("this experiment requires mean video aggregation")
    if config.get("fusion_space") != "logit":
        raise EnsembleError("this experiment requires logit fusion")


def build_ensemble_report(
    primary_records: Sequence[ScoreRecord],
    specialist_records: Sequence[ScoreRecord],
    config: dict[str, Any],
) -> dict[str, Any]:
    """정렬·결합·선택을 수행하고 식별자가 없는 집계 보고서를 만든다."""

    _validate_config(config)
    primary_model = str(config["primary_model"])
    specialist_model = str(config["specialist_model"])
    conditions = list(config["conditions"])
    specialist_conditions = list(config["specialist_conditions"])
    aligned = align_score_records(
        primary_model,
        primary_records,
        specialist_model,
        specialist_records,
        expected_conditions=conditions,
    )
    target_recall = float(config["selection_target_recall"])
    reports: list[dict[str, Any]] = []
    for policy_id, policy_type, weight, routed_conditions in _candidate_policies(config):
        records = fuse_aligned_records(
            aligned,
            primary_weight=weight,
            specialist_conditions=routed_conditions,
        )
        reports.append(
            evaluate_policy(
                policy_id,
                records,
                aggregation=str(config["aggregation"]),
                target_recall=target_recall,
                policy_type=policy_type,
                primary_weight=weight,
                specialist_conditions=specialist_conditions,
            )
        )

    by_id = {report["policy_id"]: report for report in reports}
    baseline = by_id["primary_only"]
    baseline_clean = baseline["clean_operating_point_at_target_recall"]
    specialist_condition = specialist_conditions[0]
    baseline_specialist = baseline["condition_validation"][specialist_condition]
    non_specialist_conditions = [
        condition
        for condition in conditions
        if condition not in specialist_conditions and condition != "clean"
    ]
    eligible: list[tuple[float, float, int, float, str]] = []
    for report in reports:
        clean = report["clean_operating_point_at_target_recall"]
        specialist = report["condition_validation"][specialist_condition]
        specialist_video = specialist["video"]
        specialist_operating = specialist["operating_point_at_target_recall"]
        auc_improvement = float(
            specialist_video["roc_auc"]
            - baseline_specialist["video"]["roc_auc"]
        )
        fpr_improvement = float(
            baseline_specialist["operating_point_at_target_recall"]["fpr"]
            - specialist_operating["fpr"]
        )
        non_specialist_auc_regressions = {
            condition: float(
                baseline["condition_validation"][condition]["video"]["roc_auc"]
                - report["condition_validation"][condition]["video"]["roc_auc"]
            )
            for condition in non_specialist_conditions
        }
        checks = {
            "clean_target_recall_pass": bool(clean["recall"] >= target_recall),
            "clean_fpr_not_worse": bool(
                clean["fpr"]
                <= baseline_clean["fpr"]
                + float(config["maximum_clean_fpr_regression"])
                + 1e-12
            ),
            "specialist_target_recall_pass": bool(
                specialist_operating["recall"] >= target_recall
            ),
            "specialist_improved": bool(
                auc_improvement >= float(config["minimum_specialist_auc_improvement"])
                or fpr_improvement >= float(config["minimum_specialist_fpr_improvement"])
            ),
            "non_specialist_auc_not_worse": bool(
                all(
                    regression
                    <= float(config["maximum_non_specialist_auc_regression"]) + 1e-12
                    for regression in non_specialist_auc_regressions.values()
                )
            ),
        }
        if report["policy_id"] == "primary_only":
            checks["specialist_improved"] = True
        report["comparison_to_primary"] = {
            "specialist_condition": specialist_condition,
            "specialist_auc_improvement": auc_improvement,
            "specialist_fpr_improvement_at_target_recall": fpr_improvement,
            "non_specialist_auc_regressions": non_specialist_auc_regressions,
            "checks": checks,
            "eligible": bool(all(checks.values())),
        }
        if report["policy_id"] != "primary_only" and all(checks.values()):
            type_priority = 0 if report["policy_type"] == "condition_aware_logit_fusion" else 1
            specialist_latency = float(specialist["latency"]["p95_ms"])
            eligible.append(
                (
                    float(specialist_operating["fpr"]),
                    -float(specialist_video["roc_auc"]),
                    type_priority,
                    specialist_latency,
                    str(report["policy_id"]),
                )
            )

    selected_policy = min(eligible)[-1] if eligible else "primary_only"
    paired_key_fingerprint = _canonical_hash(
        [
            [left.split, left.condition, left.video_id, left.frame_index, left.label]
            for left, _right in aligned
        ]
    )
    report: dict[str, Any] = {
        "status": "completed",
        "evaluation_scope": "validation_only_ensemble_selection",
        "selection_split": "validation",
        "official_test_used_for_selection": False,
        "official_test_inference_performed": False,
        "experiment_id": config["experiment_id"],
        "primary_model": primary_model,
        "specialist_model": specialist_model,
        "fusion_space": "logit",
        "aggregation": config["aggregation"],
        "selection_target_recall": target_recall,
        "conditions": conditions,
        "specialist_conditions": specialist_conditions,
        "paired_frame_count": len(aligned),
        "paired_video_count": len({left.video_id for left, _right in aligned}),
        "paired_input_fingerprint_sha256": paired_key_fingerprint,
        "candidate_policies": reports,
        "selected_policy": selected_policy,
        "ensemble_selected": selected_policy != "primary_only",
        "selection_rule": [
            "clean recall >= target and clean FPR does not regress",
            "specialist condition improves AUC or FPR at target recall",
            "non-specialist condition AUC does not regress beyond tolerance",
            "minimum specialist FPR at target recall",
            "maximum specialist ROC-AUC",
            "prefer condition-aware route over always-on route",
            "minimum specialist-route p95 latency",
        ],
        "api_policy_if_selected": {
            "default_route": primary_model,
            "specialist_route": specialist_model,
            "specialist_trigger": "strong_jpeg_compression_quality_gate",
            "fallback_on_specialist_failure": "return primary result with review warning",
            "automatic_blocking_approved": False,
        },
        "external_validation_pending": True,
        "operationally_approved": False,
        "privacy": {
            "contains_video_ids": False,
            "contains_frame_scores": False,
            "contains_face_crops": False,
            "contains_checkpoints": False,
            "contains_onnx_models": False,
        },
        "config_sha256": _canonical_hash(config),
    }
    report["selection_fingerprint_sha256"] = _canonical_hash(report)
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", type=Path, required=True)
    parser.add_argument("--specialist-scores", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_json(args.config)
    report = build_ensemble_report(
        read_score_records(args.primary_scores),
        read_score_records(args.specialist_scores),
        config,
    )
    write_report(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_policy": report["selected_policy"],
                "ensemble_selected": report["ensemble_selected"],
                "official_test_used_for_selection": False,
                "selection_fingerprint_sha256": report[
                    "selection_fingerprint_sha256"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
