#!/usr/bin/env python3
"""Validate the leak-prevention and fair-comparison rules in the model plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "deepfake" / "model_improvement_plan.json"


class PlanValidationError(ValueError):
    """Raised when a model-improvement plan violates an experiment guardrail."""


def load_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PlanValidationError("계획 파일의 최상위 값은 객체여야 합니다.")
    return payload


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    protocol = plan.get("protocol", {})
    evaluation = plan.get("common_evaluation", {})
    gates = plan.get("gates", {})
    stages = plan.get("stages", [])
    selection = plan.get("selection", {})
    artifacts = plan.get("artifacts", {})

    required_true = (
        "split_before_frame_extraction",
        "same_manifest_for_every_candidate",
        "official_test_locked_until_candidate_frozen",
    )
    for key in required_true:
        if protocol.get(key) is not True:
            errors.append(f"protocol.{key}는 true여야 합니다.")

    required_false = (
        "official_test_used_for_training",
        "official_test_used_for_threshold_selection",
        "test_errors_recycled_into_training",
    )
    for key in required_false:
        if protocol.get(key) is not False:
            errors.append(f"protocol.{key}는 false여야 합니다.")

    if protocol.get("hard_negative_source") != "validation_false_positives_only":
        errors.append("hard negative는 validation 오경고에서만 골라야 합니다.")
    if evaluation.get("threshold_selected_on") != "validation":
        errors.append("판정 기준값은 validation에서만 선택해야 합니다.")
    if selection.get("rank_on") != "validation":
        errors.append("모델 순위는 validation 결과로만 정해야 합니다.")
    if gates.get("train_validation_video_overlap") != 0:
        errors.append("Train/Validation 영상 교집합 허용값은 0이어야 합니다.")
    if gates.get("train_validation_group_overlap") != 0:
        errors.append("Train/Validation 인물·원본 그룹 교집합 허용값은 0이어야 합니다.")

    stage_ids = [stage.get("id") for stage in stages if isinstance(stage, dict)]
    if not stage_ids or stage_ids[0] != plan.get("baseline", {}).get("candidate_id"):
        errors.append("첫 단계는 baseline.candidate_id와 같아야 합니다.")
    if len(stage_ids) != len(set(stage_ids)):
        errors.append("stage id는 중복될 수 없습니다.")

    known: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            errors.append("각 stage는 객체여야 합니다.")
            continue
        stage_id = stage.get("id")
        for dependency in stage.get("depends_on", []):
            if dependency not in known:
                errors.append(
                    f"{stage_id}의 의존 단계 {dependency}가 앞 단계에 없습니다."
                )
        if isinstance(stage_id, str):
            known.add(stage_id)

    for source in plan.get("data_sources", []):
        if source.get("raw_data_in_git") is not False:
            errors.append(f"{source.get('id', 'unknown')} 원본 데이터는 Git에 올릴 수 없습니다.")

    forbidden_committable = {
        "raw_video",
        "face_crop",
        "per_video_score",
        "checkpoint",
        "onnx_model",
        "split_manifest_with_identifiers",
    }
    leaked = forbidden_committable.intersection(artifacts.get("committable", []))
    if leaked:
        errors.append("비공개 산출물이 committable에 포함됐습니다: " + ", ".join(sorted(leaked)))

    required_metrics = {"video_roc_auc", "real_video_fpr", "fake_video_recall", "coverage"}
    missing_metrics = required_metrics.difference(evaluation.get("mandatory_metrics", []))
    if missing_metrics:
        errors.append("필수 지표가 빠졌습니다: " + ", ".join(sorted(missing_metrics)))

    return errors


def validate_file(path: Path) -> dict[str, Any]:
    plan = load_plan(path)
    errors = validate_plan(plan)
    if errors:
        raise PlanValidationError("\n".join(f"- {error}" for error in errors))
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="딥페이크 모델 고도화 계획을 검증합니다.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = validate_file(args.plan)
    print(
        json.dumps(
            {
                "status": "ok",
                "plan": str(args.plan),
                "stages": [stage["id"] for stage in plan["stages"]],
                "message": "데이터 누수 방지와 공정 비교 규칙을 통과했습니다.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
