#!/usr/bin/env python3
"""비공개 Celeb-DF 점수를 화면용 확률 후보로 보정하고 비식별 결과만 저장한다."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from celebdf_deepfake import (
        FAKE_LABEL,
        REAL_LABEL,
        aggregate_video_scores,
        classification_metrics,
        read_score_records,
        threshold_at_fpr,
    )
except ModuleNotFoundError:  # 모듈로 불러오는 테스트·노트북 환경
    from scripts.celebdf_deepfake import (
        FAKE_LABEL,
        REAL_LABEL,
        aggregate_video_scores,
        classification_metrics,
        read_score_records,
        threshold_at_fpr,
    )


EPSILON = 1e-7


@dataclass(frozen=True)
class CalibrationModel:
    method: str
    parameters: dict[str, object]


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1.0 - EPSILON)


def _logit(values: np.ndarray) -> np.ndarray:
    probabilities = _clip(values)
    return np.log(probabilities / (1.0 - probabilities))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = _clip(probabilities)
    if labels.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same 1-D shape")
    if not np.all(np.isin(labels, [REAL_LABEL, FAKE_LABEL])):
        raise ValueError("labels must use 0=real and 1=fake")
    if bins <= 1:
        raise ValueError("bins must be greater than one")

    nll = float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )
    brier = float(np.mean((probabilities - labels) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        mask = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == bins - 1
            else (probabilities >= lower) & (probabilities < upper)
        )
        count = int(np.sum(mask))
        if count:
            confidence = float(np.mean(probabilities[mask]))
            observed_rate = float(np.mean(labels[mask]))
            ece += count / len(labels) * abs(confidence - observed_rate)
        else:
            confidence = 0.0
            observed_rate = 0.0
        bin_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": confidence,
                "observed_fake_rate": observed_rate,
            }
        )
    return {"nll": nll, "brier": brier, "ece": float(ece), "bins": bin_rows}


def fit_temperature(labels: np.ndarray, raw_scores: np.ndarray) -> CalibrationModel:
    labels = np.asarray(labels, dtype=np.float64)
    logits = _logit(raw_scores)

    def loss(log_temperature: float) -> float:
        probabilities = _clip(_sigmoid(logits / math.exp(log_temperature)))
        return float(
            -np.mean(
                labels * np.log(probabilities)
                + (1.0 - labels) * np.log(1.0 - probabilities)
            )
        )

    lower, upper = -5.0, 5.0
    best = 0.0
    for _ in range(5):
        candidates = np.linspace(lower, upper, 201)
        losses = np.asarray([loss(float(value)) for value in candidates])
        best_index = int(np.argmin(losses))
        best = float(candidates[best_index])
        step = float(candidates[1] - candidates[0])
        lower, upper = best - step, best + step
    return CalibrationModel(
        method="temperature",
        parameters={"temperature": float(math.exp(best))},
    )


def fit_platt(labels: np.ndarray, raw_scores: np.ndarray) -> CalibrationModel:
    labels = np.asarray(labels, dtype=np.float64)
    logits = _logit(raw_scores)
    design = np.column_stack([logits, np.ones_like(logits)])
    parameters = np.asarray([1.0, 0.0], dtype=np.float64)
    regularization = np.asarray([1e-4, 1e-6], dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        probabilities = _clip(_sigmoid(design @ candidate))
        nll = -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
        return float(nll + 0.5 * np.sum(regularization * candidate**2))

    for _ in range(100):
        probabilities = _sigmoid(design @ parameters)
        gradient = design.T @ (probabilities - labels) / len(labels)
        gradient += regularization * parameters
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        hessian = (design.T * weights) @ design / len(labels)
        hessian += np.diag(regularization)
        step = np.linalg.solve(hessian, gradient)
        if float(np.linalg.norm(step)) < 1e-10:
            break
        current = objective(parameters)
        scale = 1.0
        while scale >= 1e-6:
            candidate = parameters - scale * step
            if objective(candidate) <= current:
                parameters = candidate
                break
            scale *= 0.5
        else:
            break
    return CalibrationModel(
        method="platt",
        parameters={
            "slope": float(parameters[0]),
            "intercept": float(parameters[1]),
        },
    )


def fit_isotonic(labels: np.ndarray, raw_scores: np.ndarray) -> CalibrationModel:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(raw_scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    unique_scores, inverse = np.unique(sorted_scores, return_inverse=True)
    sums = np.bincount(inverse, weights=sorted_labels).astype(np.float64)
    weights = np.bincount(inverse).astype(np.float64)

    blocks: list[list[float]] = []
    for score, total, weight in zip(unique_scores, sums, weights, strict=True):
        blocks.append([float(score), float(total), float(weight)])
        while (
            len(blocks) >= 2
            and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                [right[0], left[1] + right[1], left[2] + right[2]]
            )
    compressed: list[list[float]] = []
    for block in blocks:
        if (
            compressed
            and math.isclose(
                compressed[-1][1] / compressed[-1][2],
                block[1] / block[2],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            compressed[-1][0] = block[0]
            compressed[-1][1] += block[1]
            compressed[-1][2] += block[2]
        else:
            compressed.append(block.copy())
    blocks = compressed
    boundaries = [block[0] for block in blocks]
    values = [block[1] / block[2] for block in blocks]
    return CalibrationModel(
        method="isotonic",
        parameters={"boundaries": boundaries, "values": values},
    )


def predict(model: CalibrationModel, raw_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64)
    if model.method == "temperature":
        temperature = float(model.parameters["temperature"])
        return _sigmoid(_logit(scores) / temperature)
    if model.method == "platt":
        slope = float(model.parameters["slope"])
        intercept = float(model.parameters["intercept"])
        return _sigmoid(slope * _logit(scores) + intercept)
    boundaries = np.asarray(model.parameters["boundaries"], dtype=np.float64)
    values = np.asarray(model.parameters["values"], dtype=np.float64)
    indices = np.searchsorted(boundaries, scores, side="left")
    return values[np.clip(indices, 0, len(values) - 1)]


def threshold_at_fnr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fnr: float,
) -> float:
    if not 0.0 <= target_fnr < 1.0:
        raise ValueError("target_fnr must be in [0, 1)")
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    fake_scores = np.sort(scores[labels == FAKE_LABEL])
    if not len(fake_scores):
        raise ValueError("fake samples are required to set an FNR threshold")
    candidates = np.unique(fake_scores)
    eligible = [
        float(threshold)
        for threshold in candidates
        if float(np.mean(fake_scores < threshold)) <= target_fnr
    ]
    return max(eligible) if eligible else float(fake_scores[0])


def build_calibration_report(
    labels_validation: np.ndarray,
    scores_validation: np.ndarray,
    labels_test: np.ndarray,
    scores_test: np.ndarray,
    *,
    model_fingerprint: str,
    calibration_version: str,
    target_fpr: float = 0.01,
    target_fnr: float = 0.05,
) -> dict[str, object]:
    models = (
        fit_temperature(labels_validation, scores_validation),
        fit_platt(labels_validation, scores_validation),
        fit_isotonic(labels_validation, scores_validation),
    )
    validation_before = calibration_metrics(labels_validation, scores_validation)
    test_before = calibration_metrics(labels_test, scores_test)
    comparisons: dict[str, dict[str, object]] = {}
    for model in models:
        validation_probability = predict(model, scores_validation)
        test_probability = predict(model, scores_test)
        comparisons[model.method] = {
            "parameters": model.parameters,
            "validation": calibration_metrics(
                labels_validation, validation_probability
            ),
            "official_test": calibration_metrics(labels_test, test_probability),
        }
    # 모델 선택에는 validation만 사용한다. official test는 최종 평가 전용이다.
    selected_method = min(
        comparisons,
        key=lambda method: (
            float(comparisons[method]["validation"]["brier"]),
            float(comparisons[method]["validation"]["ece"]),
            float(comparisons[method]["validation"]["nll"]),
            method,
        ),
    )
    selected = comparisons[selected_method]
    high_threshold = threshold_at_fpr(
        labels_validation, scores_validation, target_fpr
    )
    low_threshold = threshold_at_fnr(
        labels_validation, scores_validation, target_fnr
    )
    low_threshold = min(low_threshold, high_threshold)
    test_decision = classification_metrics(
        labels_test,
        scores_test,
        threshold=high_threshold,
    )
    validation_ece_pass = float(selected["validation"]["ece"]) <= 0.05
    test_ece_pass = float(selected["official_test"]["ece"]) <= 0.05
    test_fpr_pass = float(test_decision["fpr"]) <= target_fpr
    display_approved = bool(validation_ece_pass and test_ece_pass and test_fpr_pass)
    status = "validated" if display_approved else "research_only_unapproved"
    warning = (
        "Celeb-DF-v2 검증·공식 Test Gate를 통과한 영상 보정 확률입니다. "
        "실제 웹·한국인·최신 생성 방식에서는 재검증이 필요합니다."
        if display_approved
        else "점수 보정 실험은 완료했지만 ECE 또는 실제영상 오경고율 Gate를 통과하지 "
        "못했습니다. 보정 확률을 화면에 표시하지 말고 원점수와 검토 필요 문구만 사용하세요."
    )
    return {
        "schema_version": "1.0",
        "calibration_version": calibration_version,
        "scope": "deepfake_video_mean_16_frames",
        "model_fingerprint": model_fingerprint,
        "selection_split": "validation",
        "evaluation_split": "official_test",
        "official_test_used_for_selection": False,
        "raw_score_definition": "16개 대표 얼굴 프레임 sigmoid 점수의 산술평균",
        "selected_method": selected_method,
        "parameters": selected["parameters"],
        "calibration_status": status,
        "display_approved": display_approved,
        "warning": warning,
        "risk_bands": {
            "selection_split": "validation",
            "low_max_raw_score": float(low_threshold),
            "low_rule": f"validation fake FNR <= {target_fnr:g}를 만족하는 최대 기준값 미만",
            "high_min_raw_score": float(high_threshold),
            "high_rule": f"validation real FPR <= {target_fpr:g} 목표 기준값 이상",
            "review_rule": "low와 high 사이, 사람 확인 필요",
        },
        "metrics": {
            "validation_count": len(labels_validation),
            "official_test_count": len(labels_test),
            "before": {
                "validation": validation_before,
                "official_test": test_before,
            },
            "method_comparison": comparisons,
            "selected": {
                "validation": selected["validation"],
                "official_test": selected["official_test"],
            },
            "official_test_decision": test_decision,
        },
        "gate": {
            "ece_maximum": 0.05,
            "real_video_fpr_maximum": target_fpr,
            "validation_ece_pass": validation_ece_pass,
            "official_test_ece_pass": test_ece_pass,
            "official_test_real_fpr_pass": test_fpr_pass,
            "overall_pass": display_approved,
        },
        "privacy": {
            "contains_video_ids": False,
            "contains_frame_scores": False,
            "contains_faces_or_embeddings": False,
        },
    }


def plot_reliability(report: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = report["metrics"]["selected"]
    before = report["metrics"]["before"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, split, title in zip(
        axes,
        ("validation", "official_test"),
        ("Validation", "Official Test"),
        strict=True,
    ):
        for label, metrics, marker in (
            ("Before", before[split], "o"),
            ("After", selected[split], "s"),
        ):
            rows = [row for row in metrics["bins"] if row["count"]]
            axis.plot(
                [row["mean_probability"] for row in rows],
                [row["observed_fake_rate"] for row in rows],
                marker=marker,
                label=f"{label} (ECE={metrics['ece']:.3f})",
            )
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
        axis.set(
            title=title,
            xlabel="Mean predicted probability",
            ylabel="Observed fake rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle(f"Deepfake score calibration: {report['selected_method']}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument(
        "--calibration-version",
        default="celebdf-video-mean16-2026-08-08-v1",
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--target-fnr", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = [
        record
        for record in read_score_records(args.private_scores)
        if record.condition == "clean"
    ]
    videos = aggregate_video_scores(records, method="mean")
    validation = [row for row in videos if row.split == "validation"]
    official_test = [row for row in videos if row.split == "test"]
    if not validation or not official_test:
        raise ValueError("clean validation and official test video scores are required")
    labels_validation = np.asarray([row.label for row in validation], dtype=np.int8)
    scores_validation = np.asarray([row.score for row in validation], dtype=np.float64)
    labels_test = np.asarray([row.label for row in official_test], dtype=np.int8)
    scores_test = np.asarray([row.score for row in official_test], dtype=np.float64)
    report = build_calibration_report(
        labels_validation,
        scores_validation,
        labels_test,
        scores_test,
        model_fingerprint=args.model_fingerprint,
        calibration_version=args.calibration_version,
        target_fpr=args.target_fpr,
        target_fnr=args.target_fnr,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.figure:
        plot_reliability(report, args.figure)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_method": report["selected_method"],
                "calibration_status": report["calibration_status"],
                "display_approved": report["display_approved"],
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
