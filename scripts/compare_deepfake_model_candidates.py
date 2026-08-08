#!/usr/bin/env python3
"""Select one deepfake candidate from sanitized validation-only metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


class ComparisonError(ValueError):
    """Raised when candidate results cannot be compared fairly."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ComparisonError(f"candidate metrics must be an object: {path}")
    return payload


def parse_candidate(value: str) -> tuple[str, Path]:
    candidate_id, separator, raw_path = value.partition("=")
    if not separator or not candidate_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("candidate must use id=/path/to/metrics.json")
    return candidate_id.strip(), Path(raw_path).expanduser()


def _metric(payload: dict[str, Any], *keys: str) -> float:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ComparisonError("missing metric: " + ".".join(keys))
        value = value[key]
    return float(value)


def _validate_candidate(candidate_id: str, payload: dict[str, Any]) -> None:
    checks = {
        "evaluation_scope": "validation_only",
        "selection_split": "validation",
        "official_test_used_for_selection": False,
        "official_test_inference_performed": False,
        "architecture_id": candidate_id,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ComparisonError(
                f"{candidate_id}: {key} must be {expected!r}, found {payload.get(key)!r}"
            )
    conditions = payload.get("condition_validation")
    if not isinstance(conditions, dict) or "clean" not in conditions:
        raise ComparisonError(f"{candidate_id}: validation conditions are incomplete")
    _metric(payload, "validation_operating_point_at_recall_0_95", "fpr")
    _metric(payload, "validation_video_latency", "p95_ms")


def _expected_fairness(config: dict[str, Any]) -> dict[str, Any]:
    """Build the exact shared protocol required by the comparison config."""

    if float(config.get("selection_target_recall", -1.0)) != 0.95:
        raise ComparisonError(
            "this report schema requires selection_target_recall to be 0.95"
        )
    required_controls = {
        "same_crop_manifest_sha256",
        "same_train_validation_split",
        "same_seed",
        "same_input_size",
        "same_normalization",
        "same_frame_budget",
        "same_augmentation",
        "same_optimizer_and_learning_rate",
        "same_epoch_and_early_stopping_budget",
        "same_validation_metrics",
    }
    controls = set(config.get("fairness_controls", []))
    missing_controls = required_controls.difference(controls)
    if missing_controls:
        raise ComparisonError(
            "comparison config is missing fairness controls: "
            + ", ".join(sorted(missing_controls))
        )
    frame_counts = list(config.get("evaluation_frame_counts", []))
    if len(frame_counts) != 1:
        raise ComparisonError("fair comparison requires one fixed evaluation frame count")
    return {
        "seed": config["seed"],
        "input_size": config["input_size"],
        "normalization": config["normalization"],
        "train_frames_per_video": config["train_frames_per_video"],
        "target_fpr": config["target_validation_fpr"],
        "selected_frames_per_video": frame_counts[0],
        "aggregation_candidates": config["aggregation_methods"],
        "training_protocol": {
            "optimizer": config["optimizer"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "epochs": config["epochs"],
            "early_stopping_patience": config["early_stopping_patience"],
            "batch_size": config["batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "augmentation": config["augmentation"],
            "dataloader_seed": config["seed"],
            "sampler_seed": config["seed"],
        },
    }


def build_comparison(
    candidates: Sequence[tuple[str, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate fair inputs and freeze the best validation-only candidate."""

    if len(candidates) < 2:
        raise ComparisonError("at least two candidates are required")
    ids = [candidate_id for candidate_id, _ in candidates]
    if len(ids) != len(set(ids)):
        raise ComparisonError("candidate ids must be unique")
    for candidate_id, payload in candidates:
        _validate_candidate(candidate_id, payload)

    configured_candidates = {row["id"] for row in config.get("candidates", [])}
    if set(ids) != configured_candidates:
        raise ComparisonError("candidate ids do not match the comparison config")
    expected_fairness = _expected_fairness(config)
    fairness_keys = (
        "crop_manifest_sha256",
        "seed",
        "input_size",
        "normalization",
        "train_frames_per_video",
        "target_fpr",
        "selected_frames_per_video",
        "aggregation_candidates",
        "training_protocol",
    )
    fairness: dict[str, Any] = {}
    for key in fairness_keys:
        for candidate_id, payload in candidates:
            if key not in payload or payload[key] is None:
                raise ComparisonError(f"{candidate_id}: missing fairness field {key}")
        values = {json.dumps(payload[key], sort_keys=True) for _, payload in candidates}
        if len(values) != 1:
            raise ComparisonError(f"candidate fairness mismatch: {key}")
        fairness[key] = candidates[0][1][key]
        if key in expected_fairness and fairness[key] != expected_fairness[key]:
            raise ComparisonError(f"candidate fairness differs from config: {key}")

    condition_sets = [set(payload["condition_validation"]) for _, payload in candidates]
    if any(value != condition_sets[0] for value in condition_sets[1:]):
        raise ComparisonError("candidate fairness mismatch: validation conditions")
    if condition_sets[0] != set(config["conditions"]):
        raise ComparisonError("validation conditions differ from comparison config")

    rows: list[dict[str, Any]] = []
    ranked: list[tuple[float, float, float, str]] = []
    for candidate_id, payload in candidates:
        non_clean = [
            report["video"]["roc_auc"]
            for condition, report in payload["condition_validation"].items()
            if condition != "clean"
        ]
        robustness_macro_auc = float(sum(non_clean) / len(non_clean)) if non_clean else 0.0
        fpr_at_recall = _metric(
            payload,
            "validation_operating_point_at_recall_0_95",
            "fpr",
        )
        recall_at_operating_point = _metric(
            payload,
            "validation_operating_point_at_recall_0_95",
            "recall",
        )
        latency_p95 = _metric(payload, "validation_video_latency", "p95_ms")
        row = {
            "candidate_id": candidate_id,
            "model": payload.get("model"),
            "validation_fpr_at_recall_0_95": fpr_at_recall,
            "validation_recall_at_operating_point": recall_at_operating_point,
            "eligible_at_target_recall": bool(
                recall_at_operating_point >= float(config["selection_target_recall"])
            ),
            "validation_clean_roc_auc": _metric(payload, "validation_video", "roc_auc"),
            "validation_clean_fpr_at_selected_threshold": _metric(
                payload,
                "validation_video",
                "fpr",
            ),
            "validation_robustness_macro_roc_auc": robustness_macro_auc,
            "validation_latency_p95_ms": latency_p95,
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
        }
        rows.append(row)
        if row["eligible_at_target_recall"]:
            ranked.append((fpr_at_recall, -robustness_macro_auc, latency_p95, candidate_id))

    if not ranked:
        raise ComparisonError("no candidate reaches the configured target recall")
    selected_candidate = min(ranked)[-1]
    report: dict[str, Any] = {
        "status": "completed",
        "evaluation_scope": "validation_only_candidate_selection",
        "selection_split": "validation",
        "experiment_id": config["experiment_id"],
        "official_test_used_for_selection": False,
        "official_test_inference_performed_before_freeze": False,
        "selection_rule": [
            "minimum validation FPR at recall >= 0.95",
            "maximum non-clean validation macro ROC-AUC",
            "minimum validation p95 latency",
            "candidate id lexical order",
        ],
        "fairness": fairness,
        "validation_conditions": sorted(condition_sets[0]),
        "candidates": sorted(rows, key=lambda row: row["candidate_id"]),
        "selected_candidate": selected_candidate,
        "selected_candidate_frozen_before_official_test": True,
        "external_validation_pending": True,
        "operationally_approved": False,
    }
    config_canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    report["comparison_config_sha256"] = hashlib.sha256(config_canonical).hexdigest()
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["selection_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
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
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="candidate id and validation metrics path: id=/path/metrics.json",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = [
        (candidate_id, _load_json(path))
        for candidate_id, path in args.candidate
    ]
    report = build_comparison(candidates, _load_json(args.config))
    write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
