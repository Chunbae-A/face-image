#!/usr/bin/env python3
"""Audit ArcFace robustness under deterministic mobile-capture degradations.

The clean run supplies every registration embedding. Each degraded run only
supplies query embeddings. Evaluation uses the common successful query pool
across all conditions so comparisons remain paired. Raw identifiers and
embeddings stay in the trusted runtime; published outputs contain aggregate
metrics, fingerprints, and reject-reason counts only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from celebdf_faceguard import (
    PairScores,
    VideoEmbedding,
    auc_eer,
    bootstrap_auc_eer,
    build_pair_scores,
    group_eligible_records,
    load_video_embeddings,
    rates_at_threshold,
    split_subjects,
    threshold_at_far,
)


CLEAN_CONDITION = "clean"
DEFAULT_CONDITIONS = (
    CLEAN_CONDITION,
    "jpeg_q30",
    "gaussian_blur_sigma2",
    "low_light_gamma2",
    "downscale_0_25",
    "combined_mobile_stress",
)
DEFAULT_SEEDS = (20260805, 20260806, 20260807, 20260808, 20260809)
DEFAULT_FAR_POINTS = (0.01, 0.001)
DEFAULT_MAX_REFERENCE_COUNT = 5
DEFAULT_REFERENCE_COUNT = 3
DEFAULT_MINIMUM_QUERIES = 3
DEFAULT_MINIMUM_SUBJECTS = 4
EXPECTED_FRAMES_PER_VIDEO = 5
EXPECTED_MINIMUM_VALID_FRAMES = 3
TARGET_FAR = 0.001
MAXIMUM_TAR_LOSS = 0.05
MINIMUM_PROCESSING_SUCCESS_RATE = 0.98
DECISION_EPSILON = 1e-12


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected CONDITION=PATH")
    return name.strip(), Path(raw_path).expanduser()


def mapping_from_specs(
    specs: Iterable[tuple[str, Path]],
    *,
    require_clean: bool = True,
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for name, path in specs:
        if name in mapping:
            raise ValueError(f"duplicate condition mapping: {name}")
        mapping[name] = path
    if require_clean and CLEAN_CONDITION not in mapping:
        raise ValueError("condition mappings must include clean")
    return mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def reject_reason_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        counts = Counter(
            row.get("reason", "unknown") or "unknown" for row in csv.DictReader(handle)
        )
    return dict(sorted(counts.items()))


def sanitized_run_report(path: Path, expected_condition: str) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    condition = str(report.get("input_condition", "clean"))
    if condition != expected_condition:
        raise ValueError(
            f"run report condition mismatch: expected {expected_condition}, got {condition}"
        )
    allowed = (
        "status",
        "selected_video_count",
        "attempted_this_run",
        "successful_video_count_total",
        "rejected_this_run",
        "frames_per_video",
        "minimum_valid_frames",
        "input_condition",
        "elapsed_seconds",
        "manifest_sha256",
        "git_commit",
        "model_license_scope",
        "insightface_version",
        "onnxruntime_version",
        "onnxruntime_available_providers",
        "onnxruntime_selected_providers",
        "device",
        "model_name",
        "model_hashes",
    )
    sanitized = {key: report[key] for key in allowed if key in report}
    required = {
        "status",
        "selected_video_count",
        "successful_video_count_total",
        "frames_per_video",
        "minimum_valid_frames",
        "input_condition",
        "manifest_sha256",
        "model_hashes",
    }
    missing = sorted(required.difference(sanitized))
    if missing:
        raise ValueError(f"run report is missing required fields: {missing}")
    if sanitized["status"] != "completed":
        raise ValueError(f"condition run is not completed: {expected_condition}")
    if int(sanitized["frames_per_video"]) != EXPECTED_FRAMES_PER_VIDEO:
        raise ValueError(
            f"condition run must use {EXPECTED_FRAMES_PER_VIDEO} frames: "
            f"{expected_condition}"
        )
    if int(sanitized["minimum_valid_frames"]) != EXPECTED_MINIMUM_VALID_FRAMES:
        raise ValueError(
            f"condition run must require {EXPECTED_MINIMUM_VALID_FRAMES} valid frames: "
            f"{expected_condition}"
        )
    return sanitized


def _records_by_video(records: Sequence[VideoEmbedding]) -> dict[str, VideoEmbedding]:
    mapping: dict[str, VideoEmbedding] = {}
    for record in records:
        if record.video_id in mapping:
            raise ValueError(f"duplicate video_id in condition run: {record.video_id}")
        mapping[record.video_id] = record
    return mapping


def _ordered_protocol_groups(
    records: Sequence[VideoEmbedding],
    *,
    minimum_videos: int,
    minimum_valid_frames: int,
) -> dict[str, list[VideoEmbedding]]:
    """Group records without reordering the clean-registration/query boundary."""
    grouped: dict[str, list[VideoEmbedding]] = {}
    seen_videos: set[str] = set()
    for record in records:
        if record.video_id in seen_videos:
            raise ValueError(f"duplicate video_id in protocol records: {record.video_id}")
        seen_videos.add(record.video_id)
        if record.valid_frames < minimum_valid_frames:
            raise ValueError(
                f"protocol record has too few valid frames: {record.video_id}"
            )
        grouped.setdefault(record.subject_id, []).append(record)
    if any(len(subject_records) < minimum_videos for subject_records in grouped.values()):
        raise ValueError("protocol subject has too few registration/query videos")
    return grouped


def quality_summary(
    records: Sequence[VideoEmbedding],
    *,
    selected_video_count: int,
) -> dict[str, object]:
    if not records:
        raise ValueError("condition embedding run is empty")
    if selected_video_count <= 0 or len(records) > selected_video_count:
        raise ValueError("selected video count is inconsistent with embeddings")
    return {
        "successful_video_count": len(records),
        "success_rate": len(records) / selected_video_count,
        "all_subject_count": len({record.subject_id for record in records}),
        "valid_frames_mean": float(np.mean([record.valid_frames for record in records])),
        "valid_frames_min": int(min(record.valid_frames for record in records)),
        "mean_detection_score": float(
            np.nanmean([record.mean_detection_score for record in records])
        ),
        "mean_decode_seconds_per_video": float(
            np.mean([record.decode_seconds for record in records])
        ),
        "mean_transform_seconds_per_video": float(
            np.mean([record.transform_seconds for record in records])
        ),
        "mean_inference_seconds_per_video": float(
            np.mean([record.inference_seconds for record in records])
        ),
    }


def build_common_protocol_records(
    condition_records: Mapping[str, Sequence[VideoEmbedding]],
    *,
    seed: int,
    max_reference_count: int = DEFAULT_MAX_REFERENCE_COUNT,
    minimum_queries: int = DEFAULT_MINIMUM_QUERIES,
    minimum_valid_frames: int = 3,
) -> tuple[dict[str, list[VideoEmbedding]], dict[str, object]]:
    clean = condition_records[CLEAN_CONDITION]
    clean_grouped = group_eligible_records(
        clean,
        minimum_videos=max_reference_count + minimum_queries,
        minimum_valid_frames=minimum_valid_frames,
        seed=seed,
    )
    by_condition = {
        condition: _records_by_video(records)
        for condition, records in condition_records.items()
    }
    mixed = {condition: [] for condition in condition_records}
    registration_ids: set[str] = set()
    common_query_ids: set[str] = set()
    eligible_subjects: list[str] = []

    for subject, ordered_clean in clean_grouped.items():
        registrations = ordered_clean[:max_reference_count]
        query_candidates = ordered_clean[max_reference_count:]
        common_queries: list[VideoEmbedding] = []
        for clean_query in query_candidates:
            matches = [
                by_condition[condition].get(clean_query.video_id)
                for condition in condition_records
            ]
            if all(match is not None and match.subject_id == subject for match in matches):
                common_queries.append(clean_query)
        if len(common_queries) < minimum_queries:
            continue

        eligible_subjects.append(subject)
        registration_ids.update(record.video_id for record in registrations)
        common_query_ids.update(record.video_id for record in common_queries)
        for condition in condition_records:
            mixed[condition].extend(registrations)
            if condition == CLEAN_CONDITION:
                mixed[condition].extend(common_queries)
            else:
                mixed[condition].extend(
                    by_condition[condition][record.video_id] for record in common_queries
                )

    if len(eligible_subjects) < DEFAULT_MINIMUM_SUBJECTS:
        raise ValueError("fewer than four subjects have a common evaluation query pool")
    if registration_ids & common_query_ids:
        raise AssertionError("registration and common query videos overlap")
    expected_video_ids = {record.video_id for record in mixed[CLEAN_CONDITION]}
    for condition, records in mixed.items():
        if {record.video_id for record in records} != expected_video_ids:
            raise AssertionError(f"condition query pool mismatch: {condition}")

    return mixed, {
        "seed": seed,
        "eligible_subject_count": len(eligible_subjects),
        "registration_query_video_overlap": 0,
        "common_video_count": len(expected_video_ids),
        "common_query_video_count": len(common_query_ids),
        "eligible_subject_fingerprint": fingerprint(eligible_subjects),
        "registration_video_fingerprint": fingerprint(registration_ids),
        "common_query_video_fingerprint": fingerprint(common_query_ids),
    }


def _pairs_for_split(
    records: Sequence[VideoEmbedding],
    *,
    validation_subjects: Sequence[str],
    test_subjects: Sequence[str],
    reference_count: int,
    max_reference_count: int,
    minimum_valid_frames: int,
) -> tuple[PairScores, PairScores]:
    grouped = _ordered_protocol_groups(
        records,
        minimum_videos=max_reference_count + DEFAULT_MINIMUM_QUERIES,
        minimum_valid_frames=minimum_valid_frames,
    )
    if set(grouped) != set(validation_subjects) | set(test_subjects):
        raise AssertionError("condition eligible subjects differ from the clean split")
    return (
        build_pair_scores(
            grouped,
            validation_subjects,
            reference_count=reference_count,
            max_reference_count=max_reference_count,
        ),
        build_pair_scores(
            grouped,
            test_subjects,
            reference_count=reference_count,
            max_reference_count=max_reference_count,
        ),
    )


def evaluate_seed(
    condition_records: Mapping[str, Sequence[VideoEmbedding]],
    *,
    seed: int,
    far_points: Sequence[float] = DEFAULT_FAR_POINTS,
    reference_count: int = DEFAULT_REFERENCE_COUNT,
    max_reference_count: int = DEFAULT_MAX_REFERENCE_COUNT,
    minimum_valid_frames: int = 3,
    bootstrap_repeats: int = 500,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mixed, leakage = build_common_protocol_records(
        condition_records,
        seed=seed,
        max_reference_count=max_reference_count,
        minimum_valid_frames=minimum_valid_frames,
    )
    clean_grouped = _ordered_protocol_groups(
        mixed[CLEAN_CONDITION],
        minimum_videos=max_reference_count + DEFAULT_MINIMUM_QUERIES,
        minimum_valid_frames=minimum_valid_frames,
    )
    validation_subjects, test_subjects = split_subjects(clean_grouped, seed=seed)
    if set(validation_subjects) & set(test_subjects):
        raise AssertionError("validation and test identities overlap")
    leakage.update(
        {
            "validation_test_identity_overlap": 0,
            "validation_subject_count": len(validation_subjects),
            "test_subject_count": len(test_subjects),
            "validation_subject_fingerprint": fingerprint(validation_subjects),
            "test_subject_fingerprint": fingerprint(test_subjects),
        }
    )

    clean_validation, _ = _pairs_for_split(
        mixed[CLEAN_CONDITION],
        validation_subjects=validation_subjects,
        test_subjects=test_subjects,
        reference_count=reference_count,
        max_reference_count=max_reference_count,
        minimum_valid_frames=minimum_valid_frames,
    )
    clean_thresholds = {
        far: threshold_at_far(clean_validation.labels, clean_validation.scores, far)
        for far in far_points
    }

    rows: list[dict[str, object]] = []
    for condition, records in mixed.items():
        validation_pairs, test_pairs = _pairs_for_split(
            records,
            validation_subjects=validation_subjects,
            test_subjects=test_subjects,
            reference_count=reference_count,
            max_reference_count=max_reference_count,
            minimum_valid_frames=minimum_valid_frames,
        )
        roc_auc, eer = auc_eer(test_pairs.labels, test_pairs.scores)
        intervals = bootstrap_auc_eer(
            test_pairs,
            repeats=bootstrap_repeats,
            seed=seed + sum(condition.encode("utf-8")),
        )
        row: dict[str, object] = {
            "condition": condition,
            "seed": seed,
            "reference_count": reference_count,
            "eligible_subject_count": len(clean_grouped),
            "validation_subject_count": len(validation_subjects),
            "test_subject_count": len(test_subjects),
            "test_positive_pairs": int(test_pairs.labels.sum()),
            "test_negative_pairs": int((test_pairs.labels == 0).sum()),
            "test_roc_auc": roc_auc,
            "test_eer": eer,
            "roc_auc_ci_low": intervals.get("roc_auc_95ci", [None, None])[0],
            "roc_auc_ci_high": intervals.get("roc_auc_95ci", [None, None])[1],
            "eer_ci_low": intervals.get("eer_95ci", [None, None])[0],
            "eer_ci_high": intervals.get("eer_95ci", [None, None])[1],
        }
        for far in far_points:
            key = f"far_{far:g}"
            clean_threshold = clean_thresholds[far]
            condition_threshold = threshold_at_far(
                validation_pairs.labels,
                validation_pairs.scores,
                far,
            )
            for prefix, threshold in (
                ("clean_locked", clean_threshold),
                ("condition_calibrated", condition_threshold),
            ):
                row[f"{key}_{prefix}_threshold"] = threshold
                validation_rates = rates_at_threshold(
                    validation_pairs.labels,
                    validation_pairs.scores,
                    threshold,
                )
                test_rates = rates_at_threshold(
                    test_pairs.labels,
                    test_pairs.scores,
                    threshold,
                )
                for rate, value in validation_rates.items():
                    row[f"{key}_{prefix}_validation_{rate}"] = value
                for rate, value in test_rates.items():
                    row[f"{key}_{prefix}_test_{rate}"] = value
            row[f"{key}_threshold_shift"] = condition_threshold - clean_threshold
        rows.append(row)
    return rows, leakage


SUMMARY_METRICS = (
    "test_roc_auc",
    "test_eer",
    "far_0.001_clean_locked_threshold",
    "far_0.001_clean_locked_test_tar",
    "far_0.001_clean_locked_test_far",
    "far_0.001_clean_locked_test_frr",
    "far_0.001_condition_calibrated_threshold",
    "far_0.001_condition_calibrated_test_tar",
    "far_0.001_condition_calibrated_test_far",
    "far_0.001_condition_calibrated_test_frr",
    "far_0.001_threshold_shift",
)


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    summary: list[dict[str, object]] = []
    for condition in sorted(grouped, key=lambda value: (value != CLEAN_CONDITION, value)):
        group = grouped[condition]
        item: dict[str, object] = {"condition": condition, "seed_count": len(group)}
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
            item[f"{metric}_min"] = float(np.min(values))
            item[f"{metric}_max"] = float(np.max(values))
        summary.append(item)
    return summary


def decision_summary(
    summary: Sequence[Mapping[str, object]],
    quality: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    by_condition = {str(row["condition"]): row for row in summary}
    clean_tar = float(
        by_condition[CLEAN_CONDITION]["far_0.001_clean_locked_test_tar_mean"]
    )
    quality_gate_conditions: list[str] = []
    condition_findings: list[dict[str, object]] = []
    for condition, row in by_condition.items():
        tar = float(row["far_0.001_clean_locked_test_tar_mean"])
        success_rate = float(quality[condition]["success_rate"])
        tar_loss = clean_tar - tar
        quality_gate = (
            tar_loss > MAXIMUM_TAR_LOSS + DECISION_EPSILON
            or success_rate
            < MINIMUM_PROCESSING_SUCCESS_RATE - DECISION_EPSILON
        )
        if quality_gate:
            quality_gate_conditions.append(condition)
        condition_findings.append(
            {
                "condition": condition,
                "clean_locked_tar_loss_vs_clean": tar_loss,
                "success_rate": success_rate,
                "quality_gate_required": quality_gate,
            }
        )
    worst_clean_locked_far = max(
        float(row["far_0.001_clean_locked_test_far_mean"]) for row in summary
    )
    worst_calibrated_far = max(
        float(row["far_0.001_condition_calibrated_test_far_mean"])
        for row in summary
    )
    return {
        "target_far": TARGET_FAR,
        "maximum_tar_loss": MAXIMUM_TAR_LOSS,
        "minimum_processing_success_rate": MINIMUM_PROCESSING_SUCCESS_RATE,
        "worst_clean_locked_test_far_mean": worst_clean_locked_far,
        "worst_condition_calibrated_test_far_mean": worst_calibrated_far,
        "single_global_threshold_approved": (
            worst_clean_locked_far <= TARGET_FAR + DECISION_EPSILON
        ),
        "condition_calibration_approved": (
            worst_calibrated_far <= TARGET_FAR + DECISION_EPSILON
        ),
        "quality_gate_conditions": sorted(quality_gate_conditions),
        "condition_findings": sorted(
            condition_findings,
            key=lambda item: str(item["condition"]),
        ),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    known: set[str] = set()
    for row in rows:
        for key in row:
            if key not in known:
                known.add(key)
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_audit(
    *,
    embeddings: Mapping[str, Path],
    run_reports: Mapping[str, Path],
    rejects: Mapping[str, Path | None],
    output_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    far_points: Sequence[float] = DEFAULT_FAR_POINTS,
    reference_count: int = DEFAULT_REFERENCE_COUNT,
    bootstrap_repeats: int = 500,
) -> dict[str, object]:
    if set(embeddings) != set(run_reports):
        raise ValueError("embedding and run-report condition mappings must match")
    if CLEAN_CONDITION not in embeddings:
        raise ValueError("condition mappings must include clean")
    if not set(rejects).issubset(embeddings):
        raise ValueError("reject mappings contain an unknown condition")
    if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not 1 <= reference_count <= DEFAULT_MAX_REFERENCE_COUNT:
        raise ValueError("reference_count must be between 1 and 5")
    if bootstrap_repeats <= 0:
        raise ValueError("bootstrap_repeats must be positive")
    conditions = tuple(
        sorted(embeddings, key=lambda value: (value != CLEAN_CONDITION, value))
    )
    condition_records = {
        condition: load_video_embeddings(embeddings[condition])
        for condition in conditions
    }
    sanitized_runs: list[dict[str, object]] = []
    quality: dict[str, dict[str, object]] = {}
    for condition in conditions:
        run = sanitized_run_report(run_reports[condition], condition)
        selected = int(run["selected_video_count"])
        quality[condition] = quality_summary(
            condition_records[condition],
            selected_video_count=selected,
        )
        if int(run["successful_video_count_total"]) != len(
            condition_records[condition]
        ):
            raise ValueError(
                f"run report success count differs from embeddings: {condition}"
            )
        sanitized_runs.append(
            {
                "condition": condition,
                "embedding_sha256": sha256_file(embeddings[condition]),
                "run": run,
                "quality": quality[condition],
                "reject_reason_counts": reject_reason_counts(rejects.get(condition)),
            }
        )

    manifest_hashes = {str(item["run"]["manifest_sha256"]) for item in sanitized_runs}
    model_hashes = {
        json.dumps(item["run"]["model_hashes"], sort_keys=True)
        for item in sanitized_runs
    }
    if len(manifest_hashes) != 1:
        raise ValueError("condition runs used different dataset manifests")
    if len(model_hashes) != 1:
        raise ValueError("condition runs used different ArcFace model files")

    metrics: list[dict[str, object]] = []
    leakage_checks: list[dict[str, object]] = []
    for seed in seeds:
        rows, leakage = evaluate_seed(
            condition_records,
            seed=int(seed),
            far_points=far_points,
            reference_count=reference_count,
            bootstrap_repeats=bootstrap_repeats,
        )
        metrics.extend(rows)
        leakage_checks.append(leakage)
    summary = summarize_rows(metrics)
    decisions = decision_summary(summary, quality)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "celebdf_robustness_metrics.csv"
    summary_path = output_dir / "celebdf_robustness_summary.csv"
    report_path = output_dir / "celebdf_robustness_audit.json"
    write_csv(metrics_path, metrics)
    write_csv(summary_path, summary)
    report: dict[str, object] = {
        "experiment": "celebdf-arcface-robustness-v1",
        "scope": "Celeb-real query degradation robustness; not deepfake detection",
        "conditions": list(conditions),
        "seeds": [int(seed) for seed in seeds],
        "reference_count": reference_count,
        "max_reference_count": DEFAULT_MAX_REFERENCE_COUNT,
        "common_query_pool": True,
        "registration_condition": CLEAN_CONDITION,
        "threshold_selection": {
            "clean_locked": "clean validation identities only",
            "condition_calibrated": "same-condition validation identities only",
            "test_scores_used_for_selection": False,
        },
        "bootstrap_repeats": bootstrap_repeats,
        "input_runs": sanitized_runs,
        "metrics": metrics,
        "summary": summary,
        "leakage_checks": leakage_checks,
        "decisions": decisions,
        "artifacts": {
            "metrics_csv": metrics_path.name,
            "summary_csv": summary_path.name,
        },
    }
    write_json_atomic(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-run",
        action="append",
        type=parse_named_path,
        required=True,
        help="repeat CONDITION=PATH for every condition NPZ",
    )
    parser.add_argument(
        "--run-report",
        action="append",
        type=parse_named_path,
        required=True,
        help="repeat CONDITION=PATH for every condition run JSON",
    )
    parser.add_argument(
        "--rejects",
        action="append",
        type=parse_named_path,
        default=[],
        help="optional CONDITION=PATH reject CSV",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=lambda value: tuple(int(item) for item in value.split(",") if item.strip()),
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--reference-count", type=positive_int, default=DEFAULT_REFERENCE_COUNT)
    parser.add_argument("--bootstrap-repeats", type=positive_int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reject_mapping = (
        mapping_from_specs(args.rejects, require_clean=False)
        if args.rejects
        else {}
    )
    report = run_audit(
        embeddings=mapping_from_specs(args.embedding_run),
        run_reports=mapping_from_specs(args.run_report),
        rejects=reject_mapping,
        output_dir=args.output_dir,
        seeds=args.seeds,
        reference_count=args.reference_count,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "condition_count": len(report["conditions"]),
                "metric_rows": len(report["metrics"]),
                "decisions": report["decisions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
