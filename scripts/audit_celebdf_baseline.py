#!/usr/bin/env python3
"""Audit Celeb-real ArcFace results across frames, references, and split seeds.

The input NPZ files contain biometric embeddings and must remain in the trusted
runtime.  This script writes only aggregate metrics, hashes, reason counts, and
identity-free split fingerprints.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence

import numpy as np

from celebdf_faceguard import (
    VideoEmbedding,
    evaluate_embeddings,
    group_eligible_records,
    load_video_embeddings,
    split_subjects,
)


DEFAULT_SEEDS = (20260805, 20260806, 20260807, 20260808, 20260809)
DEFAULT_REFERENCE_COUNTS = (1, 3, 5)
DEFAULT_MAX_REFERENCE_COUNT = 5
EXPECTED_VIDEO_COUNT = 590


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def parse_frame_path(value: str) -> tuple[int, Path]:
    frame_text, separator, path_text = value.partition("=")
    if not separator or not frame_text.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("expected FRAMES=/path/to/file")
    try:
        frames = int(frame_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("frames must be an integer") from error
    if frames <= 0:
        raise argparse.ArgumentTypeError("frames must be positive")
    return frames, Path(path_text).expanduser()


def mapping_from_specs(specs: Iterable[tuple[int, Path]]) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for frames, path in specs:
        if frames in output:
            raise ValueError(f"duplicate frame mapping: {frames}")
        output[frames] = path
    if not output:
        raise ValueError("at least one frame mapping is required")
    return dict(sorted(output.items()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_reason_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        counts = Counter(row.get("reason", "unknown") or "unknown" for row in csv.DictReader(handle))
    return dict(sorted(counts.items()))


def sanitized_run_report(path: Path, expected_frames: int) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if int(report["frames_per_video"]) != expected_frames:
        raise ValueError(
            f"run report frame mismatch: expected {expected_frames}, got {report['frames_per_video']}"
        )
    allowed = (
        "status",
        "selected_video_count",
        "attempted_this_run",
        "successful_video_count_total",
        "rejected_this_run",
        "frames_per_video",
        "minimum_valid_frames",
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
    return {key: report[key] for key in allowed if key in report}


def quality_summary(
    records: Sequence[VideoEmbedding],
    *,
    requested_frames: int,
    minimum_valid_frames: int,
) -> dict[str, object]:
    if not records:
        raise ValueError("embedding run is empty")
    if any(record.sampled_frames > requested_frames for record in records):
        raise ValueError(f"sampled frame count exceeds requested frames={requested_frames}")
    dimensions = sorted({int(np.asarray(record.embedding).size) for record in records})
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions differ: {dimensions}")
    grouped = group_eligible_records(
        records,
        minimum_videos=8,
        minimum_valid_frames=minimum_valid_frames,
        seed=DEFAULT_SEEDS[0],
    )
    valid_frames = np.asarray([record.valid_frames for record in records], dtype=float)
    detection_scores = np.asarray(
        [record.mean_detection_score for record in records], dtype=float
    )
    decode_seconds = np.asarray([record.decode_seconds for record in records], dtype=float)
    inference_seconds = np.asarray(
        [record.inference_seconds for record in records], dtype=float
    )
    return {
        "successful_video_count": len(records),
        "success_rate": len(records) / EXPECTED_VIDEO_COUNT,
        "all_subject_count": len({record.subject_id for record in records}),
        "eligible_subject_count": len(grouped),
        "embedding_dimension": dimensions[0],
        "valid_frames_mean": float(np.mean(valid_frames)),
        "valid_frames_min": int(np.min(valid_frames)),
        "mean_detection_score": float(np.nanmean(detection_scores)),
        "mean_decode_seconds_per_video": float(np.mean(decode_seconds)),
        "mean_inference_seconds_per_video": float(np.mean(inference_seconds)),
    }


def leakage_summary(
    records: Sequence[VideoEmbedding],
    *,
    seed: int,
    minimum_valid_frames: int,
    max_reference_count: int,
) -> dict[str, object]:
    grouped = group_eligible_records(
        records,
        minimum_videos=max_reference_count + 3,
        minimum_valid_frames=minimum_valid_frames,
        seed=seed,
    )
    validation, test = split_subjects(grouped, seed=seed)
    validation_set = set(validation)
    test_set = set(test)
    if validation_set & test_set:
        raise AssertionError("validation and test identities overlap")

    registration_videos: set[str] = set()
    query_videos: set[str] = set()
    for rows in grouped.values():
        registration_videos.update(row.video_id for row in rows[:max_reference_count])
        query_videos.update(row.video_id for row in rows[max_reference_count:])
    if registration_videos & query_videos:
        raise AssertionError("registration and query videos overlap")

    video_ids = [record.video_id for record in records]
    if len(video_ids) != len(set(video_ids)):
        raise AssertionError("global duplicate video_id detected")

    return {
        "seed": seed,
        "eligible_subject_count": len(grouped),
        "validation_subject_count": len(validation),
        "test_subject_count": len(test),
        "validation_test_identity_overlap": 0,
        "registration_query_video_overlap": 0,
        "global_duplicate_video_ids": 0,
        "validation_subject_fingerprint": fingerprint(validation),
        "test_subject_fingerprint": fingerprint(test),
        "registration_video_fingerprint": fingerprint(registration_videos),
        "query_video_fingerprint": fingerprint(query_videos),
    }


def flatten_protocol_row(
    *,
    frames_per_video: int,
    seed: int,
    reference_count: int,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "frames_per_video": frames_per_video,
        "seed": seed,
        "reference_count": reference_count,
        "test_roc_auc": protocol["test_roc_auc"],
        "test_eer": protocol["test_eer"],
        "roc_auc_ci_low": protocol["roc_auc_95ci"][0],
        "roc_auc_ci_high": protocol["roc_auc_95ci"][1],
        "eer_ci_low": protocol["eer_95ci"][0],
        "eer_ci_high": protocol["eer_95ci"][1],
        "test_positive_pairs": protocol["test_positive_pairs"],
        "test_negative_pairs": protocol["test_negative_pairs"],
    }
    for far_key, point in protocol["operating_points"].items():
        row[f"{far_key}_threshold"] = point["threshold_selected_on_validation"]
        for split in ("validation", "test"):
            for metric in ("tar", "far", "frr"):
                row[f"{far_key}_{split}_{metric}"] = point[split][metric]
    return row


SUMMARY_METRICS = (
    "test_roc_auc",
    "test_eer",
    "far_0.01_threshold",
    "far_0.01_test_tar",
    "far_0.01_test_far",
    "far_0.01_test_frr",
    "far_0.001_threshold",
    "far_0.001_test_tar",
    "far_0.001_test_far",
    "far_0.001_test_frr",
)


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (int(row["frames_per_video"]), int(row["reference_count"]))
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for (frames, references), group in sorted(grouped.items()):
        output: dict[str, object] = {
            "frames_per_video": frames,
            "reference_count": references,
            "seed_count": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group]
            output[f"{metric}_mean"] = statistics.fmean(values)
            output[f"{metric}_std"] = statistics.pstdev(values)
            output[f"{metric}_min"] = min(values)
            output[f"{metric}_max"] = max(values)
        summary.append(output)
    return summary


def lookup_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    frames: int,
    references: int,
    metric: str,
) -> float | None:
    for row in rows:
        if int(row["frames_per_video"]) == frames and int(row["reference_count"]) == references:
            return float(row[metric])
    return None


def decision_summary(summary: Sequence[Mapping[str, object]]) -> dict[str, object]:
    frame5 = lookup_summary(
        summary,
        frames=5,
        references=3,
        metric="far_0.001_test_tar_mean",
    )
    frame10 = lookup_summary(
        summary,
        frames=10,
        references=3,
        metric="far_0.001_test_tar_mean",
    )
    ref3 = frame10
    ref5 = lookup_summary(
        summary,
        frames=10,
        references=5,
        metric="far_0.001_test_tar_mean",
    )
    decisions: dict[str, object] = {}
    if frame5 is not None and frame10 is not None:
        loss = frame10 - frame5
        decisions["frames"] = {
            "frame_5_tar": frame5,
            "frame_10_tar": frame10,
            "frame_5_tar_loss": loss,
            "criterion_max_loss": 0.005,
            "recommendation": "use_5_frames" if loss < 0.005 else "keep_10_frames",
        }
    if ref3 is not None and ref5 is not None:
        gain = ref5 - ref3
        decisions["registration"] = {
            "reference_3_tar": ref3,
            "reference_5_tar": ref5,
            "reference_5_tar_gain": gain,
            "criterion_min_gain": 0.01,
            "recommendation": "use_3_references" if gain < 0.01 else "use_5_references",
        }
    return decisions


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    *,
    embeddings: Mapping[int, Path],
    run_reports: Mapping[int, Path],
    rejects: Mapping[int, Path],
    output_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    reference_counts: Sequence[int] = DEFAULT_REFERENCE_COUNTS,
    bootstrap_repeats: int = 500,
    max_reference_count: int = DEFAULT_MAX_REFERENCE_COUNT,
) -> dict[str, object]:
    if set(embeddings) != set(run_reports):
        raise ValueError("embedding and run-report frame mappings must match")
    if max(reference_counts) > max_reference_count:
        raise ValueError("reference count exceeds reserved registration videos")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    metric_rows: list[dict[str, object]] = []
    input_runs: list[dict[str, object]] = []
    leakage_checks: list[dict[str, object]] = []
    model_hash_sets: list[dict[str, str]] = []

    for frames, embedding_path in sorted(embeddings.items()):
        run = sanitized_run_report(run_reports[frames], frames)
        minimum_valid_frames = int(run.get("minimum_valid_frames", min(3, frames)))
        records = load_video_embeddings(embedding_path)
        quality = quality_summary(
            records,
            requested_frames=frames,
            minimum_valid_frames=minimum_valid_frames,
        )
        model_hashes = dict(run.get("model_hashes", {}))
        model_hash_sets.append(model_hashes)
        input_runs.append(
            {
                "frames_per_video": frames,
                "embedding_sha256": sha256_file(embedding_path),
                "quality": quality,
                "reject_reason_counts": reject_reason_counts(rejects.get(frames)),
                "run": run,
            }
        )

        for seed in seeds:
            leakage = leakage_summary(
                records,
                seed=seed,
                minimum_valid_frames=minimum_valid_frames,
                max_reference_count=max_reference_count,
            )
            leakage_checks.append({"frames_per_video": frames, **leakage})
            evaluation = evaluate_embeddings(
                records,
                seed=seed,
                minimum_videos=max_reference_count + 3,
                minimum_valid_frames=minimum_valid_frames,
                bootstrap_repeats=bootstrap_repeats,
                reference_counts=reference_counts,
                max_reference_count=max_reference_count,
            )
            if set(evaluation["validation_subjects"]) & set(evaluation["test_subjects"]):
                raise AssertionError("evaluation leaked identities across validation and test")
            for reference_count in reference_counts:
                metric_rows.append(
                    flatten_protocol_row(
                        frames_per_video=frames,
                        seed=seed,
                        reference_count=reference_count,
                        protocol=evaluation["protocols"][f"reference_{reference_count}"],
                    )
                )

    if any(model_hashes != model_hash_sets[0] for model_hashes in model_hash_sets[1:]):
        raise ValueError("model hashes differ across frame-count runs")

    summary_rows = summarize_rows(metric_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "celebdf_baseline_audit_metrics.csv"
    summary_csv = output_dir / "celebdf_baseline_audit_summary.csv"
    report_json = output_dir / "celebdf_baseline_audit.json"
    write_csv(metrics_csv, metric_rows)
    write_csv(summary_csv, summary_rows)

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "scope": "Celeb-real identity verification baseline; not deepfake detection",
        "seeds": list(seeds),
        "reference_counts": list(reference_counts),
        "max_reference_count": max_reference_count,
        "query_pool_note": "all reference protocols use videos after the first max_reference_count",
        "bootstrap_repeats": bootstrap_repeats,
        "input_runs": input_runs,
        "leakage_checks": leakage_checks,
        "metrics": metric_rows,
        "summary": summary_rows,
        "decisions": decision_summary(summary_rows),
        "artifacts": {
            "metrics_csv": metrics_csv.name,
            "summary_csv": summary_csv.name,
        },
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-run",
        action="append",
        type=parse_frame_path,
        required=True,
        metavar="FRAMES=NPZ",
    )
    parser.add_argument(
        "--run-report",
        action="append",
        type=parse_frame_path,
        required=True,
        metavar="FRAMES=JSON",
    )
    parser.add_argument(
        "--rejects",
        action="append",
        type=parse_frame_path,
        default=[],
        metavar="FRAMES=CSV",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--reference-counts",
        type=parse_int_list,
        default=DEFAULT_REFERENCE_COUNTS,
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--max-reference-count", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(
        embeddings=mapping_from_specs(args.embedding_run),
        run_reports=mapping_from_specs(args.run_report),
        rejects=mapping_from_specs(args.rejects) if args.rejects else {},
        output_dir=args.output_dir,
        seeds=args.seeds,
        reference_counts=args.reference_counts,
        bootstrap_repeats=args.bootstrap_repeats,
        max_reference_count=args.max_reference_count,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "input_run_count": len(report["input_runs"]),
                "metric_row_count": len(report["metrics"]),
                "output_dir": str(args.output_dir),
                "decisions": report["decisions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
