#!/usr/bin/env python3
"""Celeb-DF-v2 FaceGuard inventory, extraction, and verification evaluation.

The identity-verification protocol deliberately uses only ``Celeb-real``.
Each video is one independent sample: frame embeddings are aggregated to a
single video embedding before registration and evaluation.  The first five
deterministically ordered videos are reserved for registration, while queries
start after video five for both the 3-reference and 5-reference protocols.
This keeps every query video disjoint from every registration video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np


CELEB_REAL_RE = re.compile(
    r"^(?:.*/)?Celeb-real/id(?P<subject>\d+)_(?P<video>\d+)\.mp4$",
    re.IGNORECASE,
)
DEFAULT_SEED = 20260805
DEFAULT_MIN_VIDEOS = 8
DEFAULT_MAX_REFERENCE_COUNT = 5


@dataclass(frozen=True)
class ArchiveVideo:
    archive_member: str
    relative_path: str
    subject_id: str
    video_id: str
    uncompressed_bytes: int
    crc32: int


@dataclass(frozen=True)
class VideoEmbedding:
    subject_id: str
    video_id: str
    relative_path: str
    embedding: np.ndarray
    sampled_frames: int
    valid_frames: int
    mean_detection_score: float
    mean_face_area_ratio: float
    decode_seconds: float
    inference_seconds: float
    transform_seconds: float = 0.0


@dataclass(frozen=True)
class PairScores:
    labels: np.ndarray
    scores: np.ndarray
    query_subjects: np.ndarray


def _normalized_member_path(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def parse_celeb_real_member(name: str, *, size: int = 0, crc32: int = 0) -> ArchiveVideo | None:
    normalized = _normalized_member_path(name)
    match = CELEB_REAL_RE.fullmatch(normalized)
    if match is None:
        return None
    subject_number = int(match.group("subject"))
    video_number = int(match.group("video"))
    filename = f"id{subject_number}_{video_number:04d}.mp4"
    return ArchiveVideo(
        archive_member=name,
        relative_path=f"Celeb-real/{filename}",
        subject_id=f"id{subject_number}",
        video_id=filename.removesuffix(".mp4"),
        uncompressed_bytes=int(size),
        crc32=int(crc32),
    )


def inventory_zip(zip_path: Path) -> list[ArchiveVideo]:
    rows: list[ArchiveVideo] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            row = parse_celeb_real_member(
                info.filename,
                size=info.file_size,
                crc32=info.CRC,
            )
            if row is not None:
                if info.flag_bits & 0x1:
                    raise ValueError(f"encrypted ZIP member is unsupported: {info.filename}")
                rows.append(row)
    rows.sort(key=lambda item: (_subject_number(item.subject_id), item.video_id))
    if not rows:
        raise ValueError("no Celeb-real/idN_NNNN.mp4 files were found in the ZIP")
    members = [row.archive_member for row in rows]
    if len(members) != len(set(members)):
        raise ValueError("duplicate Celeb-real member names were found in the ZIP")
    return rows


def _subject_number(subject_id: str) -> int:
    match = re.fullmatch(r"id(\d+)", subject_id)
    if match is None:
        raise ValueError(f"invalid subject_id: {subject_id}")
    return int(match.group(1))


def inventory_summary(rows: Sequence[ArchiveVideo]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.subject_id] = counts.get(row.subject_id, 0) + 1
    ordered_counts = dict(
        sorted(counts.items(), key=lambda item: _subject_number(item[0]))
    )
    eligible = [
        subject for subject, count in ordered_counts.items() if count >= DEFAULT_MIN_VIDEOS
    ]
    return {
        "dataset": "Celeb-DF-v2/Celeb-real",
        "video_count": len(rows),
        "subject_count": len(counts),
        "uncompressed_bytes": sum(row.uncompressed_bytes for row in rows),
        "minimum_videos_per_subject": min(counts.values()),
        "maximum_videos_per_subject": max(counts.values()),
        "eligible_subjects_ge_8_videos": len(eligible),
        "excluded_subjects_lt_8_videos": sorted(
            (subject for subject, count in counts.items() if count < DEFAULT_MIN_VIDEOS),
            key=_subject_number,
        ),
        "videos_per_subject": ordered_counts,
    }


def write_manifest(rows: Sequence[ArchiveVideo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_manifest(path: Path) -> list[ArchiveVideo]:
    rows: list[ArchiveVideo] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                ArchiveVideo(
                    archive_member=raw["archive_member"],
                    relative_path=raw["relative_path"],
                    subject_id=raw["subject_id"],
                    video_id=raw["video_id"],
                    uncompressed_bytes=int(raw["uncompressed_bytes"]),
                    crc32=int(raw["crc32"]),
                )
            )
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def select_smoke_rows(
    rows: Sequence[ArchiveVideo],
    *,
    subjects: int = 2,
    videos_per_subject: int = 1,
) -> list[ArchiveVideo]:
    if subjects <= 0 or videos_per_subject <= 0:
        raise ValueError("smoke selection sizes must be positive")
    grouped: dict[str, list[ArchiveVideo]] = {}
    for row in rows:
        grouped.setdefault(row.subject_id, []).append(row)
    chosen: list[ArchiveVideo] = []
    for subject in sorted(grouped, key=_subject_number)[:subjects]:
        chosen.extend(sorted(grouped[subject], key=lambda item: item.video_id)[:videos_per_subject])
    return chosen


def _safe_target(output_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative path: {relative_path}")
    root = output_root.resolve()
    target = (root / Path(*relative.parts)).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path escapes output root: {relative_path}")
    return target


def extract_rows(
    zip_path: Path,
    rows: Sequence[ArchiveVideo],
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    written_bytes = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        for row in rows:
            if row.archive_member not in members:
                raise KeyError(f"ZIP member is missing: {row.archive_member}")
            target = _safe_target(output_root, row.relative_path)
            if (
                target.exists()
                and not overwrite
                and target.stat().st_size == row.uncompressed_bytes
            ):
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(row.archive_member) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            if temporary.stat().st_size != row.uncompressed_bytes:
                temporary.unlink(missing_ok=True)
                raise IOError(f"extracted size mismatch: {row.archive_member}")
            os.replace(temporary, target)
            extracted += 1
            written_bytes += row.uncompressed_bytes
    return {
        "selected": len(rows),
        "extracted": extracted,
        "skipped": skipped,
        "written_bytes": written_bytes,
    }


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding norm must be finite and positive")
    return value / norm


def save_video_embeddings(records: Sequence[VideoEmbedding], path: Path) -> None:
    if not records:
        raise ValueError("cannot save an empty embedding collection")
    dimensions = {np.asarray(record.embedding).shape for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions are inconsistent: {dimensions}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            subject_ids=np.asarray([record.subject_id for record in records]),
            video_ids=np.asarray([record.video_id for record in records]),
            relative_paths=np.asarray([record.relative_path for record in records]),
            embeddings=np.stack([l2_normalize(record.embedding) for record in records]),
            sampled_frames=np.asarray([record.sampled_frames for record in records], dtype=np.int32),
            valid_frames=np.asarray([record.valid_frames for record in records], dtype=np.int32),
            mean_detection_scores=np.asarray(
                [record.mean_detection_score for record in records], dtype=np.float32
            ),
            mean_face_area_ratios=np.asarray(
                [record.mean_face_area_ratio for record in records], dtype=np.float32
            ),
            decode_seconds=np.asarray(
                [record.decode_seconds for record in records], dtype=np.float32
            ),
            inference_seconds=np.asarray(
                [record.inference_seconds for record in records], dtype=np.float32
            ),
            transform_seconds=np.asarray(
                [record.transform_seconds for record in records], dtype=np.float32
            ),
        )
    os.replace(temporary, path)


def load_video_embeddings(path: Path) -> list[VideoEmbedding]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "subject_ids",
            "video_ids",
            "relative_paths",
            "embeddings",
            "sampled_frames",
            "valid_frames",
            "mean_detection_scores",
            "mean_face_area_ratios",
            "decode_seconds",
            "inference_seconds",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"embedding file is missing arrays: {sorted(missing)}")
        count = len(payload["subject_ids"])
        if any(len(payload[key]) != count for key in required):
            raise ValueError("embedding arrays do not have the same row count")
        transform_seconds = (
            payload["transform_seconds"]
            if "transform_seconds" in payload.files
            else np.zeros(count, dtype=np.float32)
        )
        if len(transform_seconds) != count:
            raise ValueError("embedding transform_seconds does not match row count")
        return [
            VideoEmbedding(
                subject_id=str(payload["subject_ids"][index]),
                video_id=str(payload["video_ids"][index]),
                relative_path=str(payload["relative_paths"][index]),
                embedding=l2_normalize(payload["embeddings"][index]),
                sampled_frames=int(payload["sampled_frames"][index]),
                valid_frames=int(payload["valid_frames"][index]),
                mean_detection_score=float(payload["mean_detection_scores"][index]),
                mean_face_area_ratio=float(payload["mean_face_area_ratios"][index]),
                decode_seconds=float(payload["decode_seconds"][index]),
                inference_seconds=float(payload["inference_seconds"][index]),
                transform_seconds=float(transform_seconds[index]),
            )
            for index in range(count)
        ]


def _stable_order_key(subject_id: str, video_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{subject_id}:{video_id}".encode()).hexdigest()


def group_eligible_records(
    records: Sequence[VideoEmbedding],
    *,
    minimum_videos: int = DEFAULT_MIN_VIDEOS,
    minimum_valid_frames: int = 3,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[VideoEmbedding]]:
    grouped: dict[str, list[VideoEmbedding]] = {}
    seen_videos: set[str] = set()
    for record in records:
        if record.video_id in seen_videos:
            raise ValueError(f"duplicate video_id in embeddings: {record.video_id}")
        seen_videos.add(record.video_id)
        if record.valid_frames < minimum_valid_frames:
            continue
        l2_normalize(record.embedding)
        grouped.setdefault(record.subject_id, []).append(record)
    eligible = {
        subject: sorted(
            values,
            key=lambda item: _stable_order_key(subject, item.video_id, seed),
        )
        for subject, values in grouped.items()
        if len(values) >= minimum_videos
    }
    return dict(sorted(eligible.items(), key=lambda item: _subject_number(item[0])))


def split_subjects(
    subjects: Iterable[str],
    *,
    validation_fraction: float = 0.30,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str]]:
    ordered = sorted(set(subjects), key=_subject_number)
    if len(ordered) < 4:
        raise ValueError("at least four eligible subjects are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ordered, dtype=str)
    rng.shuffle(shuffled)
    validation_count = min(
        len(ordered) - 2,
        max(2, int(round(len(ordered) * validation_fraction))),
    )
    validation = sorted(shuffled[:validation_count].tolist(), key=_subject_number)
    test = sorted(shuffled[validation_count:].tolist(), key=_subject_number)
    return validation, test


def build_pair_scores(
    grouped: dict[str, list[VideoEmbedding]],
    subjects: Sequence[str],
    *,
    reference_count: int,
    max_reference_count: int = DEFAULT_MAX_REFERENCE_COUNT,
) -> PairScores:
    if not 1 <= reference_count <= max_reference_count:
        raise ValueError("reference_count must be between 1 and max_reference_count")
    selected = [subject for subject in subjects if subject in grouped]
    if len(selected) < 2:
        raise ValueError("at least two subjects are required for negative pairs")
    templates: dict[str, np.ndarray] = {}
    queries: dict[str, list[VideoEmbedding]] = {}
    for subject in selected:
        rows = grouped[subject]
        if len(rows) <= max_reference_count:
            raise ValueError(f"subject has no query video after registration: {subject}")
        templates[subject] = l2_normalize(
            np.mean(
                np.stack([row.embedding for row in rows[:reference_count]]),
                axis=0,
            )
        )
        queries[subject] = rows[max_reference_count:]

    labels: list[int] = []
    scores: list[float] = []
    query_subjects: list[str] = []
    for query_subject in selected:
        for query in queries[query_subject]:
            embedding = l2_normalize(query.embedding)
            for template_subject in selected:
                labels.append(int(query_subject == template_subject))
                scores.append(float(embedding @ templates[template_subject]))
                query_subjects.append(query_subject)
    return PairScores(
        labels=np.asarray(labels, dtype=np.int8),
        scores=np.asarray(scores, dtype=np.float64),
        query_subjects=np.asarray(query_subjects),
    )


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be same-length one-dimensional arrays")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("both positive and negative scores are required")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.r_[np.where(np.diff(sorted_scores))[0], len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_labels)[distinct]
    false_positives = (1 + distinct) - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    thresholds = np.r_[np.inf, sorted_scores[distinct]]
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def auc_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:  # NumPy < 2.0
        auc = float(np.trapz(tpr, fpr))
    false_negative_rate = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - false_negative_rate)))
    eer = float((fpr[index] + false_negative_rate[index]) / 2.0)
    return auc, eer


def threshold_at_far(labels: np.ndarray, scores: np.ndarray, target_far: float) -> float:
    if not 0 <= target_far < 1:
        raise ValueError("target_far must be in [0, 1)")
    negative_scores = np.sort(np.asarray(scores)[np.asarray(labels) == 0])[::-1]
    if len(negative_scores) == 0:
        raise ValueError("negative scores are required")
    allowed_false_accepts = int(math.floor(target_far * len(negative_scores)))
    if allowed_false_accepts == 0:
        return float(np.nextafter(negative_scores[0], np.inf))
    if allowed_false_accepts >= len(negative_scores):
        return float(-np.inf)
    return float(np.nextafter(negative_scores[allowed_false_accepts], np.inf))


def rates_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    return {
        "tar": float(np.mean(positives >= threshold)),
        "far": float(np.mean(negatives >= threshold)),
        "frr": float(np.mean(positives < threshold)),
    }


def bootstrap_auc_eer(
    pairs: PairScores,
    *,
    repeats: int = 500,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[float]]:
    if repeats <= 0:
        return {}
    subjects = np.unique(pairs.query_subjects)
    if len(subjects) < 2:
        raise ValueError("at least two query subjects are required for bootstrap")
    by_subject = {
        subject: np.where(pairs.query_subjects == subject)[0] for subject in subjects
    }
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    eer_values: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([by_subject[subject] for subject in sampled])
        auc, eer = auc_eer(pairs.labels[indices], pairs.scores[indices])
        auc_values.append(auc)
        eer_values.append(eer)
    return {
        "roc_auc_95ci": [
            float(np.quantile(auc_values, 0.025)),
            float(np.quantile(auc_values, 0.975)),
        ],
        "eer_95ci": [
            float(np.quantile(eer_values, 0.025)),
            float(np.quantile(eer_values, 0.975)),
        ],
    }


def evaluate_embeddings(
    records: Sequence[VideoEmbedding],
    *,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = 0.30,
    minimum_videos: int = DEFAULT_MIN_VIDEOS,
    minimum_valid_frames: int = 3,
    far_points: Sequence[float] = (0.01, 0.001),
    bootstrap_repeats: int = 500,
    reference_counts: Sequence[int] = (3, 5),
    max_reference_count: int = DEFAULT_MAX_REFERENCE_COUNT,
) -> dict[str, object]:
    reference_counts = tuple(sorted(set(int(value) for value in reference_counts)))
    if not reference_counts:
        raise ValueError("reference_counts cannot be empty")
    if reference_counts[0] < 1 or reference_counts[-1] > max_reference_count:
        raise ValueError("reference_counts must be between 1 and max_reference_count")
    if minimum_videos <= max_reference_count:
        raise ValueError("minimum_videos must leave at least one post-registration query")
    grouped = group_eligible_records(
        records,
        minimum_videos=minimum_videos,
        minimum_valid_frames=minimum_valid_frames,
        seed=seed,
    )
    validation_subjects, test_subjects = split_subjects(
        grouped,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    protocols: dict[str, object] = {}
    for reference_count in reference_counts:
        validation_pairs = build_pair_scores(
            grouped,
            validation_subjects,
            reference_count=reference_count,
            max_reference_count=max_reference_count,
        )
        test_pairs = build_pair_scores(
            grouped,
            test_subjects,
            reference_count=reference_count,
            max_reference_count=max_reference_count,
        )
        roc_auc, eer = auc_eer(test_pairs.labels, test_pairs.scores)
        operating_points: dict[str, object] = {}
        for far in far_points:
            threshold = threshold_at_far(
                validation_pairs.labels,
                validation_pairs.scores,
                far,
            )
            operating_points[f"far_{far:g}"] = {
                "threshold_selected_on_validation": threshold,
                "validation": rates_at_threshold(
                    validation_pairs.labels,
                    validation_pairs.scores,
                    threshold,
                ),
                "test": rates_at_threshold(
                    test_pairs.labels,
                    test_pairs.scores,
                    threshold,
                ),
            }
        protocols[f"reference_{reference_count}"] = {
            "test_roc_auc": roc_auc,
            "test_eer": eer,
            "test_positive_pairs": int(test_pairs.labels.sum()),
            "test_negative_pairs": int((test_pairs.labels == 0).sum()),
            "validation_positive_pairs": int(validation_pairs.labels.sum()),
            "validation_negative_pairs": int((validation_pairs.labels == 0).sum()),
            "operating_points": operating_points,
            **bootstrap_auc_eer(
                test_pairs,
                repeats=bootstrap_repeats,
                seed=seed + reference_count,
            ),
        }
    all_subjects = {record.subject_id for record in records}
    return {
        "status": "measured_from_video_embeddings",
        "model_scope": "pretrained ArcFace baseline; no fine-tuning",
        "threshold_note": "thresholds selected on identity-disjoint validation subjects",
        "reference_counts": list(reference_counts),
        "max_reference_count": max_reference_count,
        "query_start_index": max_reference_count,
        "seed": seed,
        "video_embedding_count": len(records),
        "all_subject_count": len(all_subjects),
        "eligible_subject_count": len(grouped),
        "excluded_subject_count": len(all_subjects) - len(grouped),
        "validation_subject_count": len(validation_subjects),
        "test_subject_count": len(test_subjects),
        "validation_subjects": validation_subjects,
        "test_subjects": test_subjects,
        "protocols": protocols,
    }


def _inventory_command(args: argparse.Namespace) -> dict[str, object]:
    rows = inventory_zip(args.zip)
    write_manifest(rows, args.manifest)
    summary = inventory_summary(rows)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {"manifest": str(args.manifest), **summary}


def _extract_command(args: argparse.Namespace) -> dict[str, object]:
    rows = read_manifest(args.manifest)
    selected = rows
    if args.mode == "smoke":
        selected = select_smoke_rows(
            rows,
            subjects=args.smoke_subjects,
            videos_per_subject=args.smoke_videos_per_subject,
        )
    return {
        "mode": args.mode,
        "output": str(args.output),
        **extract_rows(
            args.zip,
            selected,
            args.output,
            overwrite=args.overwrite,
        ),
    }


def _evaluate_command(args: argparse.Namespace) -> dict[str, object]:
    report = evaluate_embeddings(
        load_video_embeddings(args.embeddings),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        minimum_videos=args.minimum_videos,
        minimum_valid_frames=args.minimum_valid_frames,
        bootstrap_repeats=args.bootstrap_repeats,
        reference_counts=args.reference_counts,
        max_reference_count=args.max_reference_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"output": str(args.output), **report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="build a Celeb-real ZIP manifest")
    inventory.add_argument("zip", type=Path)
    inventory.add_argument("--manifest", type=Path, required=True)
    inventory.add_argument("--summary", type=Path)
    inventory.set_defaults(handler=_inventory_command)

    extract = subparsers.add_parser("extract", help="safely extract selected Celeb-real videos")
    extract.add_argument("zip", type=Path)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--mode", choices=("smoke", "full"), default="full")
    extract.add_argument("--smoke-subjects", type=int, default=2)
    extract.add_argument("--smoke-videos-per-subject", type=int, default=1)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(handler=_extract_command)

    evaluate = subparsers.add_parser("evaluate", help="evaluate video-level ArcFace embeddings")
    evaluate.add_argument("--embeddings", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    evaluate.add_argument("--validation-fraction", type=float, default=0.30)
    evaluate.add_argument("--minimum-videos", type=int, default=DEFAULT_MIN_VIDEOS)
    evaluate.add_argument("--minimum-valid-frames", type=int, default=3)
    evaluate.add_argument("--bootstrap-repeats", type=int, default=500)
    evaluate.add_argument(
        "--reference-counts",
        type=lambda value: tuple(int(item) for item in value.split(",") if item.strip()),
        default=(3, 5),
        help="comma-separated registration video counts; queries always start after max-reference-count",
    )
    evaluate.add_argument(
        "--max-reference-count",
        type=int,
        default=DEFAULT_MAX_REFERENCE_COUNT,
        help="number of ordered videos reserved before the common query pool",
    )
    evaluate.set_defaults(handler=_evaluate_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = args.handler(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
