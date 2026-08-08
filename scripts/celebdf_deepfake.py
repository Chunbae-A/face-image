#!/usr/bin/env python3
"""Celeb-DF-v2 inventory, leakage-safe split, and deepfake metrics.

The official Celeb-DF test list uses ``1`` for real and ``0`` for fake.  This
module deliberately converts it to the service convention ``0=real, 1=fake``
and validates the path-derived class so an accidentally inverted experiment
fails before training starts.

Manifests produced here are private runtime artifacts because they contain
dataset filenames and identity-like identifiers.  Only aggregate summaries
and metrics are suitable for committing to Git.
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
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np


REAL_LABEL = 0
FAKE_LABEL = 1
DEFAULT_SEED = 20260807
EXPECTED_DATASET_COUNTS = {
    "Celeb-real": 590,
    "YouTube-real": 300,
    "Celeb-synthesis": 5639,
}
EXPECTED_OFFICIAL_TEST_COUNT = 518

CELEB_REAL_RE = re.compile(
    r"^(?:.*/)?Celeb-real/(?P<target>id\d+)_(?P<clip>\d+)\.mp4$",
    re.IGNORECASE,
)
YOUTUBE_REAL_RE = re.compile(
    r"^(?:.*/)?YouTube-real/(?P<clip>\d+)\.mp4$",
    re.IGNORECASE,
)
CELEB_FAKE_RE = re.compile(
    r"^(?:.*/)?Celeb-synthesis/(?P<target>id\d+)_(?P<donor>id\d+)_(?P<clip>\d+)\.mp4$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetVideo:
    archive_member: str
    relative_path: str
    video_id: str
    dataset: str
    label: int
    official_test: bool
    split: str
    group_id: str
    target_identity: str
    donor_identity: str
    uncompressed_bytes: int
    crc32: int


@dataclass(frozen=True)
class ScoreRecord:
    split: str
    video_id: str
    label: int
    frame_index: int
    score: float
    latency_ms: float = 0.0
    condition: str = "clean"


def _normalized_member_path(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def parse_video_member(
    name: str,
    *,
    size: int = 0,
    crc32: int = 0,
) -> DatasetVideo | None:
    """Parse one supported video path using the internal fake-positive labels."""
    normalized = _normalized_member_path(name)
    match = CELEB_REAL_RE.fullmatch(normalized)
    if match is not None:
        filename = normalized.rsplit("/", 1)[-1]
        target = match.group("target").lower()
        return DatasetVideo(
            archive_member=name,
            relative_path=f"Celeb-real/{filename}",
            video_id=f"Celeb-real/{filename.removesuffix('.mp4')}",
            dataset="Celeb-real",
            label=REAL_LABEL,
            official_test=False,
            split="unassigned",
            group_id=f"celeb:{target}",
            target_identity=target,
            donor_identity="",
            uncompressed_bytes=int(size),
            crc32=int(crc32),
        )

    match = YOUTUBE_REAL_RE.fullmatch(normalized)
    if match is not None:
        filename = normalized.rsplit("/", 1)[-1]
        clip = match.group("clip")
        return DatasetVideo(
            archive_member=name,
            relative_path=f"YouTube-real/{filename}",
            video_id=f"YouTube-real/{filename.removesuffix('.mp4')}",
            dataset="YouTube-real",
            label=REAL_LABEL,
            official_test=False,
            split="unassigned",
            # Celeb-DF does not publish subject IDs for this directory.  Keeping
            # each source video together is the strongest available grouping.
            group_id=f"youtube:{clip}",
            target_identity="",
            donor_identity="",
            uncompressed_bytes=int(size),
            crc32=int(crc32),
        )

    match = CELEB_FAKE_RE.fullmatch(normalized)
    if match is not None:
        filename = normalized.rsplit("/", 1)[-1]
        target = match.group("target").lower()
        donor = match.group("donor").lower()
        return DatasetVideo(
            archive_member=name,
            relative_path=f"Celeb-synthesis/{filename}",
            video_id=f"Celeb-synthesis/{filename.removesuffix('.mp4')}",
            dataset="Celeb-synthesis",
            label=FAKE_LABEL,
            official_test=False,
            split="unassigned",
            # Naming is targetID-donorID-targetVideoIndex.  Grouping on the
            # first ID keeps an original target person/video context in one
            # internal split; donor IDs are measured separately below.
            group_id=f"celeb:{target}",
            target_identity=target,
            donor_identity=donor,
            uncompressed_bytes=int(size),
            crc32=int(crc32),
        )
    return None


def parse_official_test_list(text: str) -> dict[str, int]:
    """Return ``relative_path -> internal label`` from the official list."""
    result: dict[str, int] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in {"0", "1"}:
            raise ValueError(f"invalid official test line {line_number}: {raw!r}")
        path = _normalized_member_path(parts[1])
        # Official Celeb-DF convention: 1=real, 0=fake.
        internal_label = REAL_LABEL if parts[0] == "1" else FAKE_LABEL
        if path in result:
            raise ValueError(f"duplicate official test path: {path}")
        result[path] = internal_label
    if not result:
        raise ValueError("official test list is empty")
    return result


def inventory_zip(
    zip_path: Path,
    *,
    require_expected_counts: bool = True,
) -> tuple[list[DatasetVideo], str]:
    """Inventory all three Celeb-DF video directories without extracting them."""
    with zipfile.ZipFile(zip_path) as archive:
        list_members = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and _normalized_member_path(info.filename).endswith(
                "List_of_testing_videos.txt"
            )
        ]
        if len(list_members) != 1:
            raise ValueError(
                f"exactly one official test list is required, found {len(list_members)}"
            )
        test_text = archive.read(list_members[0]).decode("utf-8-sig")
        official = parse_official_test_list(test_text)

        rows: list[DatasetVideo] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            row = parse_video_member(
                info.filename,
                size=info.file_size,
                crc32=info.CRC,
            )
            if row is None:
                continue
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member is unsupported: {info.filename}")
            official_label = official.get(row.relative_path)
            if official_label is not None and official_label != row.label:
                raise ValueError(
                    "official label/path mismatch for "
                    f"{row.relative_path}: list={official_label}, path={row.label}"
                )
            rows.append(
                replace(
                    row,
                    official_test=official_label is not None,
                    split="test" if official_label is not None else "unassigned",
                )
            )

    if not rows:
        raise ValueError("no Celeb-DF videos were found in the ZIP")
    relative_paths = [row.relative_path for row in rows]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("duplicate normalized video paths were found")
    missing_test_paths = sorted(set(official).difference(relative_paths))
    if missing_test_paths:
        raise ValueError(
            f"official test paths missing from ZIP: {len(missing_test_paths)}"
        )

    rows.sort(key=lambda row: (row.dataset, row.relative_path))
    if require_expected_counts:
        summary = inventory_summary(rows, official_test_text=test_text)
        if summary["dataset_counts"] != EXPECTED_DATASET_COUNTS:
            raise ValueError(
                f"unexpected dataset counts: {summary['dataset_counts']}"
            )
        if summary["official_test_count"] != EXPECTED_OFFICIAL_TEST_COUNT:
            raise ValueError(
                f"unexpected official test count: {summary['official_test_count']}"
            )
    return rows, test_text


def inventory_directory(
    dataset_root: Path,
    *,
    require_expected_counts: bool = True,
) -> tuple[list[DatasetVideo], str]:
    """Inventory a Kaggle-auto-extracted Celeb-DF directory."""
    dataset_root = dataset_root.expanduser().resolve()
    test_path = dataset_root / "List_of_testing_videos.txt"
    if not test_path.is_file():
        raise FileNotFoundError(f"official test list is missing: {test_path}")
    test_text = test_path.read_text(encoding="utf-8-sig")
    official = parse_official_test_list(test_text)
    rows: list[DatasetVideo] = []
    for directory in EXPECTED_DATASET_COUNTS:
        video_dir = dataset_root / directory
        if not video_dir.is_dir():
            raise FileNotFoundError(f"dataset directory is missing: {video_dir}")
        for path in sorted(video_dir.glob("*.mp4")):
            relative_path = path.relative_to(dataset_root).as_posix()
            row = parse_video_member(relative_path, size=path.stat().st_size, crc32=0)
            if row is None:
                raise ValueError(f"unsupported Celeb-DF video filename: {relative_path}")
            official_label = official.get(row.relative_path)
            if official_label is not None and official_label != row.label:
                raise ValueError(
                    "official label/path mismatch for "
                    f"{row.relative_path}: list={official_label}, path={row.label}"
                )
            rows.append(
                replace(
                    row,
                    archive_member=relative_path,
                    official_test=official_label is not None,
                    split="test" if official_label is not None else "unassigned",
                )
            )
    missing_test_paths = sorted(set(official).difference(row.relative_path for row in rows))
    if missing_test_paths:
        raise ValueError(
            f"official test paths missing from directory: {len(missing_test_paths)}"
        )
    rows.sort(key=lambda row: (row.dataset, row.relative_path))
    if require_expected_counts:
        summary = inventory_summary(rows, official_test_text=test_text)
        if summary["dataset_counts"] != EXPECTED_DATASET_COUNTS:
            raise ValueError(f"unexpected dataset counts: {summary['dataset_counts']}")
        if summary["official_test_count"] != EXPECTED_OFFICIAL_TEST_COUNT:
            raise ValueError(
                f"unexpected official test count: {summary['official_test_count']}"
            )
    return rows, test_text


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _choose_validation_groups(
    groups: Iterable[str],
    *,
    validation_fraction: float,
    seed: int,
) -> set[str]:
    ordered = sorted(set(groups), key=lambda value: _stable_key(value, seed))
    if len(ordered) <= 1:
        return set()
    count = min(len(ordered) - 1, max(1, int(round(len(ordered) * validation_fraction))))
    return set(ordered[:count])


def assign_train_validation_split(
    rows: Sequence[DatasetVideo],
    *,
    validation_fraction: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> list[DatasetVideo]:
    """Assign non-test rows before any frame extraction.

    Celebrity real/fake videos are grouped by the original target identity,
    which also keeps the target video context in one internal split.  Donor
    identities occur across many target pairs, so their overlap is measured
    rather than falsely claimed to be zero.  YouTube real videos have no
    published subject identifier, so each source video is one indivisible
    group.  The official test membership is never changed.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    non_test = [row for row in rows if not row.official_test]
    if not non_test:
        raise ValueError("at least one non-test video is required")

    celeb_groups = [row.group_id for row in non_test if row.group_id.startswith("celeb:")]
    youtube_groups = [
        row.group_id for row in non_test if row.group_id.startswith("youtube:")
    ]
    validation_groups = _choose_validation_groups(
        celeb_groups,
        validation_fraction=validation_fraction,
        seed=seed,
    ) | _choose_validation_groups(
        youtube_groups,
        validation_fraction=validation_fraction,
        seed=seed + 1,
    )

    assigned = [
        row
        if row.official_test
        else replace(
            row,
            split="validation" if row.group_id in validation_groups else "train",
        )
        for row in rows
    ]
    audit = leakage_audit(assigned)
    if audit["train_validation_video_overlap"] != 0:
        raise AssertionError("train/validation video leakage detected")
    if audit["train_validation_group_overlap"] != 0:
        raise AssertionError("train/validation group leakage detected")
    if audit["official_test_outside_test_split"] != 0:
        raise AssertionError("official test video escaped the test split")
    for split in ("train", "validation", "test"):
        labels = {row.label for row in assigned if row.split == split}
        if labels != {REAL_LABEL, FAKE_LABEL}:
            raise ValueError(f"split {split!r} does not contain both labels: {labels}")
    return assigned


def leakage_audit(rows: Sequence[DatasetVideo]) -> dict[str, int]:
    by_split = {
        split: [row for row in rows if row.split == split]
        for split in ("train", "validation", "test")
    }

    def values(split: str, field: str) -> set[str]:
        return {str(getattr(row, field)) for row in by_split[split]}

    return {
        "train_validation_video_overlap": len(
            values("train", "video_id") & values("validation", "video_id")
        ),
        "train_test_video_overlap": len(
            values("train", "video_id") & values("test", "video_id")
        ),
        "validation_test_video_overlap": len(
            values("validation", "video_id") & values("test", "video_id")
        ),
        "train_validation_group_overlap": len(
            values("train", "group_id") & values("validation", "group_id")
        ),
        "train_validation_donor_identity_overlap_observed": len(
            (values("train", "donor_identity") - {""})
            & (values("validation", "donor_identity") - {""})
        ),
        # The published benchmark can contain identities seen outside its test
        # list.  We measure this instead of pretending it is zero.
        "train_test_group_overlap_observed": len(
            values("train", "group_id") & values("test", "group_id")
        ),
        "validation_test_group_overlap_observed": len(
            values("validation", "group_id") & values("test", "group_id")
        ),
        "train_test_donor_identity_overlap_observed": len(
            (values("train", "donor_identity") - {""})
            & (values("test", "donor_identity") - {""})
        ),
        "validation_test_donor_identity_overlap_observed": len(
            (values("validation", "donor_identity") - {""})
            & (values("test", "donor_identity") - {""})
        ),
        "official_test_outside_test_split": sum(
            row.official_test and row.split != "test" for row in rows
        ),
        "nonofficial_video_in_test_split": sum(
            (not row.official_test) and row.split == "test" for row in rows
        ),
    }


def inventory_summary(
    rows: Sequence[DatasetVideo],
    *,
    official_test_text: str = "",
) -> dict[str, object]:
    dataset_counts = {
        dataset: sum(row.dataset == dataset for row in rows)
        for dataset in EXPECTED_DATASET_COUNTS
    }
    split_counts = {
        split: {
            "total": sum(row.split == split for row in rows),
            "real": sum(row.split == split and row.label == REAL_LABEL for row in rows),
            "fake": sum(row.split == split and row.label == FAKE_LABEL for row in rows),
        }
        for split in ("train", "validation", "test", "unassigned")
    }
    payload: dict[str, object] = {
        "dataset": "Celeb-DF-v2",
        "video_count": len(rows),
        "real_video_count": sum(row.label == REAL_LABEL for row in rows),
        "fake_video_count": sum(row.label == FAKE_LABEL for row in rows),
        "dataset_counts": dataset_counts,
        "official_test_count": sum(row.official_test for row in rows),
        "split_counts": split_counts,
        "uncompressed_bytes": sum(row.uncompressed_bytes for row in rows),
        "label_convention": {"real": REAL_LABEL, "fake": FAKE_LABEL},
        "leakage_audit": leakage_audit(rows),
    }
    if official_test_text:
        payload["official_test_list_sha256"] = hashlib.sha256(
            official_test_text.encode("utf-8")
        ).hexdigest()
    return payload


def write_manifest(rows: Sequence[DatasetVideo], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    os.replace(temporary, path)


def read_manifest(path: Path) -> list[DatasetVideo]:
    rows: list[DatasetVideo] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                DatasetVideo(
                    archive_member=raw["archive_member"],
                    relative_path=raw["relative_path"],
                    video_id=raw["video_id"],
                    dataset=raw["dataset"],
                    label=int(raw["label"]),
                    official_test=raw["official_test"].casefold() == "true",
                    split=raw["split"],
                    group_id=raw["group_id"],
                    target_identity=raw["target_identity"],
                    donor_identity=raw.get(
                        "donor_identity",
                        raw.get("source_identity", ""),
                    ),
                    uncompressed_bytes=int(raw["uncompressed_bytes"]),
                    crc32=int(raw["crc32"]),
                )
            )
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def select_smoke_rows(
    rows: Sequence[DatasetVideo],
    *,
    videos_per_class_per_split: int = 1,
    seed: int = DEFAULT_SEED,
) -> list[DatasetVideo]:
    """Select a deterministic real/fake sample from every split."""
    if videos_per_class_per_split <= 0:
        raise ValueError("videos_per_class_per_split must be positive")
    selected: list[DatasetVideo] = []
    for split in ("train", "validation", "test"):
        for label in (REAL_LABEL, FAKE_LABEL):
            candidates = sorted(
                (row for row in rows if row.split == split and row.label == label),
                key=lambda row: _stable_key(row.video_id, seed),
            )
            selected.extend(candidates[:videos_per_class_per_split])
    return sorted(selected, key=lambda row: (row.split, row.label, row.video_id))


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
    rows: Sequence[DatasetVideo],
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


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be same-length one-dimensional arrays")
    if not np.all(np.isin(labels, [REAL_LABEL, FAKE_LABEL])):
        raise ValueError("labels must contain only 0=real and 1=fake")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("both real and fake samples are required")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.r_[np.where(np.diff(sorted_scores))[0], len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_labels)[distinct]
    false_positives = 1 + distinct - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    thresholds = np.r_[np.inf, sorted_scores[distinct]]
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))  # pragma: no cover - NumPy < 2


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must have the same 1-D shape")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("at least one fake sample is required")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def precision_recall_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must have the same 1-D shape")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("at least one fake sample is required")
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    distinct = np.r_[np.where(np.diff(ordered_scores))[0], len(ordered_scores) - 1]
    true_positives = np.cumsum(ordered_labels)[distinct]
    false_positives = 1 + distinct - true_positives
    precision = true_positives / np.maximum(1, true_positives + false_positives)
    recall = true_positives / positives
    return np.r_[1.0, precision].astype(float), np.r_[0.0, recall].astype(float)


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    real_scores = np.sort(scores[labels == REAL_LABEL])[::-1]
    if len(real_scores) == 0:
        raise ValueError("real samples are required to set an FPR threshold")
    allowed_false_positives = int(math.floor(target_fpr * len(real_scores)))
    if allowed_false_positives == 0:
        return float(np.nextafter(real_scores[0], np.inf))
    if allowed_false_positives >= len(real_scores):
        return float(-np.inf)
    return float(np.nextafter(real_scores[allowed_false_positives], np.inf))


def operating_point_at_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
) -> dict[str, float | int]:
    """Return the lowest-FPR operating point that reaches the requested recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    fpr, recall, thresholds = roc_curve(labels, scores)
    eligible = np.flatnonzero(recall >= target_recall)
    if not len(eligible):  # pragma: no cover - a valid binary ROC reaches recall 1
        raise ValueError("target recall cannot be reached")
    # ROC points are ordered from strict to permissive. The first qualifying
    # point therefore has the smallest FPR, with deterministic tie handling.
    index = int(eligible[0])
    metrics = classification_metrics(labels, scores, threshold=float(thresholds[index]))
    return {
        "target_recall": float(target_recall),
        **metrics,
    }


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int8)
    tp = int(np.sum((labels == FAKE_LABEL) & (predictions == FAKE_LABEL)))
    tn = int(np.sum((labels == REAL_LABEL) & (predictions == REAL_LABEL)))
    fp = int(np.sum((labels == REAL_LABEL) & (predictions == FAKE_LABEL)))
    fn = int(np.sum((labels == FAKE_LABEL) & (predictions == REAL_LABEL)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
    fnr_curve = 1.0 - tpr_curve
    eer_index = int(np.argmin(np.abs(fpr_curve - fnr_curve)))
    return {
        "count": len(labels),
        "real_count": int(np.sum(labels == REAL_LABEL)),
        "fake_count": int(np.sum(labels == FAKE_LABEL)),
        "threshold": float(threshold),
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "accuracy": float((tp + tn) / len(labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "eer": float((fpr_curve[eer_index] + fnr_curve[eer_index]) / 2.0),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def aggregate_video_scores(
    records: Sequence[ScoreRecord],
    *,
    method: str,
    top_fraction: float = 0.25,
) -> list[ScoreRecord]:
    if method not in {"mean", "median", "top_k"}:
        raise ValueError(f"unsupported aggregation method: {method}")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    grouped: dict[tuple[str, str, str], list[ScoreRecord]] = {}
    for record in records:
        grouped.setdefault((record.split, record.condition, record.video_id), []).append(record)

    aggregated: list[ScoreRecord] = []
    for (split, condition, video_id), values in sorted(grouped.items()):
        labels = {value.label for value in values}
        if len(labels) != 1:
            raise ValueError(f"video has inconsistent labels: {video_id}")
        scores = np.asarray([value.score for value in values], dtype=np.float64)
        if method == "mean":
            score = float(np.mean(scores))
        elif method == "median":
            score = float(np.median(scores))
        else:
            count = max(1, int(math.ceil(len(scores) * top_fraction)))
            score = float(np.mean(np.sort(scores)[-count:]))
        aggregated.append(
            ScoreRecord(
                split=split,
                video_id=video_id,
                label=next(iter(labels)),
                frame_index=-1,
                score=score,
                latency_ms=float(sum(value.latency_ms for value in values)),
                condition=condition,
            )
        )
    return aggregated


def _records_arrays(records: Sequence[ScoreRecord]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([record.label for record in records], dtype=np.int8),
        np.asarray([record.score for record in records], dtype=np.float64),
    )


def latency_summary(records: Sequence[ScoreRecord]) -> dict[str, float]:
    values = np.asarray([record.latency_ms for record in records], dtype=np.float64)
    if not len(values):
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def evaluate_score_records(
    records: Sequence[ScoreRecord],
    *,
    target_fpr: float = 0.01,
    aggregation_methods: Sequence[str] = ("mean", "median", "top_k"),
) -> dict[str, object]:
    """Select aggregation/threshold on validation and freeze them for test."""
    clean = [record for record in records if record.condition == "clean"]
    validation_frames = [record for record in clean if record.split == "validation"]
    test_frames = [record for record in clean if record.split == "test"]
    if not validation_frames or not test_frames:
        raise ValueError("clean validation and test frame scores are required")

    method_reports: dict[str, dict[str, object]] = {}
    ranked: list[tuple[float, float, float, int, str]] = []
    for method_index, method in enumerate(aggregation_methods):
        validation_video = aggregate_video_scores(validation_frames, method=method)
        labels, scores = _records_arrays(validation_video)
        threshold = threshold_at_fpr(labels, scores, target_fpr)
        metrics = classification_metrics(labels, scores, threshold=threshold)
        method_reports[method] = {"threshold": threshold, "validation": metrics}
        ranked.append(
            (
                float(metrics["roc_auc"]),
                float(metrics["average_precision"]),
                float(metrics["f1"]),
                -method_index,
                method,
            )
        )
    selected_method = max(ranked)[-1]
    threshold = float(method_reports[selected_method]["threshold"])

    selected_video = aggregate_video_scores(clean, method=selected_method)
    validation_video = [row for row in selected_video if row.split == "validation"]
    test_video = [row for row in selected_video if row.split == "test"]
    val_labels, val_scores = _records_arrays(validation_video)
    test_labels, test_scores = _records_arrays(test_video)
    frame_labels, frame_scores = _records_arrays(test_frames)

    condition_reports: dict[str, dict[str, object]] = {}
    for condition in sorted({record.condition for record in records}):
        condition_test = [
            record
            for record in records
            if record.split == "test" and record.condition == condition
        ]
        if not condition_test:
            continue
        videos = aggregate_video_scores(condition_test, method=selected_method)
        labels, scores = _records_arrays(videos)
        condition_reports[condition] = {
            "video": classification_metrics(labels, scores, threshold=threshold),
            "latency": latency_summary(videos),
        }

    test_metrics = classification_metrics(test_labels, test_scores, threshold=threshold)
    test_fpr_curve, test_tpr_curve, _ = roc_curve(test_labels, test_scores)
    test_precision_curve, test_recall_curve = precision_recall_curve(
        test_labels,
        test_scores,
    )
    return {
        "label_convention": {"real": REAL_LABEL, "fake": FAKE_LABEL},
        "selection_split": "validation",
        "official_test_used_for_selection": False,
        "target_fpr": target_fpr,
        "aggregation_candidates": list(aggregation_methods),
        "aggregation_validation": method_reports,
        "selected_aggregation": selected_method,
        "selected_threshold": threshold,
        "validation_video": classification_metrics(
            val_labels, val_scores, threshold=threshold
        ),
        "validation_operating_point_at_recall_0_95": operating_point_at_recall(
            val_labels,
            val_scores,
            0.95,
        ),
        "test_frame": classification_metrics(
            frame_labels, frame_scores, threshold=threshold
        ),
        "test_video": test_metrics,
        "test_video_curves": {
            "roc_fpr": test_fpr_curve.tolist(),
            "roc_tpr": test_tpr_curve.tolist(),
            "pr_recall": test_recall_curve.tolist(),
            "pr_precision": test_precision_curve.tolist(),
        },
        "test_video_latency": latency_summary(test_video),
        "condition_test": condition_reports,
        "research_gate": {
            "video_roc_auc_minimum": 0.90,
            "real_video_fpr_maximum": 0.01,
            "video_roc_auc_pass": bool(test_metrics["roc_auc"] >= 0.90),
            "real_video_fpr_pass": bool(test_metrics["fpr"] <= 0.01),
            "overall_pass": bool(
                test_metrics["roc_auc"] >= 0.90 and test_metrics["fpr"] <= 0.01
            ),
        },
    }


def write_score_records(records: Sequence[ScoreRecord], path: Path) -> None:
    if not records:
        raise ValueError("cannot write empty score records")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
    os.replace(temporary, path)


def read_score_records(path: Path) -> list[ScoreRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            ScoreRecord(
                split=row["split"],
                video_id=row["video_id"],
                label=int(row["label"]),
                frame_index=int(row["frame_index"]),
                score=float(row["score"]),
                latency_ms=float(row.get("latency_ms", 0.0)),
                condition=row.get("condition", "clean"),
            )
            for row in csv.DictReader(handle)
        ]


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="inventory and split the full ZIP")
    inventory.add_argument("zip_path", type=Path)
    inventory.add_argument("--manifest", type=Path, required=True)
    inventory.add_argument("--summary", type=Path, required=True)
    inventory.add_argument("--validation-fraction", type=float, default=0.15)
    inventory.add_argument("--seed", type=int, default=DEFAULT_SEED)

    inventory_directory_parser = commands.add_parser(
        "inventory-directory",
        help="inventory a Kaggle-auto-extracted full dataset directory",
    )
    inventory_directory_parser.add_argument("dataset_root", type=Path)
    inventory_directory_parser.add_argument("--manifest", type=Path, required=True)
    inventory_directory_parser.add_argument("--summary", type=Path, required=True)
    inventory_directory_parser.add_argument("--validation-fraction", type=float, default=0.15)
    inventory_directory_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    extract = commands.add_parser("extract", help="safely extract selected split videos")
    extract.add_argument("zip_path", type=Path)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
    )
    extract.add_argument("--mode", choices=("smoke", "full"), default="full")
    extract.add_argument("--smoke-videos-per-class-per-split", type=int, default=1)
    extract.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("evaluate", help="evaluate private frame-score CSV")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--target-fpr", type=float, default=0.01)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"inventory", "inventory-directory"}:
        if args.command == "inventory":
            rows, test_text = inventory_zip(args.zip_path)
        else:
            rows, test_text = inventory_directory(args.dataset_root)
        assigned = assign_train_validation_split(
            rows,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        write_manifest(assigned, args.manifest)
        summary = inventory_summary(assigned, official_test_text=test_text)
        summary["split_seed"] = args.seed
        summary["validation_fraction"] = args.validation_fraction
        _write_json(summary, args.summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "extract":
        rows = read_manifest(args.manifest)
        selected = rows if args.split == "all" else [row for row in rows if row.split == args.split]
        if args.mode == "smoke":
            selected = select_smoke_rows(
                selected,
                videos_per_class_per_split=args.smoke_videos_per_class_per_split,
            )
        print(
            json.dumps(
                extract_rows(
                    args.zip_path,
                    selected,
                    args.output,
                    overwrite=args.overwrite,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "evaluate":
        report = evaluate_score_records(
            read_score_records(args.predictions),
            target_fpr=args.target_fpr,
        )
        _write_json(report, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
