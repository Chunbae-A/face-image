#!/usr/bin/env python3
"""Validate subject-level FaceGuard train/validation/test separation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "path",
    "subject_id",
    "split",
    "source_id",
    "sha256",
    "is_augmented",
}
ALLOWED_SPLITS = {"train", "validation", "test"}
TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n", ""}


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    severity: str = "error"


def _parse_boolean(value: str, row_number: int) -> tuple[bool, Finding | None]:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True, None
    if normalized in FALSE_VALUES:
        return False, None
    return False, Finding(
        "invalid_boolean",
        f"row {row_number}: is_augmented={value!r} is not a supported boolean",
    )


def _cross_split_findings(
    label: str, values: dict[str, set[str]], allow_empty: bool = False
) -> Iterable[Finding]:
    for value, splits in sorted(values.items()):
        if not value and allow_empty:
            continue
        if len(splits) > 1:
            yield Finding(
                f"cross_split_{label}",
                f"{label} {value!r} occurs in splits {sorted(splits)}",
            )


def validate_manifest(path: Path) -> dict[str, object]:
    findings: list[Finding] = []
    subject_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    seen_paths: set[str] = set()
    split_counts: dict[str, int] = defaultdict(int)
    rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            findings.append(
                Finding("missing_columns", f"missing required columns: {missing}")
            )
            return _report(path, rows, split_counts, findings)

        for row_number, row in enumerate(reader, start=2):
            rows += 1
            split = row["split"].strip().lower()
            subject = row["subject_id"].strip()
            source = row["source_id"].strip()
            sha256 = row["sha256"].strip().lower()
            image_path = row["path"].strip()
            augmented, boolean_finding = _parse_boolean(
                row["is_augmented"], row_number
            )
            if boolean_finding:
                findings.append(boolean_finding)

            if split not in ALLOWED_SPLITS:
                findings.append(
                    Finding(
                        "invalid_split",
                        f"row {row_number}: split {split!r} is not one of {sorted(ALLOWED_SPLITS)}",
                    )
                )
                continue
            split_counts[split] += 1
            if not subject:
                findings.append(
                    Finding("missing_subject", f"row {row_number}: subject_id is empty")
                )
            else:
                subject_splits[subject].add(split)
            if not source:
                findings.append(
                    Finding("missing_source", f"row {row_number}: source_id is empty")
                )
            else:
                source_splits[source].add(split)
            if sha256:
                hash_splits[sha256].add(split)
            else:
                findings.append(
                    Finding(
                        "missing_sha256",
                        f"row {row_number}: sha256 is empty; exact duplicate leakage cannot be checked",
                        severity="warning",
                    )
                )
            if not image_path:
                findings.append(
                    Finding("missing_path", f"row {row_number}: path is empty")
                )
            elif image_path in seen_paths:
                findings.append(
                    Finding(
                        "duplicate_path",
                        f"row {row_number}: path {image_path!r} is repeated",
                    )
                )
            seen_paths.add(image_path)
            if augmented and split != "train":
                findings.append(
                    Finding(
                        "augmentation_outside_train",
                        f"row {row_number}: augmented sample is in {split}",
                    )
                )

    findings.extend(_cross_split_findings("subject", subject_splits))
    findings.extend(_cross_split_findings("source", source_splits))
    findings.extend(_cross_split_findings("sha256", hash_splits))
    return _report(path, rows, split_counts, findings)


def _report(
    path: Path,
    rows: int,
    split_counts: dict[str, int],
    findings: list[Finding],
) -> dict[str, object]:
    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    return {
        "manifest": str(path),
        "rows": rows,
        "split_counts": dict(sorted(split_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "valid": errors == 0,
        "findings": [finding.__dict__ for finding in findings],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    report = validate_manifest(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
