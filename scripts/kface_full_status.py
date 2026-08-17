#!/usr/bin/env python3
"""K-FACE 전체 처리 체크포인트를 집계해 진행률과 예상 완료 시각을 보여준다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def summarize(
    output_dir: Path,
    *,
    total_subjects: int,
    pairs_per_subject: int,
) -> dict[str, object]:
    if total_subjects <= 0 or pairs_per_subject <= 0:
        raise ValueError("전체 인물 수와 인물별 이미지 쌍 수는 양수여야 합니다.")
    totals: Counter[str] = Counter()
    subject_pairs: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    fingerprints: set[str] = set()
    for path in output_dir.glob("subjects/*/checkpoints/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if payload.get("complete") is not True:
            continue
        selected = int(payload.get("selected_pairs", 0))
        totals["checkpoint_count"] += 1
        totals["processed_pairs"] += selected
        totals["accepted_pairs"] += int(payload.get("accepted_pairs", 0))
        totals["rejected_pairs"] += int(payload.get("rejected_pairs", 0))
        totals["elapsed_milliseconds"] += round(
            float(payload.get("elapsed_seconds", 0.0)) * 1000
        )
        reject_reasons.update(payload.get("reject_reasons", {}))
        subject_pairs[path.parents[1].name] += selected
        fingerprint = payload.get("config_fingerprint")
        if fingerprint:
            fingerprints.add(str(fingerprint))

    if len(fingerprints) > 1:
        raise RuntimeError("서로 다른 설정의 체크포인트가 한 폴더에 섞였습니다.")
    total_pairs = total_subjects * pairs_per_subject
    processed_pairs = totals["processed_pairs"]
    elapsed_seconds = totals["elapsed_milliseconds"] / 1000
    pairs_per_second = processed_pairs / elapsed_seconds if elapsed_seconds else 0.0
    remaining_pairs = max(0, total_pairs - processed_pairs)
    remaining_seconds = (
        remaining_pairs / pairs_per_second if pairs_per_second else None
    )
    now = datetime.now().astimezone()
    estimated_finish = (
        now + timedelta(seconds=remaining_seconds)
        if remaining_seconds is not None
        else None
    )
    complete_subjects = sum(
        count == pairs_per_subject for count in subject_pairs.values()
    )
    return {
        "status": "complete" if processed_pairs == total_pairs else "running",
        "total_subjects": total_subjects,
        "complete_subjects": complete_subjects,
        "total_images": total_pairs * 2,
        "processed_images": processed_pairs * 2,
        "progress_percent": round(processed_pairs / total_pairs * 100, 4),
        "accepted_images": totals["accepted_pairs"] * 2,
        "rejected_images": totals["rejected_pairs"] * 2,
        "reject_reasons_by_pair": dict(sorted(reject_reasons.items())),
        "checkpoint_count": totals["checkpoint_count"],
        "measured_images_per_second": round(pairs_per_second * 2, 3),
        "estimated_remaining_hours": (
            round(remaining_seconds / 3600, 3)
            if remaining_seconds is not None
            else None
        ),
        "estimated_finish": (
            estimated_finish.isoformat(timespec="seconds")
            if estimated_finish is not None
            else None
        ),
        "config_fingerprint": next(iter(fingerprints), None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--total-subjects", type=int, default=400)
    parser.add_argument("--pairs-per-subject", type=int, default=10_800)
    args = parser.parse_args()
    payload = summarize(
        args.output_dir,
        total_subjects=args.total_subjects,
        pairs_per_subject=args.pairs_per_subject,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
