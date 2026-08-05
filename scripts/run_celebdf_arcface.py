#!/usr/bin/env python3
"""Extract video-level ArcFace embeddings from Celeb-DF-v2 Celeb-real videos.

This runner is intended for Google Colab or another environment with OpenCV,
InsightFace, and ONNX Runtime installed.  It never saves face crops or sampled
frames.  Every successful video produces one normalized 512-D embedding, and
the NPZ checkpoint is atomically replaced at a configurable interval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from celebdf_faceguard import (
    ArchiveVideo,
    VideoEmbedding,
    l2_normalize,
    load_video_embeddings,
    read_manifest,
    save_video_embeddings,
    select_smoke_rows,
)


def sample_frame_indices(frame_count: int, requested: int) -> list[int]:
    """Return unique, evenly spaced frame indices while avoiding hard cuts at ends."""
    if frame_count <= 0 or requested <= 0:
        return []
    if frame_count <= requested:
        return list(range(frame_count))
    first = min(frame_count - 1, max(0, int(round(frame_count * 0.08))))
    last = max(first, min(frame_count - 1, int(round(frame_count * 0.92)) - 1))
    indices = np.linspace(first, last, num=requested, dtype=int)
    return sorted(set(int(index) for index in indices))


def _face_area_ratio(face: Any, frame_shape: Sequence[int]) -> float:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    left, top, right, bottom = [float(value) for value in face.bbox]
    area = max(0.0, right - left) * max(0.0, bottom - top)
    return area / float(height * width)


def select_primary_face(
    faces: Sequence[Any],
    frame_shape: Sequence[int],
    running_template: np.ndarray | None,
) -> Any | None:
    """Choose the largest first face, then track by embedding similarity."""
    candidates = [
        face for face in faces if getattr(face, "normed_embedding", None) is not None
    ]
    if not candidates:
        return None
    if running_template is None:
        return max(candidates, key=lambda face: _face_area_ratio(face, frame_shape))
    template = l2_normalize(running_template)
    return max(
        candidates,
        key=lambda face: float(l2_normalize(face.normed_embedding) @ template),
    )


def embed_video(
    video_path: Path,
    row: ArchiveVideo,
    face_app: Any,
    *,
    frames_per_video: int,
    minimum_valid_frames: int,
) -> tuple[VideoEmbedding | None, dict[str, object] | None]:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, {"video_id": row.video_id, "reason": "video_open_failed"}
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(frame_count, frames_per_video)
        if not indices:
            return None, {"video_id": row.video_id, "reason": "invalid_frame_count"}

        embeddings: list[np.ndarray] = []
        detection_scores: list[float] = []
        face_area_ratios: list[float] = []
        decode_seconds = 0.0
        inference_seconds = 0.0
        for frame_index in indices:
            decode_start = time.perf_counter()
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            decode_seconds += time.perf_counter() - decode_start
            if not ok or frame is None:
                continue

            inference_start = time.perf_counter()
            faces = face_app.get(frame)
            inference_seconds += time.perf_counter() - inference_start
            running_template = (
                l2_normalize(np.mean(np.stack(embeddings), axis=0))
                if embeddings
                else None
            )
            selected = select_primary_face(faces, frame.shape, running_template)
            if selected is None:
                continue
            embeddings.append(l2_normalize(selected.normed_embedding))
            detection_scores.append(float(getattr(selected, "det_score", np.nan)))
            face_area_ratios.append(_face_area_ratio(selected, frame.shape))

        if len(embeddings) < minimum_valid_frames:
            return None, {
                "video_id": row.video_id,
                "reason": "insufficient_valid_faces",
                "sampled_frames": len(indices),
                "valid_frames": len(embeddings),
            }
        return (
            VideoEmbedding(
                subject_id=row.subject_id,
                video_id=row.video_id,
                relative_path=row.relative_path,
                embedding=l2_normalize(np.mean(np.stack(embeddings), axis=0)),
                sampled_frames=len(indices),
                valid_frames=len(embeddings),
                mean_detection_score=float(np.nanmean(detection_scores)),
                mean_face_area_ratio=float(np.mean(face_area_ratios)),
                decode_seconds=decode_seconds,
                inference_seconds=inference_seconds,
            ),
            None,
        )
    finally:
        capture.release()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _write_rejects(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _model_hashes(model_root: Path, model_name: str) -> dict[str, str]:
    model_dir = model_root.expanduser() / "models" / model_name
    if not model_dir.exists():
        return {}
    return {
        str(path.relative_to(model_dir)): _sha256(path)
        for path in sorted(model_dir.rglob("*.onnx"))
    }


def initialize_face_app(model_name: str, model_root: Path, det_size: int) -> tuple[Any, dict[str, object]]:
    import insightface  # type: ignore
    import onnxruntime as ort  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore

    available = ort.get_available_providers()
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    if not providers:
        raise RuntimeError(f"no supported ONNX Runtime provider found: {available}")
    app = FaceAnalysis(
        name=model_name,
        root=str(model_root.expanduser()),
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    cuda = "CUDAExecutionProvider" in providers
    app.prepare(
        ctx_id=0 if cuda else -1,
        det_size=(det_size, det_size),
    )
    inventory = {
        "insightface_version": getattr(insightface, "__version__", "unknown"),
        "onnxruntime_version": ort.__version__,
        "onnxruntime_available_providers": available,
        "onnxruntime_selected_providers": providers,
        "device": "cuda" if cuda else "cpu",
        "model_name": model_name,
        "model_root": str(model_root.expanduser()),
        "model_hashes": _model_hashes(model_root, model_name),
    }
    return app, inventory


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if not args.accept_noncommercial_model_license:
        raise PermissionError(
            "InsightFace-provided pretrained models are non-commercial research only; "
            "pass --accept-noncommercial-model-license after reviewing the license."
        )
    manifest_rows = read_manifest(args.manifest)
    selected_rows = manifest_rows
    if args.mode == "smoke":
        selected_rows = select_smoke_rows(
            manifest_rows,
            subjects=args.smoke_subjects,
            videos_per_subject=args.smoke_videos_per_subject,
        )

    existing: list[VideoEmbedding] = []
    if args.output.exists():
        existing = load_video_embeddings(args.output)
    completed = {record.video_id for record in existing}
    records = list(existing)
    rejects: list[dict[str, object]] = []

    face_app, runtime_inventory = initialize_face_app(
        args.model_name,
        args.model_root,
        args.det_size,
    )
    started = datetime.now(timezone.utc)
    processed_since_checkpoint = 0
    attempted = 0
    for index, row in enumerate(selected_rows, start=1):
        if row.video_id in completed:
            continue
        attempted += 1
        video_path = args.video_root / Path(row.relative_path)
        if not video_path.exists():
            rejects.append({"video_id": row.video_id, "reason": "video_missing"})
            if args.fail_fast:
                raise FileNotFoundError(video_path)
            continue
        try:
            record, reject = embed_video(
                video_path,
                row,
                face_app,
                frames_per_video=args.frames_per_video,
                minimum_valid_frames=args.minimum_valid_frames,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            record = None
            reject = {
                "video_id": row.video_id,
                "reason": "unexpected_error",
                "error_type": type(exc).__name__,
                "message": str(exc)[:300],
            }
        if record is not None:
            records.append(record)
            completed.add(record.video_id)
            processed_since_checkpoint += 1
        if reject is not None:
            rejects.append(reject)

        if processed_since_checkpoint >= args.checkpoint_every:
            save_video_embeddings(records, args.output)
            _write_rejects(rejects, args.rejects)
            processed_since_checkpoint = 0
        if index == 1 or index % args.progress_every == 0 or index == len(selected_rows):
            print(
                json.dumps(
                    {
                        "selected": len(selected_rows),
                        "visited": index,
                        "successful_total": len(records),
                        "rejected_this_run": len(rejects),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if not records:
        raise RuntimeError("no video embeddings were produced")
    save_video_embeddings(records, args.output)
    _write_rejects(rejects, args.rejects)
    ended = datetime.now(timezone.utc)
    report = {
        "status": "completed",
        "mode": args.mode,
        "selected_video_count": len(selected_rows),
        "attempted_this_run": attempted,
        "successful_video_count_total": len(records),
        "rejected_this_run": len(rejects),
        "frames_per_video": args.frames_per_video,
        "minimum_valid_frames": args.minimum_valid_frames,
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "video_root": str(args.video_root),
        "output": str(args.output),
        "rejects": str(args.rejects),
        "git_commit": _git_commit(),
        "model_license_scope": "InsightFace-provided weights: non-commercial research only",
        **runtime_inventory,
    }
    args.run_report.parent.mkdir(parents=True, exist_ok=True)
    args.run_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--smoke-subjects", type=int, default=2)
    parser.add_argument("--smoke-videos-per-subject", type=int, default=1)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--minimum-valid-frames", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--model-root", type=Path, default=Path("~/.insightface"))
    parser.add_argument("--accept-noncommercial-model-license", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_pipeline(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
