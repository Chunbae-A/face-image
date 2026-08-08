#!/usr/bin/env python3
"""Preprocess, train, evaluate, and export the Celeb-DF deepfake baseline.

Heavy dependencies are imported lazily so repository unit tests can validate
sampling, manifests, and score selection without installing PyTorch or
InsightFace.  Face crops, per-video IDs, frame scores, checkpoints, and ONNX
files are private runtime artifacts and must not be committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from celebdf_deepfake import (
    DEFAULT_SEED,
    DatasetVideo,
    ScoreRecord,
    aggregate_video_scores,
    classification_metrics,
    evaluate_score_records,
    latency_summary,
    operating_point_at_recall,
    read_manifest,
    roc_auc,
    select_smoke_rows,
    threshold_at_fpr,
    write_score_records,
)


DEFAULT_INPUT_SIZE = 380
DEFAULT_ALIGNED_CROP_SIZE = 224
SUPPORTED_ARCHITECTURES = ("efficientnet_b4", "xception")
SUPPORTED_NORMALIZATIONS = ("architecture_default", "half")
MODEL_SPECS: dict[str, dict[str, object]] = {
    "efficientnet_b4": {
        "display_name": "EfficientNet-B4",
        "implementation": "torchvision/efficientnet_b4",
        "default_input_size": 380,
        "default_mean": (0.485, 0.456, 0.406),
        "default_std": (0.229, 0.224, 0.225),
    },
    "xception": {
        "display_name": "Xception",
        "implementation": "timm/legacy_xception.tf_in1k",
        "default_input_size": 299,
        "default_mean": (0.5, 0.5, 0.5),
        "default_std": (0.5, 0.5, 0.5),
    },
}
EVALUATION_FRAME_COUNTS = (8, 16, 32)
EVALUATION_CONDITIONS = (
    "clean",
    "jpeg_q30",
    "gaussian_blur_sigma2",
    "low_light_gamma2",
    "downscale_0_25",
)


def model_spec(architecture: str) -> dict[str, object]:
    try:
        return MODEL_SPECS[architecture]
    except KeyError as error:
        raise ValueError(f"unsupported architecture: {architecture}") from error


def normalization_spec(
    architecture: str,
    normalization: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    spec = model_spec(architecture)
    if normalization == "architecture_default":
        return tuple(spec["default_mean"]), tuple(spec["default_std"])  # type: ignore[arg-type]
    if normalization == "half":
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    raise ValueError(f"unsupported normalization: {normalization}")


@dataclass(frozen=True)
class CropRecord:
    split: str
    video_id: str
    label: int
    frame_index: int
    relative_crop_path: str
    detection_score: float
    face_area_ratio: float


def sample_frame_indices(frame_count: int, requested: int) -> list[int]:
    """Choose unique evenly-spaced frames while avoiding title/end cards."""
    if frame_count <= 0 or requested <= 0:
        return []
    if frame_count <= requested:
        return list(range(frame_count))
    first = min(frame_count - 1, max(0, int(round(frame_count * 0.08))))
    last = max(first, min(frame_count - 1, int(round(frame_count * 0.92)) - 1))
    return sorted(
        set(int(index) for index in np.linspace(first, last, num=requested, dtype=int))
    )


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row}) if rows else ["reason", "count"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_crop_manifest(rows: Sequence[CropRecord], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty crop manifest")
    _write_csv_atomic([asdict(row) for row in rows], path)


def read_crop_manifest(path: Path) -> list[CropRecord]:
    rows: list[CropRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                CropRecord(
                    split=raw["split"],
                    video_id=raw["video_id"],
                    label=int(raw["label"]),
                    frame_index=int(raw["frame_index"]),
                    relative_crop_path=raw["relative_crop_path"],
                    detection_score=float(raw["detection_score"]),
                    face_area_ratio=float(raw["face_area_ratio"]),
                )
            )
    if not rows:
        raise ValueError(f"crop manifest is empty: {path}")
    return rows


def select_frame_subset(
    rows: Sequence[CropRecord],
    frames_per_video: int,
) -> list[CropRecord]:
    """Select up to N crops per video without moving a video across splits."""
    if frames_per_video <= 0:
        raise ValueError("frames_per_video must be positive")
    grouped: dict[tuple[str, str], list[CropRecord]] = {}
    for row in rows:
        grouped.setdefault((row.split, row.video_id), []).append(row)
    selected: list[CropRecord] = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda row: row.frame_index)
        labels = {row.label for row in values}
        if len(labels) != 1:
            raise ValueError(f"video has inconsistent crop labels: {key[1]}")
        if len(values) <= frames_per_video:
            selected.extend(values)
            continue
        positions = np.linspace(0, len(values) - 1, num=frames_per_video, dtype=int)
        selected.extend(values[int(position)] for position in sorted(set(positions.tolist())))
    return selected


def _face_area_ratio(face: Any, frame_shape: Sequence[int]) -> float:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    left, top, right, bottom = [float(value) for value in face.bbox]
    return max(0.0, right - left) * max(0.0, bottom - top) / float(height * width)


def select_largest_face(faces: Sequence[Any], frame_shape: Sequence[int]) -> Any | None:
    candidates = [face for face in faces if getattr(face, "kps", None) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda face: _face_area_ratio(face, frame_shape))


def initialize_face_detector(
    model_name: str,
    model_root: Path,
    det_size: int,
) -> tuple[Any, dict[str, object]]:
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
        allowed_modules=["detection"],
        providers=providers,
    )
    use_cuda = "CUDAExecutionProvider" in providers
    app.prepare(ctx_id=0 if use_cuda else -1, det_size=(det_size, det_size))
    model_dir = model_root.expanduser() / "models" / model_name
    model_hashes = {
        str(path.relative_to(model_dir)): _sha256(path)
        for path in sorted(model_dir.rglob("*.onnx"))
    } if model_dir.exists() else {}
    return app, {
        "detector": f"InsightFace/{model_name}/detection",
        "insightface_version": getattr(insightface, "__version__", "unknown"),
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "selected_providers": providers,
        "device": "cuda" if use_cuda else "cpu",
        "detector_model_hashes": model_hashes,
        "detector_license_scope": "InsightFace-provided weights: non-commercial research only",
    }


def _save_rgb_jpeg_atomic(bgr_crop: np.ndarray, path: Path) -> None:
    rgb = np.ascontiguousarray(bgr_crop[..., ::-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    Image.fromarray(rgb).save(temporary, format="JPEG", quality=95, subsampling=0)
    os.replace(temporary, path)


def preprocess_video(
    video_path: Path,
    row: DatasetVideo,
    detector: Any,
    crop_root: Path,
    *,
    frames_per_video: int,
    minimum_valid_frames: int,
    aligned_crop_size: int,
) -> tuple[list[CropRecord], dict[str, object] | None, dict[str, float]]:
    import cv2  # type: ignore
    from insightface.utils import face_align  # type: ignore

    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], {"reason": "video_open_failed"}, {"elapsed_seconds": time.perf_counter() - started}
    records: list[CropRecord] = []
    decode_seconds = 0.0
    detection_seconds = 0.0
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(frame_count, frames_per_video)
        if not indices:
            return [], {"reason": "invalid_frame_count"}, {
                "elapsed_seconds": time.perf_counter() - started
            }
        video_key = _stable_digest(row.video_id)[:20]
        for frame_index in indices:
            decode_start = time.perf_counter()
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            decode_seconds += time.perf_counter() - decode_start
            if not ok or frame is None:
                continue
            detection_start = time.perf_counter()
            faces = detector.get(frame)
            detection_seconds += time.perf_counter() - detection_start
            face = select_largest_face(faces, frame.shape)
            if face is None:
                continue
            aligned = face_align.norm_crop(
                frame,
                landmark=np.asarray(face.kps),
                image_size=aligned_crop_size,
            )
            relative = f"{row.split}/{video_key}/{frame_index:06d}.jpg"
            _save_rgb_jpeg_atomic(aligned, crop_root / relative)
            records.append(
                CropRecord(
                    split=row.split,
                    video_id=row.video_id,
                    label=row.label,
                    frame_index=frame_index,
                    relative_crop_path=relative,
                    detection_score=float(getattr(face, "det_score", np.nan)),
                    face_area_ratio=_face_area_ratio(face, frame.shape),
                )
            )
        if len(records) < minimum_valid_frames:
            return [], {
                "reason": "insufficient_valid_faces",
                "sampled_frames": len(indices),
                "valid_frames": len(records),
            }, {
                "decode_seconds": decode_seconds,
                "detection_seconds": detection_seconds,
                "elapsed_seconds": time.perf_counter() - started,
            }
        return records, None, {
            "decode_seconds": decode_seconds,
            "detection_seconds": detection_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        capture.release()


def preprocess(args: argparse.Namespace) -> dict[str, object]:
    if not args.accept_noncommercial_detector_license:
        raise PermissionError(
            "Review the InsightFace pretrained-model license, then pass "
            "--accept-noncommercial-detector-license."
        )
    rows = read_manifest(args.manifest)
    selected_rows = rows if args.mode == "full" else select_smoke_rows(
        rows,
        videos_per_class_per_split=args.smoke_videos_per_class_per_split,
    )
    detector, runtime = initialize_face_detector(
        args.detector_model,
        args.model_root,
        args.det_size,
    )
    contract = {
        "manifest_sha256": _sha256(args.manifest),
        "frames_per_video": args.frames_per_video,
        "minimum_valid_frames": args.minimum_valid_frames,
        "aligned_crop_size": args.aligned_crop_size,
        "detector_model": args.detector_model,
        "det_size": args.det_size,
        "detector_model_hashes": runtime["detector_model_hashes"],
        "mode": args.mode,
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    records: list[CropRecord] = []
    if args.crop_manifest.exists():
        if not args.run_report.exists():
            raise ValueError("existing crop manifest requires its run report")
        previous = json.loads(args.run_report.read_text(encoding="utf-8"))
        if previous.get("resume_fingerprint") != fingerprint:
            raise ValueError("preprocessing resume settings do not match the existing cache")
        records = read_crop_manifest(args.crop_manifest)
    completed = {row.video_id for row in records}
    rejects: list[dict[str, object]] = []
    timings: list[float] = []
    started = datetime.now(timezone.utc)
    attempted = 0

    def report(status: str) -> dict[str, object]:
        now = datetime.now(timezone.utc)
        completed_videos = len({row.video_id for row in records})
        split_video_counts = {
            split: len({row.video_id for row in records if row.split == split})
            for split in ("train", "validation", "test")
        }
        reject_reasons: dict[str, int] = {}
        for reject in rejects:
            reason = str(reject["reason"])
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        return {
            "status": status,
            "started_utc": started.isoformat(),
            "updated_utc": now.isoformat(),
            "elapsed_seconds": (now - started).total_seconds(),
            "selected_video_count": len(selected_rows),
            "attempted_this_run": attempted,
            "successful_video_count_total": completed_videos,
            "crop_count_total": len(records),
            "successful_videos_by_split": split_video_counts,
            "reject_count_this_run": len(rejects),
            "reject_reasons_this_run": reject_reasons,
            "preprocess_video_seconds_p50": float(np.quantile(timings, 0.50)) if timings else 0.0,
            "preprocess_video_seconds_p95": float(np.quantile(timings, 0.95)) if timings else 0.0,
            "resume_fingerprint": fingerprint,
            "contract": contract,
            **runtime,
        }

    _write_json_atomic(report("running"), args.run_report)
    processed_since_checkpoint = 0
    for index, row in enumerate(selected_rows, start=1):
        if row.video_id in completed:
            continue
        attempted += 1
        video_path = args.video_root / Path(row.relative_path)
        if not video_path.exists():
            rejects.append(
                {"video_key": _stable_digest(row.video_id)[:20], "reason": "video_missing"}
            )
            continue
        try:
            crops, reject, timing = preprocess_video(
                video_path,
                row,
                detector,
                args.crop_root,
                frames_per_video=args.frames_per_video,
                minimum_valid_frames=args.minimum_valid_frames,
                aligned_crop_size=args.aligned_crop_size,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            crops = []
            reject = {"reason": "unexpected_error", "error_type": type(exc).__name__}
            timing = {"elapsed_seconds": 0.0}
        timings.append(float(timing.get("elapsed_seconds", 0.0)))
        if crops:
            records.extend(crops)
            completed.add(row.video_id)
            processed_since_checkpoint += 1
        if reject:
            rejects.append(
                {"video_key": _stable_digest(row.video_id)[:20], **reject}
            )

        if processed_since_checkpoint >= args.checkpoint_every_videos:
            write_crop_manifest(records, args.crop_manifest)
            _write_csv_atomic(rejects, args.rejects)
            _write_json_atomic(report("running"), args.run_report)
            processed_since_checkpoint = 0
        if index == 1 or index % args.progress_every == 0 or index == len(selected_rows):
            print(
                json.dumps(
                    {
                        "selected": len(selected_rows),
                        "visited": index,
                        "successful_videos": len(completed),
                        "crop_count": len(records),
                        "rejects_this_run": len(rejects),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if not records:
        raise RuntimeError("preprocessing produced no valid face crops")
    write_crop_manifest(records, args.crop_manifest)
    _write_csv_atomic(rejects, args.rejects)
    final = report("completed")
    final["ended_utc"] = final["updated_utc"]
    _write_json_atomic(final, args.run_report)
    return final


def apply_evaluation_condition(image: Image.Image, condition: str) -> Image.Image:
    image = image.convert("RGB")
    if condition == "clean":
        return image
    if condition == "jpeg_q30":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=30, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if condition == "gaussian_blur_sigma2":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if condition == "low_light_gamma2":
        array = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(
            np.rint(np.square(array) * 255.0).clip(0, 255).astype(np.uint8)
        )
    if condition == "downscale_0_25":
        width, height = image.size
        reduced = image.resize(
            (max(1, width // 4), max(1, height // 4)),
            resample=Image.Resampling.BILINEAR,
        )
        return reduced.resize((width, height), resample=Image.Resampling.BILINEAR)
    raise ValueError(f"unsupported evaluation condition: {condition}")


class RandomJPEGCompression:
    def __init__(self, probability: float = 0.30, minimum_quality: int = 30):
        self.probability = probability
        self.minimum_quality = minimum_quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=random.randint(self.minimum_quality, 90),
            subsampling=2,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()


class RandomLowLight:
    def __init__(self, probability: float = 0.20):
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return ImageEnhance.Brightness(image).enhance(random.uniform(0.35, 0.75))


class RandomResizeDegradation:
    def __init__(self, probability: float = 0.25):
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        width, height = image.size
        scale = random.uniform(0.25, 0.75)
        reduced = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resample=Image.Resampling.BILINEAR,
        )
        return reduced.resize((width, height), resample=Image.Resampling.BILINEAR)


class AddGaussianNoise:
    def __init__(self, probability: float = 0.20, sigma: float = 0.02):
        self.probability = probability
        self.sigma = sigma

    def __call__(self, tensor: Any) -> Any:
        if random.random() >= self.probability:
            return tensor
        import torch  # type: ignore

        return torch.clamp(tensor + torch.randn_like(tensor) * self.sigma, 0.0, 1.0)


def build_transform(
    *,
    train_mode: bool,
    input_size: int,
    architecture: str = "efficientnet_b4",
    normalization: str = "architecture_default",
):
    from torchvision import transforms  # type: ignore

    mean, std = normalization_spec(architecture, normalization)
    if train_mode:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                RandomResizeDegradation(),
                RandomJPEGCompression(),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=0.20,
                ),
                RandomLowLight(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                AddGaussianNoise(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class CropDataset:
    def __init__(
        self,
        rows: Sequence[CropRecord],
        crop_root: Path,
        transform: Any,
        *,
        condition: str = "clean",
    ):
        self.rows = list(rows)
        self.crop_root = crop_root
        self.transform = transform
        self.condition = condition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.crop_root / row.relative_crop_path
        with Image.open(path) as image:
            transformed = self.transform(
                apply_evaluation_condition(image.convert("RGB"), self.condition)
            )
        return transformed, row.label, index


def build_model(*, architecture: str = "efficientnet_b4", pretrained: bool):
    spec = model_spec(architecture)
    if architecture == "efficientnet_b4":
        from torch import nn  # type: ignore
        from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4  # type: ignore

        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        inventory = {
            "pretrained_weights": (
                "EfficientNet_B4_Weights.DEFAULT" if pretrained else None
            ),
            "pretrained_weights_url": (
                EfficientNet_B4_Weights.DEFAULT.url if pretrained else None
            ),
            "pretrained_weights_license": "torchvision model weight terms",
        }
    elif architecture == "xception":
        try:
            import timm  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "Xception requires timm. Install the pinned deepfake requirements."
            ) from error
        model = timm.create_model(
            "legacy_xception.tf_in1k",
            pretrained=pretrained,
            num_classes=1,
            exportable=True,
        )
        pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
        inventory = {
            "pretrained_weights": "legacy_xception.tf_in1k" if pretrained else None,
            "pretrained_weights_url": (
                (pretrained_cfg.get("url") or pretrained_cfg.get("hf_hub_id"))
                if pretrained
                else None
            ),
            "pretrained_weights_license": pretrained_cfg.get("license", "apache-2.0"),
            "timm_version": getattr(timm, "__version__", "unknown"),
        }
    else:  # pragma: no cover - guarded by model_spec
        raise ValueError(f"unsupported architecture: {architecture}")
    return model, {
        "architecture_id": architecture,
        "architecture": spec["implementation"],
        "display_name": spec["display_name"],
        **inventory,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch  # type: ignore

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _environment_inventory(device: Any) -> dict[str, object]:
    import torch  # type: ignore
    import torchvision  # type: ignore

    inventory: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import timm  # type: ignore

        inventory["timm"] = timm.__version__
    except ImportError:
        inventory["timm"] = None
    return inventory


def _make_loader(
    rows: Sequence[CropRecord],
    crop_root: Path,
    *,
    input_size: int,
    batch_size: int,
    workers: int,
    train_mode: bool,
    seed: int,
    condition: str = "clean",
    architecture: str = "efficientnet_b4",
    normalization: str = "architecture_default",
):
    import torch  # type: ignore
    from torch.utils.data import DataLoader, WeightedRandomSampler  # type: ignore

    dataset = CropDataset(
        rows,
        crop_root,
        build_transform(
            train_mode=train_mode,
            input_size=input_size,
            architecture=architecture,
            normalization=normalization,
        ),
        condition=condition,
    )
    sampler = None
    shuffle = False
    if train_mode:
        labels = np.asarray([row.label for row in rows], dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"training requires both labels, found counts={counts.tolist()}")
        weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def infer_loader(
    model: Any,
    loader: Any,
    rows: Sequence[CropRecord],
    device: Any,
    *,
    condition: str,
) -> list[ScoreRecord]:
    import torch  # type: ignore

    model.eval()
    output: list[ScoreRecord] = []
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(images).flatten()
            probabilities = torch.sigmoid(logits)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0 / len(images)
            for label, row_index, score in zip(
                labels.tolist(),
                indices.tolist(),
                probabilities.detach().cpu().tolist(),
            ):
                row = rows[int(row_index)]
                if int(label) != row.label:
                    raise AssertionError("dataloader label does not match crop manifest")
                output.append(
                    ScoreRecord(
                        split=row.split,
                        video_id=row.video_id,
                        label=row.label,
                        frame_index=row.frame_index,
                        score=float(score),
                        latency_ms=float(latency_ms),
                        condition=condition,
                    )
                )
    return output


def _validation_metric(records: Sequence[ScoreRecord]) -> dict[str, object]:
    videos = aggregate_video_scores(records, method="mean")
    labels = np.asarray([row.label for row in videos], dtype=np.int8)
    scores = np.asarray([row.score for row in videos], dtype=np.float64)
    threshold = threshold_at_fpr(labels, scores, 0.01)
    return classification_metrics(labels, scores, threshold=threshold)


def train(args: argparse.Namespace) -> dict[str, object]:
    import torch  # type: ignore
    from torch import nn  # type: ignore

    _seed_everything(args.seed)
    all_rows = read_crop_manifest(args.crop_manifest)
    selected = select_frame_subset(all_rows, args.train_frames_per_video)
    train_rows = [row for row in selected if row.split == "train"]
    validation_rows = [row for row in selected if row.split == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("training and validation crops are required")
    if any(row.split == "test" for row in train_rows + validation_rows):
        raise AssertionError("official test crop entered model fitting")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            f"CUDA is required for the full {model_spec(args.architecture)['display_name']} training run"
        )
    model, model_inventory = build_model(
        architecture=args.architecture,
        pretrained=True,
    )
    model.to(device)
    train_loader = _make_loader(
        train_rows,
        args.crop_root,
        input_size=args.input_size,
        batch_size=args.batch_size,
        workers=args.workers,
        train_mode=True,
        seed=args.seed,
        architecture=args.architecture,
        normalization=args.normalization,
    )
    validation_loader = _make_loader(
        validation_rows,
        args.crop_root,
        input_size=args.input_size,
        batch_size=args.batch_size,
        workers=args.workers,
        train_mode=False,
        seed=args.seed,
        architecture=args.architecture,
        normalization=args.normalization,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, object]] = []
    best_auc = -math.inf
    epochs_without_improvement = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = labels.to(device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).flatten()
                loss = criterion(logits, targets) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(loss.detach().cpu()) * args.gradient_accumulation_steps

        validation_scores = infer_loader(
            model,
            validation_loader,
            validation_rows,
            device,
            condition="clean",
        )
        validation_metrics = _validation_metric(validation_scores)
        epoch_report = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, len(train_loader)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_video": validation_metrics,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, ensure_ascii=False), flush=True)
        scheduler.step()

        current_auc = float(validation_metrics["roc_auc"])
        if current_auc > best_auc + args.minimum_auc_improvement:
            best_auc = current_auc
            epochs_without_improvement = 0
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "normalization": args.normalization,
                    "input_size": args.input_size,
                    "train_frames_per_video": args.train_frames_per_video,
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "best_validation_video_auc": best_auc,
                    "crop_manifest_sha256": _sha256(args.crop_manifest),
                    "model_inventory": model_inventory,
                },
                temporary,
            )
            os.replace(temporary, args.checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                break

    if not args.checkpoint.exists():
        raise RuntimeError("training did not produce a checkpoint")
    report: dict[str, object] = {
        "status": "completed",
        "architecture_id": args.architecture,
        "architecture": model_inventory["display_name"],
        "objective": "binary cross entropy with logits",
        "label_convention": {"real": 0, "fake": 1},
        "balanced_sampling": "inverse class-frequency WeightedRandomSampler",
        "official_test_used_for_training": False,
        "input_size": args.input_size,
        "normalization": args.normalization,
        "normalization_mean": normalization_spec(
            args.architecture,
            args.normalization,
        )[0],
        "normalization_std": normalization_spec(
            args.architecture,
            args.normalization,
        )[1],
        "train_frames_per_video": args.train_frames_per_video,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_validation_video_auc": best_auc,
        "train_frame_count": len(train_rows),
        "validation_frame_count": len(validation_rows),
        "train_video_count": len({row.video_id for row in train_rows}),
        "validation_video_count": len({row.video_id for row in validation_rows}),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "crop_manifest_sha256": _sha256(args.crop_manifest),
        "history": history,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "amp": use_amp,
        },
        "augmentation": [
            "horizontal_flip",
            "resize_degradation",
            "jpeg_compression",
            "gaussian_blur",
            "low_light",
            "color_jitter",
            "gaussian_noise",
        ],
        **model_inventory,
        **_environment_inventory(device),
    }
    _write_json_atomic(report, args.train_report)
    return report


def _load_checkpoint_model(checkpoint_path: Path, device: Any):
    import torch  # type: ignore

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = str(checkpoint.get("architecture", ""))
    model_spec(architecture)
    model, _ = build_model(architecture=architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _infer_crop_rows(
    model: Any,
    rows: Sequence[CropRecord],
    args: argparse.Namespace,
    device: Any,
    condition: str,
    *,
    architecture: str,
    normalization: str,
) -> list[ScoreRecord]:
    loader = _make_loader(
        rows,
        args.crop_root,
        input_size=args.input_size,
        batch_size=args.batch_size,
        workers=args.workers,
        train_mode=False,
        seed=args.seed,
        condition=condition,
        architecture=architecture,
        normalization=normalization,
    )
    return infer_loader(model, loader, rows, device, condition=condition)


def _validation_selection_report(
    records: Sequence[ScoreRecord],
    *,
    target_fpr: float,
    aggregation_methods: Sequence[str] = ("mean", "median", "top_k"),
) -> dict[str, object]:
    methods: dict[str, dict[str, object]] = {}
    ranked: list[tuple[float, float, float, int, str]] = []
    for method_index, method in enumerate(aggregation_methods):
        videos = aggregate_video_scores(records, method=method)
        labels = np.asarray([row.label for row in videos], dtype=np.int8)
        scores = np.asarray([row.score for row in videos], dtype=np.float64)
        threshold = threshold_at_fpr(labels, scores, target_fpr)
        metrics = classification_metrics(labels, scores, threshold=threshold)
        methods[method] = {"threshold": threshold, "metrics": metrics}
        ranked.append(
            (
                float(metrics["roc_auc"]),
                float(metrics["average_precision"]),
                float(metrics["f1"]),
                -method_index,
                method,
            )
        )
    selected = max(ranked)[-1]
    return {
        "aggregation_candidates": methods,
        "selected_aggregation": selected,
        "selected_threshold": methods[selected]["threshold"],
        "selected_metrics": methods[selected]["metrics"],
    }


def _validation_only_report(
    records: Sequence[ScoreRecord],
    *,
    selected_aggregation: str,
    selected_threshold: float,
    target_fpr: float,
) -> dict[str, object]:
    clean_frames = [row for row in records if row.condition == "clean"]
    if not clean_frames:
        raise ValueError("clean validation scores are required")
    clean_videos = aggregate_video_scores(
        clean_frames,
        method=selected_aggregation,
    )
    labels = np.asarray([row.label for row in clean_videos], dtype=np.int8)
    scores = np.asarray([row.score for row in clean_videos], dtype=np.float64)
    condition_reports: dict[str, dict[str, object]] = {}
    for condition in sorted({row.condition for row in records}):
        condition_frames = [row for row in records if row.condition == condition]
        videos = aggregate_video_scores(
            condition_frames,
            method=selected_aggregation,
        )
        condition_labels = np.asarray([row.label for row in videos], dtype=np.int8)
        condition_scores = np.asarray([row.score for row in videos], dtype=np.float64)
        condition_reports[condition] = {
            "video": classification_metrics(
                condition_labels,
                condition_scores,
                threshold=selected_threshold,
            ),
            "latency": latency_summary(videos),
        }
    return {
        "evaluation_scope": "validation_only",
        "selection_split": "validation",
        "official_test_used_for_selection": False,
        "official_test_inference_performed": False,
        "target_fpr": target_fpr,
        "selected_aggregation": selected_aggregation,
        "selected_threshold": selected_threshold,
        "validation_video": classification_metrics(
            labels,
            scores,
            threshold=selected_threshold,
        ),
        "validation_operating_point_at_recall_0_95": operating_point_at_recall(
            labels,
            scores,
            0.95,
        ),
        "validation_video_latency": latency_summary(clean_videos),
        "condition_validation": condition_reports,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    import torch  # type: ignore

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_checkpoint_model(args.checkpoint, device)
    architecture = str(checkpoint["architecture"])
    normalization = str(checkpoint.get("normalization", "architecture_default"))
    if int(checkpoint["input_size"]) != args.input_size:
        raise ValueError("evaluation input size does not match the checkpoint")
    all_rows = read_crop_manifest(args.crop_manifest)
    if _sha256(args.crop_manifest) != checkpoint["crop_manifest_sha256"]:
        raise ValueError("crop manifest does not match the training checkpoint")

    validation_max = [
        row
        for row in select_frame_subset(all_rows, max(args.frame_counts))
        if row.split == "validation"
    ]
    validation_all_scores = _infer_crop_rows(
        model,
        validation_max,
        args,
        device,
        "clean",
        architecture=architecture,
        normalization=normalization,
    )
    validation_by_key = {
        (row.video_id, row.frame_index): row for row in validation_all_scores
    }
    frame_count_reports: dict[str, dict[str, object]] = {}
    ranked_counts: list[tuple[float, float, float, int, int]] = []
    for frame_count in args.frame_counts:
        crop_subset = [
            row
            for row in select_frame_subset(all_rows, frame_count)
            if row.split == "validation"
        ]
        scores = [validation_by_key[(row.video_id, row.frame_index)] for row in crop_subset]
        report = _validation_selection_report(
            scores,
            target_fpr=args.target_fpr,
            aggregation_methods=args.aggregation_methods,
        )
        frame_count_reports[str(frame_count)] = report
        metrics = report["selected_metrics"]
        ranked_counts.append(
            (
                float(metrics["roc_auc"]),
                float(metrics["average_precision"]),
                float(metrics["f1"]),
                -frame_count,
                frame_count,
            )
        )
    selected_frame_count = max(ranked_counts)[-1]
    selected_validation_crops = [
        row
        for row in select_frame_subset(all_rows, selected_frame_count)
        if row.split == "validation"
    ]
    selected_validation_scores = [
        validation_by_key[(row.video_id, row.frame_index)]
        for row in selected_validation_crops
    ]
    selected_frame_report = frame_count_reports[str(selected_frame_count)]
    selected_aggregation = str(selected_frame_report["selected_aggregation"])
    selected_threshold = float(selected_frame_report["selected_threshold"])

    if args.validation_only:
        validation_scores = list(selected_validation_scores)
        for condition in args.conditions:
            if condition == "clean":
                continue
            validation_scores.extend(
                _infer_crop_rows(
                    model,
                    selected_validation_crops,
                    args,
                    device,
                    condition,
                    architecture=architecture,
                    normalization=normalization,
                )
            )
        write_score_records(validation_scores, args.private_scores)
        validation_metrics = _validation_only_report(
            validation_scores,
            selected_aggregation=selected_aggregation,
            selected_threshold=selected_threshold,
            target_fpr=args.target_fpr,
        )
        expected_validation_videos = len(
            {row.video_id for row in all_rows if row.split == "validation"}
        )
        scored_validation_videos = len(
            {row.video_id for row in selected_validation_crops}
        )
        validation_metrics.update(
            {
                "frame_count_validation_comparison": frame_count_reports,
                "selected_frames_per_video": selected_frame_count,
                "aggregation_candidates": list(args.aggregation_methods),
                "coverage": {
                    "validation_video_count_scored": scored_validation_videos,
                    "validation_video_count_expected": expected_validation_videos,
                    "validation_coverage": (
                        scored_validation_videos / expected_validation_videos
                        if expected_validation_videos
                        else 0.0
                    ),
                },
                "checkpoint_sha256": _sha256(args.checkpoint),
                "crop_manifest_sha256": _sha256(args.crop_manifest),
                "architecture_id": architecture,
                "model": model_spec(architecture)["display_name"],
                "input_size": args.input_size,
                "normalization": normalization,
                "train_frames_per_video": checkpoint["train_frames_per_video"],
                "seed": checkpoint["seed"],
                "environment": _environment_inventory(device),
                "private_artifacts_committed": False,
            }
        )
        _write_json_atomic(validation_metrics, args.metrics)
        return validation_metrics

    # Only after frame count, aggregation, and threshold are fixed on validation
    # do we run the official test split.
    official_test_crops = [
        row
        for row in select_frame_subset(all_rows, selected_frame_count)
        if row.split == "test"
    ]
    all_scores = list(selected_validation_scores)
    for condition in args.conditions:
        all_scores.extend(
            _infer_crop_rows(
                model,
                official_test_crops,
                args,
                device,
                condition,
                architecture=architecture,
                normalization=normalization,
            )
        )
    write_score_records(all_scores, args.private_scores)
    final_metrics = evaluate_score_records(
        all_scores,
        target_fpr=args.target_fpr,
        aggregation_methods=args.aggregation_methods,
    )
    final_metrics["frame_count_validation_comparison"] = frame_count_reports
    final_metrics["selected_frames_per_video"] = selected_frame_count
    final_metrics["aggregation_candidates"] = list(args.aggregation_methods)
    final_metrics["official_test_policy"] = (
        "frame count, aggregation, and threshold selected on validation before test inference"
    )
    final_metrics["evaluation_scope"] = "official_test_after_validation_freeze"
    final_metrics["official_test_inference_performed"] = True
    final_metrics["coverage"] = {
        "validation_video_count": len(
            {row.video_id for row in selected_validation_crops}
        ),
        "official_test_video_count_scored": len(
            {row.video_id for row in official_test_crops}
        ),
        "official_test_expected_video_count": 518,
        "official_test_coverage": len({row.video_id for row in official_test_crops}) / 518.0,
    }
    final_metrics["checkpoint_sha256"] = _sha256(args.checkpoint)
    final_metrics["crop_manifest_sha256"] = _sha256(args.crop_manifest)
    final_metrics["architecture_id"] = architecture
    final_metrics["model"] = model_spec(architecture)["display_name"]
    final_metrics["input_size"] = args.input_size
    final_metrics["normalization"] = normalization
    final_metrics["train_frames_per_video"] = checkpoint["train_frames_per_video"]
    final_metrics["seed"] = checkpoint["seed"]
    final_metrics["environment"] = _environment_inventory(device)
    final_metrics["private_artifacts_committed"] = False
    _write_json_atomic(final_metrics, args.metrics)
    return final_metrics


def export_onnx(args: argparse.Namespace) -> dict[str, object]:
    import torch  # type: ignore

    device = torch.device("cpu")
    model, checkpoint = _load_checkpoint_model(args.checkpoint, device)
    input_size = int(checkpoint["input_size"])
    example = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.onnx.export(
        model,
        example,
        temporary,
        input_names=["image"],
        output_names=["fake_logit"],
        dynamic_axes={"image": {0: "batch"}, "fake_logit": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        # PyTorch 2.9+ defaults to the dynamo exporter, which requires the
        # optional onnxscript package. The legacy exporter matches our
        # dynamic_axes contract and keeps the Kaggle runtime reproducible.
        dynamo=False,
    )
    os.replace(temporary, args.output)
    report = {
        "status": "completed",
        "architecture_id": checkpoint["architecture"],
        "architecture": model_spec(str(checkpoint["architecture"]))["display_name"],
        "normalization": checkpoint.get("normalization", "architecture_default"),
        "input_shape": ["batch", 3, input_size, input_size],
        "output": "fake_logit; sigmoid(logit) is the fake probability-like score",
        "opset": 17,
        "onnx_sha256": _sha256(args.output),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tracked_in_git": False,
    }
    _write_json_atomic(report, args.report)
    return report


def smoke_onnx(args: argparse.Namespace) -> dict[str, object]:
    import onnxruntime as ort  # type: ignore

    rows = read_crop_manifest(args.crop_manifest)
    if not rows:
        raise ValueError("a crop is required for ONNX smoke inference")
    session = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
    )
    transform = build_transform(
        train_mode=False,
        input_size=args.input_size,
        architecture=args.architecture,
        normalization=args.normalization,
    )
    with Image.open(args.crop_root / rows[0].relative_crop_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).numpy()
    started = time.perf_counter()
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logit = float(np.asarray(output).reshape(-1)[0])
    result = {
        "status": "passed",
        "provider": session.get_providers()[0],
        "architecture_id": args.architecture,
        "normalization": args.normalization,
        "input_size": args.input_size,
        "output_is_finite": math.isfinite(logit),
        "processing_ms": elapsed_ms,
        "model_sha256": _sha256(args.model),
        "sample_identity_in_report": False,
    }
    if not result["output_is_finite"]:
        raise RuntimeError("ONNX smoke output is not finite")
    _write_json_atomic(result, args.report)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = commands.add_parser("preprocess")
    preprocess_parser.add_argument("--manifest", type=Path, required=True)
    preprocess_parser.add_argument("--video-root", type=Path, required=True)
    preprocess_parser.add_argument("--crop-root", type=Path, required=True)
    preprocess_parser.add_argument("--crop-manifest", type=Path, required=True)
    preprocess_parser.add_argument("--rejects", type=Path, required=True)
    preprocess_parser.add_argument("--run-report", type=Path, required=True)
    preprocess_parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    preprocess_parser.add_argument("--smoke-videos-per-class-per-split", type=int, default=1)
    preprocess_parser.add_argument("--frames-per-video", type=int, default=32)
    preprocess_parser.add_argument("--minimum-valid-frames", type=int, default=4)
    preprocess_parser.add_argument("--aligned-crop-size", type=int, default=DEFAULT_ALIGNED_CROP_SIZE)
    preprocess_parser.add_argument("--det-size", type=int, default=640)
    preprocess_parser.add_argument("--detector-model", default="buffalo_l")
    preprocess_parser.add_argument("--model-root", type=Path, default=Path("~/.insightface"))
    preprocess_parser.add_argument("--checkpoint-every-videos", type=int, default=25)
    preprocess_parser.add_argument("--progress-every", type=int, default=25)
    preprocess_parser.add_argument("--fail-fast", action="store_true")
    preprocess_parser.add_argument(
        "--accept-noncommercial-detector-license",
        action="store_true",
    )

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--crop-manifest", type=Path, required=True)
    train_parser.add_argument("--crop-root", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--train-report", type=Path, required=True)
    train_parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default="efficientnet_b4",
    )
    train_parser.add_argument(
        "--normalization",
        choices=SUPPORTED_NORMALIZATIONS,
        default="architecture_default",
    )
    train_parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    train_parser.add_argument("--train-frames-per-video", type=int, default=16)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--early-stopping-patience", type=int, default=3)
    train_parser.add_argument("--minimum-auc-improvement", type=float, default=1e-4)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--disable-amp", action="store_true")
    train_parser.add_argument("--require-cuda", action="store_true")

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--crop-manifest", type=Path, required=True)
    evaluate_parser.add_argument("--crop-root", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--private-scores", type=Path, required=True)
    evaluate_parser.add_argument("--metrics", type=Path, required=True)
    evaluate_parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--workers", type=int, default=2)
    evaluate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    evaluate_parser.add_argument("--target-fpr", type=float, default=0.01)
    evaluate_parser.add_argument(
        "--validation-only",
        action="store_true",
        help="score validation conditions without running official test inference",
    )
    evaluate_parser.add_argument(
        "--frame-counts",
        type=int,
        nargs="+",
        default=list(EVALUATION_FRAME_COUNTS),
    )
    evaluate_parser.add_argument(
        "--aggregation-methods",
        nargs="+",
        choices=("mean", "median", "top_k"),
        default=["mean", "median", "top_k"],
    )
    evaluate_parser.add_argument(
        "--conditions",
        nargs="+",
        choices=EVALUATION_CONDITIONS,
        default=list(EVALUATION_CONDITIONS),
    )

    export_parser = commands.add_parser("export-onnx")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--report", type=Path, required=True)

    smoke_parser = commands.add_parser("smoke-onnx")
    smoke_parser.add_argument("--model", type=Path, required=True)
    smoke_parser.add_argument("--crop-manifest", type=Path, required=True)
    smoke_parser.add_argument("--crop-root", type=Path, required=True)
    smoke_parser.add_argument("--report", type=Path, required=True)
    smoke_parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    smoke_parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default="efficientnet_b4",
    )
    smoke_parser.add_argument(
        "--normalization",
        choices=SUPPORTED_NORMALIZATIONS,
        default="architecture_default",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preprocess":
        result = preprocess(args)
    elif args.command == "train":
        result = train(args)
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "export-onnx":
        result = export_onnx(args)
    elif args.command == "smoke-onnx":
        result = smoke_onnx(args)
    else:  # pragma: no cover
        raise AssertionError(f"unexpected command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
