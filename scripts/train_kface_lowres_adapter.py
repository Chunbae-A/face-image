#!/usr/bin/env python3
"""K-FACE 저화질 ArcFace 특징을 보정하는 residual adapter를 학습한다.

학습·validation·test 인물을 완전히 분리하고 두 학습 손실 후보를
validation에서만 선택한 뒤 잠긴 test를 한 번 평가한다. 원본 얼굴,
인물 ID, 임베딩과 개별 점수는 결과에 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_kface_full_embeddings import (
    ScoreEngine,
    ScoreHistogram,
    _even_positions,
    _load_subject,
    _metrics,
    _threshold_for_far,
    _unit_vector,
    discover_subject_files,
)

EMBEDDING_DIMENSIONS = 512


@dataclass(frozen=True)
class AdapterCandidate:
    """같은 구조에 적용할 학습 손실 조합."""

    name: str
    paired_cosine_weight: float
    supervised_contrastive_weight: float
    identity_preservation_weight: float = 0.05
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("후보 이름이 필요합니다.")
        if self.paired_cosine_weight <= 0:
            raise ValueError("쌍 cosine 손실 가중치는 0보다 커야 합니다.")
        if self.supervised_contrastive_weight < 0:
            raise ValueError("대조 손실 가중치는 0 이상이어야 합니다.")
        if self.identity_preservation_weight < 0 or self.temperature <= 0:
            raise ValueError("보존 가중치와 temperature를 확인하세요.")


DEFAULT_CANDIDATES = (
    AdapterCandidate(
        "paired_cosine",
        paired_cosine_weight=1.0,
        supervised_contrastive_weight=0.0,
    ),
    AdapterCandidate(
        "paired_plus_identity_contrastive",
        paired_cosine_weight=0.5,
        supervised_contrastive_weight=0.5,
    ),
)


def split_subjects(
    subject_ids: Sequence[str],
    *,
    seed: int,
    train_share: float = 0.60,
    validation_share: float = 0.20,
) -> dict[str, list[str]]:
    """재현 가능한 인물 단위 60/20/20 분리."""

    values = sorted(set(subject_ids))
    if len(values) < 10 or not 0 < train_share < 1 or not 0 < validation_share < 1:
        raise ValueError("적절한 인물 수와 분할 비율이 필요합니다.")
    if train_share + validation_share >= 1:
        raise ValueError("학습·validation 비율의 합은 1보다 작아야 합니다.")
    order = np.random.default_rng(seed).permutation(len(values))
    train_end = round(len(values) * train_share)
    validation_end = train_end + round(len(values) * validation_share)
    result = {
        "train": [values[int(item)] for item in order[:train_end]],
        "validation": [values[int(item)] for item in order[train_end:validation_end]],
        "test": [values[int(item)] for item in order[validation_end:]],
    }
    if min(len(items) for items in result.values()) < 2:
        raise ValueError("각 분할에 인물이 2명 이상 필요합니다.")
    if set(result["train"]) & set(result["validation"]):
        raise AssertionError("학습·validation 인물이 겹칩니다.")
    if set(result["train"]) & set(result["test"]):
        raise AssertionError("학습·test 인물이 겹칩니다.")
    if set(result["validation"]) & set(result["test"]):
        raise AssertionError("validation·test 인물이 겹칩니다.")
    return result


def split_fingerprint(subject_ids: Sequence[str]) -> str:
    """인물 ID를 노출하지 않고 분할 재현성만 확인한다."""

    payload = "\n".join(sorted(subject_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _torch_module() -> Any:
    try:
        import torch
    except ImportError:
        raise RuntimeError("어댑터 학습에는 PyTorch가 필요합니다.") from None
    return torch


def build_adapter(*, hidden_dimensions: int = 128, residual_scale: float = 0.25) -> Any:
    """초기에 입력과 동일하게 동작하는 512→hidden→512 residual MLP."""

    if hidden_dimensions <= 0 or residual_scale <= 0:
        raise ValueError("양수 hidden 크기와 residual scale이 필요합니다.")
    torch = _torch_module()

    class LowResolutionEmbeddingAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalization = torch.nn.LayerNorm(EMBEDDING_DIMENSIONS)
            self.input_projection = torch.nn.Linear(
                EMBEDDING_DIMENSIONS, hidden_dimensions
            )
            self.activation = torch.nn.GELU()
            self.output_projection = torch.nn.Linear(
                hidden_dimensions, EMBEDDING_DIMENSIONS
            )
            self.residual_scale = residual_scale
            torch.nn.init.zeros_(self.output_projection.weight)
            torch.nn.init.zeros_(self.output_projection.bias)

        def forward(self, values: Any) -> Any:
            delta = self.output_projection(
                self.activation(self.input_projection(self.normalization(values)))
            )
            return torch.nn.functional.normalize(
                values + self.residual_scale * delta,
                dim=1,
            )

    return LowResolutionEmbeddingAdapter()


def _supervised_contrastive_loss(
    predictions: Any,
    targets: Any,
    labels: Any,
    *,
    temperature: float,
) -> Any:
    """같은 인물의 중화질 특징 전체를 positive로 사용한다."""

    torch = _torch_module()
    logits = predictions @ targets.T / temperature
    positive = labels[:, None].eq(labels[None, :])
    positive_logits = logits.masked_fill(~positive, float("-inf"))
    return -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()


def _load_training_subject(
    paths: Sequence[Path],
    *,
    minimum_detection_score: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    subject = _load_subject(paths)
    mask = (subject["low_quality"][:, 0] >= minimum_detection_score) & (
        subject["medium_quality"][:, 0] >= minimum_detection_score
    )
    positions = np.flatnonzero(mask)
    if len(positions) < 2:
        return (
            np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32),
            np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32),
        )
    positions = positions[rng.permutation(len(positions))]
    return (
        subject["low_embeddings"][positions],
        subject["medium_embeddings"][positions],
    )


def iter_cross_subject_batches(
    subject_files: Mapping[str, Sequence[Path]],
    subject_ids: Sequence[str],
    *,
    minimum_detection_score: float,
    group_subjects: int,
    samples_per_subject: int,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """여러 인물의 쌍을 한 batch에 섞으면서 학습 쌍을 한 번씩 사용한다."""

    if group_subjects < 2 or samples_per_subject <= 0:
        raise ValueError("그룹은 2명 이상이고 인물별 샘플은 양수여야 합니다.")
    rng = np.random.default_rng(seed)
    ordered = [subject_ids[int(item)] for item in rng.permutation(len(subject_ids))]
    completed = 0
    for group_start in range(0, len(ordered), group_subjects):
        group = ordered[group_start : group_start + group_subjects]
        loaded: list[tuple[np.ndarray, np.ndarray]] = []
        for subject_id in group:
            loaded.append(
                _load_training_subject(
                    subject_files[subject_id],
                    minimum_detection_score=minimum_detection_score,
                    rng=rng,
                )
            )
            completed += 1
            if progress and (completed == 1 or completed % 20 == 0):
                progress(
                    {
                        "stage": "loading_training_subjects",
                        "processed_subjects": completed,
                        "total_subjects": len(subject_ids),
                    }
                )
        cursors = [0] * len(loaded)
        while True:
            low_parts: list[np.ndarray] = []
            medium_parts: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            for label, (low, medium) in enumerate(loaded):
                start = cursors[label]
                stop = min(start + samples_per_subject, len(low))
                if stop <= start:
                    continue
                low_parts.append(low[start:stop])
                medium_parts.append(medium[start:stop])
                labels.append(np.full(stop - start, label, dtype=np.int64))
                cursors[label] = stop
            if not low_parts:
                break
            batch_labels = np.concatenate(labels)
            yield (
                np.concatenate(low_parts),
                np.concatenate(medium_parts),
                batch_labels,
            )


def train_candidates(
    subject_files: Mapping[str, Sequence[Path]],
    train_subjects: Sequence[str],
    *,
    candidates: Sequence[AdapterCandidate],
    minimum_detection_score: float,
    hidden_dimensions: int,
    residual_scale: float,
    learning_rate: float,
    weight_decay: float,
    group_subjects: int,
    samples_per_subject: int,
    seed: int,
    device: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """같은 batch로 두 손실 후보를 공정하게 1 epoch 학습한다."""

    torch = _torch_module()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    resolved_device = "cuda" if device == "cuda" else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    models: dict[str, Any] = {}
    optimizers: dict[str, Any] = {}
    for candidate in candidates:
        torch.manual_seed(seed)
        model = build_adapter(
            hidden_dimensions=hidden_dimensions,
            residual_scale=residual_scale,
        ).to(resolved_device)
        models[candidate.name] = model
        optimizers[candidate.name] = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    totals: dict[str, dict[str, float]] = {
        candidate.name: {
            "loss_sum": 0.0,
            "paired_sum": 0.0,
            "contrastive_sum": 0.0,
            "identity_sum": 0.0,
        }
        for candidate in candidates
    }
    batch_count = 0
    pair_count = 0
    started = time.perf_counter()
    for low_values, medium_values, label_values in iter_cross_subject_batches(
        subject_files,
        train_subjects,
        minimum_detection_score=minimum_detection_score,
        group_subjects=group_subjects,
        samples_per_subject=samples_per_subject,
        seed=seed,
        progress=progress,
    ):
        low = torch.as_tensor(low_values, dtype=torch.float32, device=resolved_device)
        medium = torch.as_tensor(
            medium_values, dtype=torch.float32, device=resolved_device
        )
        labels = torch.as_tensor(label_values, dtype=torch.long, device=resolved_device)
        for candidate in candidates:
            model = models[candidate.name]
            optimizer = optimizers[candidate.name]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(low)
            paired = (1.0 - torch.sum(prediction * medium, dim=1)).mean()
            identity = (1.0 - torch.sum(prediction * low, dim=1)).mean()
            if candidate.supervised_contrastive_weight > 0:
                contrastive = _supervised_contrastive_loss(
                    prediction,
                    medium,
                    labels,
                    temperature=candidate.temperature,
                )
            else:
                contrastive = torch.zeros((), device=resolved_device)
            loss = (
                candidate.paired_cosine_weight * paired
                + candidate.supervised_contrastive_weight * contrastive
                + candidate.identity_preservation_weight * identity
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            item = totals[candidate.name]
            item["loss_sum"] += float(loss.detach().cpu())
            item["paired_sum"] += float(paired.detach().cpu())
            item["contrastive_sum"] += float(contrastive.detach().cpu())
            item["identity_sum"] += float(identity.detach().cpu())
        batch_count += 1
        pair_count += len(low_values)
        if progress and (batch_count == 1 or batch_count % 200 == 0):
            progress(
                {
                    "stage": "adapter_training",
                    "batches": batch_count,
                    "pairs": pair_count,
                    "device": resolved_device,
                }
            )
    if batch_count <= 0:
        raise ValueError("학습에 사용할 쌍이 없습니다.")
    summary = {
        name: {
            "mean_loss": item["loss_sum"] / batch_count,
            "mean_paired_cosine_loss": item["paired_sum"] / batch_count,
            "mean_supervised_contrastive_loss": item["contrastive_sum"] / batch_count,
            "mean_identity_preservation_loss": item["identity_sum"] / batch_count,
        }
        for name, item in totals.items()
    }
    metadata = {
        "epochs": 1,
        "batch_count": batch_count,
        "pair_count": pair_count,
        "group_subjects": group_subjects,
        "samples_per_subject_per_batch": samples_per_subject,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "training_seconds": time.perf_counter() - started,
        "device": resolved_device,
    }
    return models, summary, metadata


def _enrollment_templates(
    subject_files: Mapping[str, Sequence[Path]],
    subject_ids: Sequence[str],
    *,
    reference_count: int,
    minimum_detection_score: float,
) -> tuple[np.ndarray, dict[str, set[int]]]:
    centers: list[np.ndarray] = []
    used: dict[str, set[int]] = {}
    for subject_id in subject_ids:
        subject = _load_subject(subject_files[subject_id])
        available = np.flatnonzero(
            subject["medium_quality"][:, 0] >= minimum_detection_score
        )
        if len(available) < reference_count + 1:
            raise ValueError(f"등록 사진이 부족합니다: {subject_id}")
        selected = available[_even_positions(len(available), reference_count)]
        centers.append(
            _unit_vector(np.mean(subject["medium_embeddings"][selected], axis=0))
        )
        used[subject_id] = {int(item) for item in subject["image_indices"][selected]}
    return np.stack(centers), used


def evaluate_candidates(
    subject_files: Mapping[str, Sequence[Path]],
    subject_ids: Sequence[str],
    *,
    models: Mapping[str, Any | None],
    reference_count: int,
    minimum_detection_score: float,
    bins: int,
    device: str,
    split_name: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, ScoreHistogram]]:
    """한 인물 분할의 본인·타인 점수를 후보별 histogram으로 누적한다."""

    torch = _torch_module()
    engine = ScoreEngine(device, bins)
    centers, used = _enrollment_templates(
        subject_files,
        subject_ids,
        reference_count=reference_count,
        minimum_detection_score=minimum_detection_score,
    )
    center_tensor = engine.centers(centers)
    histograms = {
        name: {
            "low": ScoreHistogram.empty(bins),
            "medium": ScoreHistogram.empty(bins),
        }
        for name in models
    }
    for model in models.values():
        if model is not None:
            model.eval()
    for position, subject_id in enumerate(subject_ids):
        subject = _load_subject(subject_files[subject_id])
        excluded = used[subject_id]
        impostor_columns = [
            index for index in range(len(subject_ids)) if index != position
        ]
        for resolution in ("low", "medium"):
            quality = subject[f"{resolution}_quality"]
            mask = quality[:, 0] >= minimum_detection_score
            mask &= np.asarray(
                [int(item) not in excluded for item in subject["image_indices"]],
                dtype=bool,
            )
            queries = subject[f"{resolution}_embeddings"][mask]
            if not len(queries):
                raise ValueError(f"평가 질의가 없습니다: {subject_id}")
            for name, model in models.items():
                if resolution == "low" and model is not None:
                    with torch.inference_mode():
                        tensor = torch.as_tensor(
                            queries,
                            dtype=torch.float32,
                            device=engine.device,
                        )
                        adapted = model(tensor)
                        if engine.device == "cuda":
                            scores = adapted @ center_tensor.T
                        else:
                            scores = adapted.detach().cpu().numpy() @ center_tensor.T
                else:
                    scores = engine.scores(queries, center_tensor)
                item = histograms[name][resolution]
                item.genuine += engine.histogram(engine.select_column(scores, position))
                item.impostor += engine.histogram(
                    engine.select_columns(scores, impostor_columns)
                )
        completed = position + 1
        if progress and (
            completed == 1 or completed % 10 == 0 or completed == len(subject_ids)
        ):
            progress(
                {
                    "stage": f"{split_name}_scoring",
                    "processed_subjects": completed,
                    "total_subjects": len(subject_ids),
                    "candidate_count": len(models),
                    "device": engine.device,
                }
            )
    return histograms


def _validation_metrics(
    histograms: Mapping[str, Mapping[str, ScoreHistogram]],
    *,
    calibration_far: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    results: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for name, conditions in histograms.items():
        candidates = {
            resolution: _threshold_for_far(item.impostor, calibration_far)
            for resolution, item in conditions.items()
        }
        threshold = max(candidates.values())
        thresholds[name] = threshold
        metrics = {
            resolution: _metrics(item, threshold)
            for resolution, item in conditions.items()
        }
        results[name] = {
            "validation_threshold_candidates": candidates,
            "operating_threshold": threshold,
            "conditions": metrics,
        }
    return results, thresholds


def _test_metrics(
    histograms: Mapping[str, Mapping[str, ScoreHistogram]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    return {
        name: {
            "operating_threshold_from_validation": thresholds[name],
            "conditions": {
                resolution: _metrics(item, thresholds[name])
                for resolution, item in conditions.items()
            },
        }
        for name, conditions in histograms.items()
    }


def _state_hash(model: Any) -> str:
    torch = _torch_module()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _parameter_count(model: Any) -> int:
    return int(sum(item.numel() for item in model.parameters()))


def _export_onnx(
    model: Any,
    output: Path,
    *,
    device: str,
) -> dict[str, Any]:
    torch = _torch_module()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    sample = torch.linspace(-1, 1, EMBEDDING_DIMENSIONS).reshape(1, -1)
    sample = torch.nn.functional.normalize(sample, dim=1)
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["low_resolution_embedding"],
        output_names=["adapted_embedding"],
        dynamic_axes={
            "low_resolution_embedding": {0: "batch"},
            "adapted_embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    payload = output.read_bytes()
    verification: dict[str, Any] = {
        "path": output.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_training_device": device,
    }
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        expected = model(sample).detach().numpy()
        observed = session.run(None, {"low_resolution_embedding": sample.numpy()})[0]
        verification["cpu_smoke_max_abs_error"] = float(
            np.max(np.abs(expected - observed))
        )
        verification["cpu_smoke_passed"] = bool(
            verification["cpu_smoke_max_abs_error"] <= 1e-5
        )
    except ImportError:
        verification["cpu_smoke_max_abs_error"] = None
        verification["cpu_smoke_passed"] = False
        verification["cpu_smoke_note"] = "onnxruntime_not_installed"
    return verification


def run_experiment(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
    candidates: Sequence[AdapterCandidate] = DEFAULT_CANDIDATES,
    split_seed: int = 20260817,
    training_seed: int = 20260817,
    reference_count: int = 5,
    minimum_detection_score: float = 0.60,
    calibration_far: float = 0.0008,
    target_far: float = 0.001,
    minimum_low_tar_improvement: float = 0.02,
    maximum_medium_tar_drop: float = 0.01,
    hidden_dimensions: int = 128,
    residual_scale: float = 0.25,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    group_subjects: int = 32,
    samples_per_subject: int = 8,
    bins: int = 40_000,
    device: str = "cuda",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """두 어댑터를 학습하고 validation 선택 후 잠긴 test를 평가한다."""

    candidates = tuple(candidates)
    if not candidates or len({item.name for item in candidates}) != len(candidates):
        raise ValueError("서로 다른 이름의 어댑터 후보가 필요합니다.")
    if not 0 < calibration_far <= target_far < 1:
        raise ValueError("calibration FAR은 target FAR 이하여야 합니다.")
    if bins < 1_000 or reference_count <= 0:
        raise ValueError("등록 수와 histogram bin을 확인하세요.")

    started = time.perf_counter()
    subject_files = discover_subject_files(input_dir)
    subject_ids = sorted(subject_files)
    splits = split_subjects(subject_ids, seed=split_seed)
    models, training_losses, training_metadata = train_candidates(
        subject_files,
        splits["train"],
        candidates=candidates,
        minimum_detection_score=minimum_detection_score,
        hidden_dimensions=hidden_dimensions,
        residual_scale=residual_scale,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        group_subjects=group_subjects,
        samples_per_subject=samples_per_subject,
        seed=training_seed,
        device=device,
        progress=progress,
    )

    validation_models: dict[str, Any | None] = {"baseline_raw_arcface": None}
    validation_models.update(models)
    validation_histograms = evaluate_candidates(
        subject_files,
        splits["validation"],
        models=validation_models,
        reference_count=reference_count,
        minimum_detection_score=minimum_detection_score,
        bins=bins,
        device=device,
        split_name="validation",
        progress=progress,
    )
    validation, thresholds = _validation_metrics(
        validation_histograms,
        calibration_far=calibration_far,
    )
    selected = max(
        models,
        key=lambda name: (
            validation[name]["conditions"]["low"]["tar"],
            -validation[name]["conditions"]["low"]["far"],
            validation[name]["conditions"]["low"]["roc_auc_approx"],
        ),
    )

    test_models = {
        "baseline_raw_arcface": None,
        selected: models[selected],
    }
    test_histograms = evaluate_candidates(
        subject_files,
        splits["test"],
        models=test_models,
        reference_count=reference_count,
        minimum_detection_score=minimum_detection_score,
        bins=bins,
        device=device,
        split_name="locked_test",
        progress=progress,
    )
    test = _test_metrics(
        test_histograms,
        {name: thresholds[name] for name in test_models},
    )

    baseline_low_tar = test["baseline_raw_arcface"]["conditions"]["low"]["tar"]
    selected_low_tar = test[selected]["conditions"]["low"]["tar"]
    improvement = selected_low_tar - baseline_low_tar
    baseline_medium_tar = test["baseline_raw_arcface"]["conditions"]["medium"]["tar"]
    selected_medium_tar = test[selected]["conditions"]["medium"]["tar"]
    medium_drop = baseline_medium_tar - selected_medium_tar
    selected_test_fars = [
        test[selected]["conditions"][resolution]["far"]
        for resolution in ("low", "medium")
    ]
    improvement_gate = (
        improvement >= minimum_low_tar_improvement
        and max(selected_test_fars) <= target_far
        and medium_drop <= maximum_medium_tar_drop
    )
    identity_gate = (
        min(
            test[selected]["conditions"][resolution]["tar"]
            for resolution in ("low", "medium")
        )
        >= 0.90
        and max(selected_test_fars) <= target_far
    )

    artifact: dict[str, Any] | None = None
    if improvement_gate and output_dir is not None:
        artifact = _export_onnx(
            models[selected],
            output_dir / "kface_lowres_embedding_adapter.onnx",
            device=device,
        )

    model_metadata = {
        name: {
            "candidate": asdict(next(item for item in candidates if item.name == name)),
            "parameter_count": _parameter_count(model),
            "state_sha256": _state_hash(model),
            "training_loss": training_losses[name],
        }
        for name, model in models.items()
    }
    return {
        "dataset": "K-FACE",
        "protocol": "subject_disjoint_lowres_embedding_adapter_v1",
        "pipeline_version": "kface-full-paired-v2",
        "input_subjects": len(subject_ids),
        "split": {
            "seed": split_seed,
            "counts": {name: len(items) for name, items in splits.items()},
            "fingerprints": {
                name: split_fingerprint(items) for name, items in splits.items()
            },
            "subject_overlap_count": 0,
            "test_used_for_training_or_candidate_selection": False,
            "locked_test_evaluations": 1,
        },
        "reference_count": reference_count,
        "minimum_detection_score": minimum_detection_score,
        "calibration_far": calibration_far,
        "target_far": target_far,
        "histogram_bins": bins,
        "architecture": {
            "type": "residual_mlp",
            "input_dimensions": EMBEDDING_DIMENSIONS,
            "hidden_dimensions": hidden_dimensions,
            "output_dimensions": EMBEDDING_DIMENSIONS,
            "residual_scale": residual_scale,
            "output_l2_normalized": True,
        },
        "training": training_metadata,
        "models": model_metadata,
        "validation": validation,
        "selection": {
            "selected_candidate": selected,
            "criterion": "highest validation low-resolution TAR, then FAR and ROC-AUC",
            "test_metrics_were_unavailable_during_selection": True,
        },
        "locked_test": test,
        "gates": {
            "minimum_low_tar_improvement": minimum_low_tar_improvement,
            "observed_low_tar_improvement": improvement,
            "maximum_medium_tar_drop": maximum_medium_tar_drop,
            "observed_medium_tar_drop": medium_drop,
            "target_maximum_far": target_far,
            "observed_maximum_far": max(selected_test_fars),
            "improvement_gate_passed": improvement_gate,
            "identity_operating_gate": {
                "minimum_tar": 0.90,
                "maximum_far": target_far,
                "passed": identity_gate,
            },
        },
        "onnx_artifact": artifact,
        "api_decision": (
            "research_candidate_external_validation_required"
            if improvement_gate and identity_gate
            else "do_not_change_api"
        ),
        "processing_seconds": time.perf_counter() - started,
        "contains_raw_paths": False,
        "contains_subject_identifiers": False,
        "contains_face_images": False,
        "contains_embeddings": False,
        "individual_scores_persisted": False,
        "model_weights_persisted": artifact is not None,
        "model_may_encode_private_training_data": True,
        "threshold_status": "research_only_unapproved",
        "note": (
            "Private K-FACE 특짓값으로 학습한 연구다. 내부 Gate를 통과해도 "
            "실제 웹·모바일 외부 검증과 데이터·모델 이용 조건 검토 전에는 "
            "API 기본값을 변경하지 않는다."
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260817)
    parser.add_argument("--training-seed", type=int, default=20260817)
    parser.add_argument("--reference-count", type=int, default=5)
    parser.add_argument("--minimum-detection-score", type=float, default=0.60)
    parser.add_argument("--calibration-far", type=float, default=0.0008)
    parser.add_argument("--target-far", type=float, default=0.001)
    parser.add_argument("--minimum-low-tar-improvement", type=float, default=0.02)
    parser.add_argument("--maximum-medium-tar-drop", type=float, default=0.01)
    parser.add_argument("--hidden-dimensions", type=int, default=128)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--group-subjects", type=int, default=32)
    parser.add_argument("--samples-per-subject", type=int, default=8)
    parser.add_argument("--bins", type=int, default=40_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)

    result = run_experiment(
        args.input_dir,
        output_dir=args.artifact_dir,
        split_seed=args.split_seed,
        training_seed=args.training_seed,
        reference_count=args.reference_count,
        minimum_detection_score=args.minimum_detection_score,
        calibration_far=args.calibration_far,
        target_far=args.target_far,
        minimum_low_tar_improvement=args.minimum_low_tar_improvement,
        maximum_medium_tar_drop=args.maximum_medium_tar_drop,
        hidden_dimensions=args.hidden_dimensions,
        residual_scale=args.residual_scale,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        group_subjects=args.group_subjects,
        samples_per_subject=args.samples_per_subject,
        bins=args.bins,
        device=args.device,
        progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selection": result["selection"],
                "gates": result["gates"],
                "api_decision": result["api_decision"],
                "processing_minutes": round(result["processing_seconds"] / 60, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
