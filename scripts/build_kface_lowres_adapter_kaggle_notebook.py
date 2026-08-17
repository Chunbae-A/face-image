#!/usr/bin/env python3
"""K-FACE 저화질 임베딩 어댑터 Private Kaggle Notebook을 생성한다."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks" / "kaggle" / "kface_lowres_embedding_adapter"
NOTEBOOK_OUTPUT = OUTPUT_DIR / "notebook.ipynb"
METADATA_OUTPUT = OUTPUT_DIR / "kernel-metadata.json"
DATASET_SOURCE = "hywznn/deepsogak-kface-arcface-private-2026-08-17"
KERNEL_ID = "hywznn/k-face-lowres-embedding-adapter"


def markdown(source: str) -> dict[str, object]:
    normalized = source.strip()
    return {
        "cell_type": "markdown",
        "id": hashlib.sha256(f"markdown:{normalized}".encode()).hexdigest()[:12],
        "metadata": {},
        "source": normalized.splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    normalized = source.strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.sha256(f"code:{normalized}".encode()).hexdigest()[:12],
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(keepends=True),
    }


def embedded_code_cell() -> dict[str, object]:
    sources = {
        "evaluate_kface_full_embeddings.py": ROOT
        / "scripts"
        / "evaluate_kface_full_embeddings.py",
        "train_kface_lowres_adapter.py": ROOT
        / "scripts"
        / "train_kface_lowres_adapter.py",
    }
    encoded = {
        name: base64.b64encode(path.read_bytes()).decode("ascii")
        for name, path in sources.items()
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    }
    return code(
        r"""# 3. GitHub에서 검증한 학습 코드 버전 고정
import base64
import hashlib
import importlib.util
import sys

EMBEDDED_FILES_B64 = """
        + repr(encoded)
        + r"""
EMBEDDED_FILES_SHA256 = """
        + repr(hashes)
        + r"""
CODE_ROOT = Path("/kaggle/temp/deepsogak_kface_adapter/scripts")
CODE_ROOT.mkdir(parents=True, exist_ok=True)

for name, encoded in EMBEDDED_FILES_B64.items():
    payload = base64.b64decode(encoded)
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_FILES_SHA256[name]:
        raise RuntimeError(f"내장 코드 SHA-256이 일치하지 않습니다: {name}")
    (CODE_ROOT / name).write_bytes(payload)

sys.path.insert(0, str(CODE_ROOT))
spec = importlib.util.spec_from_file_location(
    "train_kface_lowres_adapter",
    CODE_ROOT / "train_kface_lowres_adapter.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("어댑터 학습 코드를 불러오지 못했습니다.")
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)
print(EMBEDDED_FILES_SHA256)
"""
    )


def build_notebook() -> dict[str, object]:
    cells = [
        markdown(
            """
# 딥소각 K-FACE 저화질 ArcFace 임베딩 보정 어댑터

저화질 ArcFace 512차원 특징을 같은 촬영의 중화질 특징에 가깝게
보정하는 작은 residual MLP를 학습합니다.

- 학습 240명·validation 80명·잠긴 test 80명
- 학습·validation·test 인물 중복 0명
- 쌍 cosine 정렬과 인물 대조학습 후보 2개
- validation에서만 후보 선택
- 선택 완료 후 잠긴 test를 한 번만 평가
- 저화질 TAR +2%p, FAR 0.1% 이하일 때만 Private ONNX 후보 생성

원본 얼굴·인물 ID·임베딩·개별 점수는 Output에 저장하지 않습니다.
학습 가중치도 비공개 데이터를 기억할 수 있으므로 Gate 통과 시에만 Private
Output으로 생성하고 GitHub에는 올리지 않습니다.
"""
        ),
        code(
            """
# 1. 잠긴 실험 설정
import json
from pathlib import Path

I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED = True
RUN_FULL_TRAINING = True
SPLIT_SEED = 20260817
TRAINING_SEED = 20260817
REFERENCE_COUNT = 5
MINIMUM_DETECTION_SCORE = 0.60
CALIBRATION_FAR = 0.0008
TARGET_FAR = 0.001
MINIMUM_LOW_TAR_IMPROVEMENT = 0.02
MAXIMUM_MEDIUM_TAR_DROP = 0.01
HIDDEN_DIMENSIONS = 128
RESIDUAL_SCALE = 0.25
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GROUP_SUBJECTS = 32
SAMPLES_PER_SUBJECT = 8
HISTOGRAM_BINS = 40000

if not Path("/kaggle/input").is_dir():
    raise RuntimeError("이 Notebook은 Kaggle 전용입니다.")
if not I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED:
    raise PermissionError("K-FACE Private Kaggle 처리를 확인해야 합니다.")
if not RUN_FULL_TRAINING:
    raise ValueError("RUN_FULL_TRAINING=True로 바꾸세요.")
print({"split_seed": SPLIT_SEED, "training_seed": TRAINING_SEED})
"""
        ),
        code(
            """
# 2. GPU와 Private 특징값 400명 확인
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle Notebook Accelerator를 GPU로 설정하세요.")
manifest_candidates = sorted(Path("/kaggle/input").rglob("kface_private_manifest.json"))
if len(manifest_candidates) != 1:
    raise FileNotFoundError(f"K-FACE Private manifest 하나가 필요합니다: {manifest_candidates}")
INPUT_DIR = manifest_candidates[0].parent
private_manifest = json.loads(manifest_candidates[0].read_text(encoding="utf-8"))
if private_manifest.get("subject_count") != 400 or private_manifest.get("chunk_count") != 8800:
    raise RuntimeError(f"400명 전체 처리본이 아닙니다: {private_manifest}")
if private_manifest.get("contains_face_images") is not False:
    raise RuntimeError("원본 얼굴 이미지가 없는 Private 특짓값만 사용합니다.")
runtime_chunks = len(list(INPUT_DIR.rglob("subject_*__chunk_*.npz")))
if runtime_chunks != 8800:
    raise RuntimeError(f"특징값 chunk 수가 다릅니다: {runtime_chunks}/8800")
print({
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "subjects": private_manifest["subject_count"],
    "chunks": private_manifest["chunk_count"],
    "embedding_gb": round(private_manifest["embedding_bytes"] / 1e9, 3),
})
"""
        ),
        embedded_code_cell(),
        code(
            """
# 4. 240명 학습 → 80명 validation 선택 → 80명 잠긴 test 1회
RESULT_PATH = Path("/kaggle/working/kface_lowres_embedding_adapter.json")

def show_progress(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)

result = trainer.run_experiment(
    INPUT_DIR,
    output_dir=Path("/kaggle/working"),
    split_seed=SPLIT_SEED,
    training_seed=TRAINING_SEED,
    reference_count=REFERENCE_COUNT,
    minimum_detection_score=MINIMUM_DETECTION_SCORE,
    calibration_far=CALIBRATION_FAR,
    target_far=TARGET_FAR,
    minimum_low_tar_improvement=MINIMUM_LOW_TAR_IMPROVEMENT,
    maximum_medium_tar_drop=MAXIMUM_MEDIUM_TAR_DROP,
    hidden_dimensions=HIDDEN_DIMENSIONS,
    residual_scale=RESIDUAL_SCALE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    group_subjects=GROUP_SUBJECTS,
    samples_per_subject=SAMPLES_PER_SUBJECT,
    bins=HISTOGRAM_BINS,
    device="cuda",
    progress=show_progress,
)
trainer._atomic_json(RESULT_PATH, result)
print(json.dumps({
    "status": "complete",
    "selection": result["selection"],
    "gates": result["gates"],
    "api_decision": result["api_decision"],
    "processing_minutes": round(result["processing_seconds"] / 60, 2),
}, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """
# 5. 발표·보고서용 잠긴 test 비교 그래프
import matplotlib.pyplot as plt
import numpy as np

selected = result["selection"]["selected_candidate"]
labels = ["raw ArcFace", selected]
keys = ["baseline_raw_arcface", selected]
low_tar = [result["locked_test"][key]["conditions"]["low"]["tar"] * 100 for key in keys]
medium_tar = [result["locked_test"][key]["conditions"]["medium"]["tar"] * 100 for key in keys]
worst_far = [
    max(
        result["locked_test"][key]["conditions"][resolution]["far"]
        for resolution in ("low", "medium")
    ) * 100
    for key in keys
]

x = np.arange(len(labels))
figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
width = 0.35
axes[0].bar(x - width / 2, low_tar, width, label="low")
axes[0].bar(x + width / 2, medium_tar, width, label="medium")
axes[0].axhline(90, color="#DC2626", linestyle="--", label="TAR gate 90%")
axes[0].set_title("Locked test TAR")
axes[0].set_ylabel("TAR (%)")
axes[0].legend()
axes[1].bar(x, worst_far, color="#10B981")
axes[1].axhline(0.1, color="#DC2626", linestyle="--", label="FAR gate 0.1%")
axes[1].set_title("Locked test worst FAR")
axes[1].set_ylabel("FAR (%)")
axes[1].legend()
for axis in axes:
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
figure.suptitle("DeepSogak K-FACE low-resolution embedding adapter")
figure.tight_layout()
PLOT_PATH = Path("/kaggle/working/kface_lowres_embedding_adapter.png")
figure.savefig(PLOT_PATH, dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        code(
            """
# 6. 누수·비식별·조건부 ONNX 저장 확인
assert result["split"]["subject_overlap_count"] == 0
assert result["split"]["test_used_for_training_or_candidate_selection"] is False
assert result["split"]["locked_test_evaluations"] == 1
assert result["contains_face_images"] is False
assert result["contains_embeddings"] is False
assert result["contains_subject_identifiers"] is False
assert result["individual_scores_persisted"] is False
assert RESULT_PATH.is_file() and PLOT_PATH.is_file()
ONNX_PATH = Path("/kaggle/working/kface_lowres_embedding_adapter.onnx")
assert ONNX_PATH.is_file() == result["gates"]["improvement_gate_passed"]
print({
    "result_json": str(RESULT_PATH),
    "plot": str(PLOT_PATH),
    "private_onnx_created": ONNX_PATH.is_file(),
    "threshold_status": result["threshold_status"],
})
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpu", "is_private": True, "internet": False},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel_metadata() -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "k-face-lowres-embedding-adapter",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [DATASET_SOURCE],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    METADATA_OUTPUT.write_text(
        json.dumps(build_kernel_metadata(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_OUTPUT)
    print(METADATA_OUTPUT)


if __name__ == "__main__":
    main()
