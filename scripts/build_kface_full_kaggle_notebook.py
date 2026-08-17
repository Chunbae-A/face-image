#!/usr/bin/env python3
"""K-FACE 400명 전체 반복 검증용 비공개 Kaggle 노트북을 생성한다."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks" / "kaggle" / "kface_full_verification"
NOTEBOOK_OUTPUT = OUTPUT_DIR / "notebook.ipynb"
METADATA_OUTPUT = OUTPUT_DIR / "kernel-metadata.json"
DATASET_SOURCE = "hywznn/deepsogak-kface-arcface-private-2026-08-17"
KERNEL_ID = "hywznn/k-face-400"


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
    path = ROOT / "scripts" / "evaluate_kface_full_embeddings.py"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    return code(
        r'''# 3. 검증 코드 준비 — 저장소 버전을 노트북 안에 고정
import base64
import hashlib
import importlib.util
import sys

EMBEDDED_EVALUATOR_B64 = "''' + encoded + r'''"
EMBEDDED_EVALUATOR_SHA256 = "''' + fingerprint + r'''"
CODE_ROOT = Path("/kaggle/working/deepsogak_kface_eval")
SCRIPT_PATH = CODE_ROOT / "scripts" / "evaluate_kface_full_embeddings.py"
SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode(EMBEDDED_EVALUATOR_B64)
if hashlib.sha256(payload).hexdigest() != EMBEDDED_EVALUATOR_SHA256:
    raise RuntimeError("내장 검증 코드의 SHA-256이 일치하지 않습니다.")
SCRIPT_PATH.write_bytes(payload)

spec = importlib.util.spec_from_file_location("kface_full_evaluator", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("검증 코드를 불러오지 못했습니다.")
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)
print({"code_sha256": EMBEDDED_EVALUATOR_SHA256, "script": str(SCRIPT_PATH)})
'''
    )


def build_notebook() -> dict[str, object]:
    cells = [
        markdown(
            """
# K-FACE 400명 전체 얼굴가드 기준값 반복 검증

이 노트북은 새로운 얼굴 모델을 학습하지 않습니다. Mac에서 이미 추출한
`400명 × 저·중화질 전체` ArcFace 특징값을 이용해 딥소각 얼굴가드의 등록
사진 수와 판정 기준값을 검증합니다.

- 등록 사진: 3장, 5장, 9장
- 인물 단위 validation/test 분리
- 반복 seed 5개
- 목표: TAR 90% 이상, FAR 0.1% 이하
- 개별 얼굴 이미지·임베딩·점수는 Output에 저장하지 않음

노트북과 입력 Dataset을 모두 **Private**로 유지합니다. 결과 기준값은 실제
웹·모바일 외부 검증 전까지 `research_only_unapproved` 상태입니다.
"""
        ),
        code(
            """
# 1. 실행 설정과 비공개 처리 동의
import json
import os
from pathlib import Path

I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED = True
RUN_FULL_EVALUATION = True
REFERENCES = (3, 5, 9)
SEEDS = (20260815, 20260816, 20260817, 20260818, 20260819)
TARGET_FAR = 0.001
CALIBRATION_FAR = 0.0009
MINIMUM_DETECTION_SCORE = 0.60
HISTOGRAM_BINS = 40000

IN_KAGGLE = Path("/kaggle/input").is_dir()
if not IN_KAGGLE:
    raise RuntimeError("이 노트북은 Kaggle 전용입니다.")
if not I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED:
    raise PermissionError("K-FACE 비공개 Kaggle 처리를 확인해야 합니다.")
if not RUN_FULL_EVALUATION:
    raise ValueError("RUN_FULL_EVALUATION=True로 바꾸세요.")
print({"kaggle": True, "references": REFERENCES, "seeds": SEEDS})
"""
        ),
        code(
            """
# 2. GPU와 비공개 입력 데이터 확인
import shutil
import subprocess

import torch

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle Notebook의 Accelerator를 GPU로 설정하세요.")
manifest_candidates = sorted(Path("/kaggle/input").rglob("kface_private_manifest.json"))
if len(manifest_candidates) != 1:
    raise FileNotFoundError(
        f"K-FACE 비공개 임베딩 Dataset의 manifest 하나가 필요합니다: {manifest_candidates}"
    )
DATASET_DIR = manifest_candidates[0].parent
private_manifest = json.loads(manifest_candidates[0].read_text(encoding="utf-8"))
if private_manifest.get("subject_count") != 400 or private_manifest.get("chunk_count") != 8800:
    raise RuntimeError(f"400명 전체 처리본이 아닙니다: {private_manifest}")
if private_manifest.get("contains_face_images") is not False:
    raise RuntimeError("원본 얼굴 이미지가 없는 비공개 특징값 Dataset만 사용합니다.")

INPUT_DIR = DATASET_DIR
tar_candidates = sorted(DATASET_DIR.glob("subjects_*.tar"))
if tar_candidates:
    INPUT_DIR = Path("/kaggle/temp/kface_private_embeddings")
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True)
    for position, archive in enumerate(tar_candidates, start=1):
        subprocess.run(["tar", "-xf", str(archive), "-C", str(INPUT_DIR)], check=True)
        print({"extracted_batches": position, "total_batches": len(tar_candidates)})

runtime_chunks = len(list(INPUT_DIR.rglob("subject_*__chunk_*.npz")))
if runtime_chunks != 8800:
    raise RuntimeError(f"실행 임베딩 chunk 수가 다릅니다: {runtime_chunks}/8800")
print({
    "gpu": torch.cuda.get_device_name(0),
    "dataset": str(DATASET_DIR),
    "runtime_input": str(INPUT_DIR),
    "subjects": private_manifest["subject_count"],
    "chunks": private_manifest["chunk_count"],
    "embedding_gb": round(private_manifest["embedding_bytes"] / 1e9, 3),
})
"""
        ),
        embedded_code_cell(),
        code(
            """
# 4. 400명 전체 반복 검증 실행
RESULT_PATH = Path("/kaggle/working/kface_full_verification.json")

def show_progress(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)

result = evaluator.evaluate_full(
    INPUT_DIR,
    references=REFERENCES,
    seeds=SEEDS,
    target_far=TARGET_FAR,
    calibration_far=CALIBRATION_FAR,
    minimum_detection_score=MINIMUM_DETECTION_SCORE,
    bins=HISTOGRAM_BINS,
    device="cuda",
    progress=show_progress,
)
evaluator._atomic_json(RESULT_PATH, result)
print(json.dumps({
    "status": "complete",
    "recommendation": result["recommendation"],
    "processing_minutes": round(result["processing_seconds"] / 60, 2),
}, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """
# 5. 발표·보고서용 결과 그래프 생성
import matplotlib.pyplot as plt

reference_labels = [f"{item} refs" for item in REFERENCES]
minimum_tars = [
    result["aggregates"][f"references_{item}"]["metrics_across_seeds"]
    ["minimum_test_tar"]["minimum"] * 100
    for item in REFERENCES
]
maximum_fars = [
    result["aggregates"][f"references_{item}"]["metrics_across_seeds"]
    ["maximum_test_far"]["maximum"] * 100
    for item in REFERENCES
]

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(reference_labels, minimum_tars, color="#2F6BFF")
axes[0].axhline(90, color="#D97706", linestyle="--", label="TAR gate 90%")
axes[0].set_title("Worst test TAR across 5 seeds")
axes[0].set_ylabel("TAR (%)")
axes[0].legend()
axes[1].bar(reference_labels, maximum_fars, color="#10B981")
axes[1].axhline(0.1, color="#DC2626", linestyle="--", label="FAR gate 0.1%")
axes[1].set_title("Worst test FAR across 5 seeds")
axes[1].set_ylabel("FAR (%)")
axes[1].legend()
figure.suptitle("DeepSogak K-FACE full verification")
figure.tight_layout()
PLOT_PATH = Path("/kaggle/working/kface_full_verification.png")
figure.savefig(PLOT_PATH, dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        code(
            """
# 6. 비식별 결과 파일만 최종 확인
assert result["contains_face_images"] is False
assert result["contains_embeddings"] is False
assert result["contains_subject_identifiers"] is False
assert result["individual_scores_persisted"] is False
assert RESULT_PATH.is_file() and PLOT_PATH.is_file()
print({
    "result_json": str(RESULT_PATH),
    "plot": str(PLOT_PATH),
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
        "title": "딥소각 K-FACE 400명 전체 얼굴가드 반복 검증",
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
