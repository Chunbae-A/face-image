#!/usr/bin/env python3
"""K-FACE 품질 Gate 후보 전체 분석용 비공개 Kaggle 노트북을 생성한다."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks" / "kaggle" / "kface_quality_gate_analysis"
NOTEBOOK_OUTPUT = OUTPUT_DIR / "notebook.ipynb"
METADATA_OUTPUT = OUTPUT_DIR / "kernel-metadata.json"
DATASET_SOURCE = "hywznn/deepsogak-kface-arcface-private-2026-08-17"
KERNEL_ID = "hywznn/k-face-quality-gate-analysis"


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
    evaluator = ROOT / "scripts" / "evaluate_kface_full_embeddings.py"
    analyzer = ROOT / "scripts" / "analyze_kface_quality_gates.py"
    evaluator_payload = evaluator.read_bytes()
    analyzer_payload = analyzer.read_bytes()
    evaluator_encoded = base64.b64encode(evaluator_payload).decode("ascii")
    analyzer_encoded = base64.b64encode(analyzer_payload).decode("ascii")
    evaluator_hash = hashlib.sha256(evaluator_payload).hexdigest()
    analyzer_hash = hashlib.sha256(analyzer_payload).hexdigest()
    return code(
        r'''# 3. 분석 코드 준비 — 실행 버전을 노트북 안에 고정
import base64
import hashlib
import importlib.util
import sys

EMBEDDED_EVALUATOR_B64 = "'''
        + evaluator_encoded
        + r'''"
EMBEDDED_EVALUATOR_SHA256 = "'''
        + evaluator_hash
        + r'''"
EMBEDDED_ANALYZER_B64 = "'''
        + analyzer_encoded
        + r'''"
EMBEDDED_ANALYZER_SHA256 = "'''
        + analyzer_hash
        + r""""
CODE_ROOT = Path("/kaggle/temp/deepsogak_kface_quality/scripts")
CODE_ROOT.mkdir(parents=True, exist_ok=True)

files = {
    "evaluate_kface_full_embeddings.py": (
        EMBEDDED_EVALUATOR_B64,
        EMBEDDED_EVALUATOR_SHA256,
    ),
    "analyze_kface_quality_gates.py": (
        EMBEDDED_ANALYZER_B64,
        EMBEDDED_ANALYZER_SHA256,
    ),
}
for name, (encoded, expected_hash) in files.items():
    payload = base64.b64decode(encoded)
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise RuntimeError(f"내장 코드 SHA-256이 일치하지 않습니다: {name}")
    (CODE_ROOT / name).write_bytes(payload)

sys.path.insert(0, str(CODE_ROOT))
spec = importlib.util.spec_from_file_location(
    "analyze_kface_quality_gates",
    CODE_ROOT / "analyze_kface_quality_gates.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("품질 Gate 분석 코드를 불러오지 못했습니다.")
analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)
print({
    "evaluator_sha256": EMBEDDED_EVALUATOR_SHA256,
    "analyzer_sha256": EMBEDDED_ANALYZER_SHA256,
})
"""
    )


def build_notebook() -> dict[str, object]:
    cells = [
        markdown(
            """
# 딥소각 K-FACE 저화질 실패·품질 Gate 분석

이 노트북은 K-FACE 400명 전체 ArcFace 특징값에서 현재 등록 5장 API를
유지한 채, 어떤 얼굴을 자동 비교하고 어떤 얼굴을 재촬영 요청으로 보내야
하는지 검증합니다.

- 후보: 검출점수, 실제 얼굴 픽셀 크기, 밝기 조합 11개
- 인물 단위 validation/test 분리, seed 5개
- 목표: TAR 90% 이상, FAR 0.1% 이하
- 별도 지표: 자동 처리 coverage
- 개별 얼굴·임베딩·인물 ID·개별 점수는 Output에 저장하지 않음

입력 Dataset과 Notebook은 모두 **Private**입니다. K-FACE 내부에서 통과해도
실제 웹·모바일 외부 검증 전에는 API 기본 동작을 변경하지 않습니다.
"""
        ),
        code(
            """
# 1. 실행 설정과 비공개 처리 동의
import json
from pathlib import Path

I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED = True
RUN_FULL_QUALITY_ANALYSIS = True
REFERENCE_COUNT = 5
SEEDS = (20260815, 20260816, 20260817, 20260818, 20260819)
TARGET_FAR = 0.001
CALIBRATION_FAR = 0.0009
BASELINE_DETECTION_SCORE = 0.60
HISTOGRAM_BINS = 40000
DIAGNOSTIC_THRESHOLD = 0.3784

if not Path("/kaggle/input").is_dir():
    raise RuntimeError("이 노트북은 Kaggle 전용입니다.")
if not I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED:
    raise PermissionError("K-FACE 비공개 Kaggle 처리를 확인해야 합니다.")
if not RUN_FULL_QUALITY_ANALYSIS:
    raise ValueError("RUN_FULL_QUALITY_ANALYSIS=True로 바꾸세요.")
print({"reference_count": REFERENCE_COUNT, "seeds": SEEDS})
"""
        ),
        code(
            """
# 2. GPU와 비공개 입력 데이터 확인
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle Notebook의 Accelerator를 GPU로 설정하세요.")
manifest_candidates = sorted(Path("/kaggle/input").rglob("kface_private_manifest.json"))
if len(manifest_candidates) != 1:
    raise FileNotFoundError(
        f"K-FACE 비공개 특징값 Dataset의 manifest 하나가 필요합니다: {manifest_candidates}"
    )
INPUT_DIR = manifest_candidates[0].parent
private_manifest = json.loads(manifest_candidates[0].read_text(encoding="utf-8"))
if private_manifest.get("subject_count") != 400 or private_manifest.get("chunk_count") != 8800:
    raise RuntimeError(f"400명 전체 처리본이 아닙니다: {private_manifest}")
if private_manifest.get("contains_face_images") is not False:
    raise RuntimeError("원본 얼굴 이미지가 없는 비공개 특징값 Dataset만 사용합니다.")
runtime_chunks = len(list(INPUT_DIR.rglob("subject_*__chunk_*.npz")))
if runtime_chunks != 8800:
    raise RuntimeError(f"특징값 chunk 수가 다릅니다: {runtime_chunks}/8800")
print({
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
# 4. 400명 전체 품질 Gate 후보 반복 검증
RESULT_PATH = Path("/kaggle/working/kface_quality_gate_analysis.json")

def show_progress(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)

result = analyzer.analyze_quality_gates(
    INPUT_DIR,
    reference_count=REFERENCE_COUNT,
    seeds=SEEDS,
    target_far=TARGET_FAR,
    calibration_far=CALIBRATION_FAR,
    baseline_detection_score=BASELINE_DETECTION_SCORE,
    bins=HISTOGRAM_BINS,
    device="cuda",
    diagnostic_threshold=DIAGNOSTIC_THRESHOLD,
    progress=show_progress,
)
analyzer._atomic_json(RESULT_PATH, result)
print(json.dumps({
    "status": "complete",
    "recommendation": result["recommendation"],
    "processing_minutes": round(result["processing_seconds"] / 60, 2),
}, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """
# 5. 발표·보고서용 Gate 비교 그래프
import matplotlib.pyplot as plt
import numpy as np

labels = [item["name"] for item in result["rules"]]
minimum_tar = [
    result["aggregates"][name]["metrics_across_seeds"]["minimum_test_tar"]["minimum"] * 100
    for name in labels
]
maximum_far = [
    result["aggregates"][name]["metrics_across_seeds"]["maximum_test_far"]["maximum"] * 100
    for name in labels
]
low_coverage = [
    result["aggregates"][name]["metrics_across_seeds"]["low_test_coverage"]["minimum"] * 100
    for name in labels
]
medium_coverage = [
    result["aggregates"][name]["metrics_across_seeds"]["medium_test_coverage"]["minimum"] * 100
    for name in labels
]

x = np.arange(len(labels))
figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
axes[0].bar(x, minimum_tar, color="#2F6BFF")
axes[0].axhline(90, color="#D97706", linestyle="--", label="TAR gate 90%")
axes[0].set_title("Worst test TAR")
axes[0].set_ylabel("TAR (%)")
axes[0].legend()
axes[1].bar(x, maximum_far, color="#10B981")
axes[1].axhline(0.1, color="#DC2626", linestyle="--", label="FAR gate 0.1%")
axes[1].set_title("Worst test FAR")
axes[1].set_ylabel("FAR (%)")
axes[1].legend()
width = 0.38
axes[2].bar(x - width / 2, low_coverage, width, label="low")
axes[2].bar(x + width / 2, medium_coverage, width, label="medium")
axes[2].set_title("Minimum automatic coverage")
axes[2].set_ylabel("Coverage (%)")
axes[2].legend()
for axis in axes:
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
figure.suptitle("DeepSogak K-FACE quality gate sweep")
figure.tight_layout()
PLOT_PATH = Path("/kaggle/working/kface_quality_gate_analysis.png")
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
        "title": "k-face-quality-gate-analysis",
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
