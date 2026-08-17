#!/usr/bin/env python3
"""K-FACE 등록 전략 비교용 Private Kaggle Notebook을 생성한다."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks" / "kaggle" / "kface_enrollment_strategy_benchmark"
NOTEBOOK_OUTPUT = OUTPUT_DIR / "notebook.ipynb"
METADATA_OUTPUT = OUTPUT_DIR / "kernel-metadata.json"
DATASET_SOURCE = "hywznn/deepsogak-kface-arcface-private-2026-08-17"
KERNEL_ID = "hywznn/k-face-enrollment-strategy-benchmark"


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
        "analyze_kface_enrollment_strategies.py": ROOT
        / "scripts"
        / "analyze_kface_enrollment_strategies.py",
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
        r"""# 3. 재현 코드 준비 — GitHub 커밋의 실행 버전을 Notebook에 고정
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
CODE_ROOT = Path("/kaggle/temp/deepsogak_kface_enrollment/scripts")
CODE_ROOT.mkdir(parents=True, exist_ok=True)

for name, encoded in EMBEDDED_FILES_B64.items():
    payload = base64.b64decode(encoded)
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_FILES_SHA256[name]:
        raise RuntimeError(f"내장 코드 SHA-256이 일치하지 않습니다: {name}")
    (CODE_ROOT / name).write_bytes(payload)

sys.path.insert(0, str(CODE_ROOT))
spec = importlib.util.spec_from_file_location(
    "analyze_kface_enrollment_strategies",
    CODE_ROOT / "analyze_kface_enrollment_strategies.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("등록 전략 분석 코드를 불러오지 못했습니다.")
analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)
print(EMBEDDED_FILES_SHA256)
"""
    )


def build_notebook() -> dict[str, object]:
    cells = [
        markdown(
            """
# 딥소각 K-FACE 등록 5장 결합 방식 비교

낮은 품질 질의를 많이 버리지 않고 등록 특징을 만드는 방식으로
얼굴 비교 성능을 높일 수 있는지 검증합니다.

- 현재 단순 평균 5장
- 품질 가중 평균 5장
- 정면·측면 차이를 보존하는 등록 중심 2개
- 등록 중심 2개에 품질 가중 적용
- validation FAR 안전 여유 0.09%·0.08%·0.07%
- K-FACE 400명, 인물 단위 validation/test, seed 5개
- 질의 추가 거절 없음: 자동 처리 coverage 100%

원본 얼굴·임베딩·인물 ID·개별 점수는 Output에 저장하지 않습니다.
이 데이터 내에서 전략을 비교하므로 실제 웹·모바일 외부 검증 전에는
API 기본값을 변경하지 않습니다.
"""
        ),
        code(
            """
# 1. 실행 설정
import json
from pathlib import Path

I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED = True
RUN_FULL_BENCHMARK = True
REFERENCE_COUNT = 5
SEEDS = (20260815, 20260816, 20260817, 20260818, 20260819)
CALIBRATION_FARS = (0.0009, 0.0008, 0.0007)
TARGET_FAR = 0.001
MINIMUM_DETECTION_SCORE = 0.60
HISTOGRAM_BINS = 40000

if not Path("/kaggle/input").is_dir():
    raise RuntimeError("이 Notebook은 Kaggle 전용입니다.")
if not I_CONFIRM_KFACE_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED:
    raise PermissionError("K-FACE 비공개 Kaggle 처리를 확인해야 합니다.")
if not RUN_FULL_BENCHMARK:
    raise ValueError("RUN_FULL_BENCHMARK=True로 바꾸세요.")
print({"references": REFERENCE_COUNT, "seeds": SEEDS, "margins": CALIBRATION_FARS})
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
# 4. 등록 전략 4개 × 기준값 여유 3개 × seed 5개 전체 비교
RESULT_PATH = Path("/kaggle/working/kface_enrollment_strategy_benchmark.json")

def show_progress(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)

result = analyzer.analyze_enrollment_strategies(
    INPUT_DIR,
    reference_count=REFERENCE_COUNT,
    seeds=SEEDS,
    calibration_fars=CALIBRATION_FARS,
    target_far=TARGET_FAR,
    minimum_detection_score=MINIMUM_DETECTION_SCORE,
    bins=HISTOGRAM_BINS,
    device="cuda",
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
# 5. 발표·보고서용 비교 그래프
import matplotlib.pyplot as plt
import numpy as np

labels = list(result["aggregates"])
short_names = {
    "mean_5": "mean",
    "quality_weighted_mean_5": "weighted",
    "dual_prototype_5": "dual",
    "dual_quality_weighted_5": "dual+weighted",
}
display_labels = []
low_tar = []
medium_tar = []
maximum_far = []
for key in labels:
    item = result["aggregates"][key]
    strategy = item["strategy"]["name"]
    margin = item["calibration_far"] * 100
    display_labels.append(f"{short_names[strategy]}\\nval FAR {margin:.02f}%")
    metrics = item["metrics_across_seeds"]
    low_tar.append(metrics["low_test_tar"]["minimum"] * 100)
    medium_tar.append(metrics["medium_test_tar"]["minimum"] * 100)
    maximum_far.append(metrics["maximum_test_far"]["maximum"] * 100)

x = np.arange(len(labels))
figure, axes = plt.subplots(1, 2, figsize=(18, 6.5))
width = 0.38
axes[0].bar(x - width / 2, low_tar, width, label="low")
axes[0].bar(x + width / 2, medium_tar, width, label="medium")
axes[0].axhline(90, color="#DC2626", linestyle="--", label="TAR gate 90%")
axes[0].set_title("Worst test TAR across 5 seeds")
axes[0].set_ylabel("TAR (%)")
axes[0].legend()
axes[1].bar(x, maximum_far, color="#10B981")
axes[1].axhline(0.1, color="#DC2626", linestyle="--", label="FAR gate 0.1%")
axes[1].set_title("Worst test FAR across 5 seeds")
axes[1].set_ylabel("FAR (%)")
axes[1].legend()
for axis in axes:
    axis.set_xticks(x)
    axis.set_xticklabels(display_labels, rotation=50, ha="right", fontsize=8)
figure.suptitle("DeepSogak K-FACE enrollment strategy benchmark (coverage 100%)")
figure.tight_layout()
PLOT_PATH = Path("/kaggle/working/kface_enrollment_strategy_benchmark.png")
figure.savefig(PLOT_PATH, dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        code(
            """
# 6. 비식별 결과만 남았는지 확인
assert result["query_coverage"] == 1.0
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
        "title": "k-face-enrollment-strategy-benchmark",
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
