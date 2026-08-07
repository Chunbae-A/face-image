#!/usr/bin/env python3
"""Generate the private Kaggle notebook for deepfake score calibration."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "celebdf_score_calibration_kaggle.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def repository_bootstrap_cell() -> dict[str, object]:
    paths = [
        ROOT / "scripts" / "celebdf_deepfake.py",
        ROOT / "scripts" / "run_celebdf_deepfake.py",
        ROOT / "scripts" / "calibrate_deepfake_scores.py",
    ]
    embedded = {
        str(path.relative_to(ROOT)): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }
    fingerprint = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    source = r'''# 3. 실행 코드 준비 — 현재 저장소 코드를 노트북 안에 포함
import base64
import os
from pathlib import Path
import subprocess

EMBEDDED_FILES_B64 = ''' + repr(embedded) + r'''
EMBEDDED_CODE_SHA256 = "''' + fingerprint + r'''"

if IN_KAGGLE and CODE_SOURCE == "github":
    REPO_DIR = Path("/kaggle/temp/face-image")
    if not REPO_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
            check=True,
        )
    CODE_VERSION = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
elif IN_KAGGLE:
    REPO_DIR = Path("/kaggle/temp/face-image")
    for relative_path, encoded in EMBEDDED_FILES_B64.items():
        target = REPO_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded))
    CODE_VERSION = f"embedded:{EMBEDDED_CODE_SHA256[:12]}"
else:
    REPO_DIR = Path.cwd()
    CODE_VERSION = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
    ).strip()

os.chdir(REPO_DIR)
print({"repo": str(REPO_DIR), "code_source": CODE_SOURCE, "code_version": CODE_VERSION})
'''
    return code(source)


CELLS = [
    markdown(
        """
# 3단계 — 딥페이크 점수를 화면용 확률 후보로 보정 (Kaggle 무료 GPU)

이 노트북은 모델을 다시 학습하지 않는다. 1단계 얼굴 crop과 2단계 모델을 이용해
`Validation 836개 + 공식 Test 518개` 영상 점수를 다시 계산하고 다음 세 방법을 비교한다.

- Temperature Scaling
- Platt Scaling
- Isotonic Calibration

개별 영상 이름과 프레임 점수는 `/kaggle/temp`에만 두고 종료 전에 삭제한다.
Kaggle Output에는 비식별 보정값·ECE·Brier·그래프만 남긴다.

## 실행 전

1. Notebook은 반드시 **Private**로 유지한다.
2. Input에 `deepsogak-celebdf-preprocess` Output을 추가한다.
3. Input에 `deepsogak-celebdf-train` Output을 추가한다.
4. Accelerator는 GPU를 선택한다.
5. `Run All`을 누른다.
"""
    ),
    code(
        """
# 1. 설정과 이전 동의 확인
import os
from pathlib import Path

REPO_URL = "https://github.com/Chunbae-A/face-image.git"
BRANCH = "exp/score-calibration"
CODE_SOURCE = "embedded"  # GitHub 상태와 무관하게 같은 코드 실행
I_CONFIRM_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED = True
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = True
RUN_CALIBRATION = True
MODEL_FINGERPRINT = "c32a8532e2e1bd275b833b16460946eb307207098e0c07e2247851b71c23a6f1"
CALIBRATION_VERSION = "celebdf-video-mean16-2026-08-08-v1"

IN_KAGGLE = Path("/kaggle").exists() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if not IN_KAGGLE:
    raise RuntimeError("이 노트북은 Kaggle 전용입니다.")
if not I_CONFIRM_PRIVATE_KAGGLE_PROCESSING_IS_ALLOWED:
    raise PermissionError("Celeb-DF 비공개 Kaggle 처리를 확인해야 합니다.")
if not I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE:
    raise PermissionError("얼굴 crop 전처리의 비상업 연구용 조건을 확인해야 합니다.")
if not RUN_CALIBRATION:
    raise ValueError("RUN_CALIBRATION=True로 바꾸세요.")
print({"kaggle": IN_KAGGLE, "calibration_version": CALIBRATION_VERSION})
"""
    ),
    code(
        """
# 2. 실행 의존성 확인
%pip install -q --no-cache-dir "Pillow==11.3.0"
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
# 4. 비공개 전처리·모델 Output 자동 탐색과 복원
import shutil
import zipfile

preprocess_candidates = sorted(
    Path("/kaggle/input").rglob("celebdf_deepfake_preprocess_private.tar")
)
model_candidates = sorted(
    Path("/kaggle/input").rglob("celebdf_deepfake_private_model.zip")
)
if len(preprocess_candidates) != 1:
    raise FileNotFoundError(
        f"deepsogak-celebdf-preprocess Output의 private TAR 하나가 필요합니다: {preprocess_candidates}"
    )
if len(model_candidates) != 1:
    raise FileNotFoundError(
        f"deepsogak-celebdf-train Output의 private model ZIP 하나가 필요합니다: {model_candidates}"
    )

WORK_ROOT = Path("/kaggle/temp/celebdf_score_calibration")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["tar", "-xf", str(preprocess_candidates[0]), "-C", str(WORK_ROOT)],
    check=True,
)
MODEL_ROOT = WORK_ROOT / "model"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(model_candidates[0]) as archive:
    archive.extract("efficientnet_b4_best.pt", MODEL_ROOT)

CROP_ROOT = WORK_ROOT / "crops"
CROP_MANIFEST = WORK_ROOT / "crop_private_manifest.csv"
CHECKPOINT = MODEL_ROOT / "efficientnet_b4_best.pt"
PRIVATE_SCORES = WORK_ROOT / "frame_scores_private.csv"
BASELINE_METRICS = WORK_ROOT / "baseline_metrics.json"
if not CROP_ROOT.is_dir() or not CROP_MANIFEST.is_file() or not CHECKPOINT.is_file():
    raise RuntimeError("전처리 crop, manifest 또는 checkpoint 복원에 실패했습니다.")
print({
    "preprocess_input": str(preprocess_candidates[0]),
    "model_input": str(model_candidates[0]),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
})
"""
    ),
    code(
        """
# 5. 무료 GPU 확인
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle Settings에서 GPU Accelerator를 선택하세요.")
print({
    "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda,
    "torch": torch.__version__,
})
"""
    ),
    code(
        """
# 6. Validation·공식 Test의 clean 16프레임 점수만 재계산
import sys

subprocess.run([
    sys.executable,
    "scripts/run_celebdf_deepfake.py",
    "evaluate",
    "--crop-manifest", str(CROP_MANIFEST),
    "--crop-root", str(CROP_ROOT),
    "--checkpoint", str(CHECKPOINT),
    "--private-scores", str(PRIVATE_SCORES),
    "--metrics", str(BASELINE_METRICS),
    "--input-size", "380",
    "--batch-size", "16",
    "--workers", "2",
    "--seed", "20260807",
    "--target-fpr", "0.01",
    "--frame-counts", "16",
    "--conditions", "clean",
], check=True)
if not PRIVATE_SCORES.is_file():
    raise RuntimeError("비공개 점수 파일 생성에 실패했습니다.")
print({"private_score_bytes": PRIVATE_SCORES.stat().st_size})
"""
    ),
    code(
        """
# 7. 세 보정법 비교, Gate 판정, reliability diagram 생성
import json

OUTPUT_ROOT = Path("/kaggle/working")
CALIBRATION_JSON = OUTPUT_ROOT / "deepfake_video_calibration.json"
FIGURE = OUTPUT_ROOT / "deepfake_score_calibration.png"

subprocess.run([
    sys.executable,
    "scripts/calibrate_deepfake_scores.py",
    "--private-scores", str(PRIVATE_SCORES),
    "--output", str(CALIBRATION_JSON),
    "--figure", str(FIGURE),
    "--model-fingerprint", MODEL_FINGERPRINT,
    "--calibration-version", CALIBRATION_VERSION,
    "--target-fpr", "0.01",
    "--target-fnr", "0.05",
], check=True)
calibration = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
print({
    "selected_method": calibration["selected_method"],
    "validation_ece": calibration["metrics"]["selected"]["validation"]["ece"],
    "official_test_ece": calibration["metrics"]["selected"]["official_test"]["ece"],
    "official_test_fpr": calibration["metrics"]["official_test_decision"]["fpr"],
    "display_approved": calibration["display_approved"],
})
"""
    ),
    code(
        """
# 8. 비식별 결과만 저장하고 원점수 즉시 삭제
import json

README = OUTPUT_ROOT / "README_score_calibration.md"
README.write_text(f'''# 딥소각 딥페이크 점수 보정 결과

- 보정 방법: {calibration['selected_method']}
- Validation ECE: {calibration['metrics']['selected']['validation']['ece']}
- 공식 Test ECE: {calibration['metrics']['selected']['official_test']['ece']}
- 공식 Test 실제영상 FPR: {calibration['metrics']['official_test_decision']['fpr']}
- 화면 확률 표시 승인: {calibration['display_approved']}
- 상태: {calibration['calibration_status']}

보정은 Validation에서만 선택했고 공식 Test는 최종 평가에만 사용했다.
Gate 미통과 시 API의 `calibrated_probability`는 `null`이어야 한다.
''', encoding="utf-8")

PRIVATE_SCORES.unlink(missing_ok=True)
if PRIVATE_SCORES.exists():
    raise RuntimeError("비공개 원점수 삭제에 실패했습니다.")

RESULT_ZIP = OUTPUT_ROOT / "deepfake_score_calibration_sanitized.zip"
with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(CALIBRATION_JSON, arcname=CALIBRATION_JSON.name)
    archive.write(FIGURE, arcname=FIGURE.name)
    archive.write(README, arcname=README.name)
with zipfile.ZipFile(RESULT_ZIP) as archive:
    names = set(archive.namelist())
    assert "frame_scores_private.csv" not in names
    assert not any(name.endswith((".jpg", ".mp4", ".pt", ".onnx")) for name in names)

print({
    "result": str(RESULT_ZIP),
    "private_scores_deleted": True,
    "contains_faces_or_video_ids": False,
})
"""
    ),
    markdown(
        """
## 결과 읽는 법

- `display_approved: true`: 보정 확률 표시가 연구 Gate를 통과했다는 뜻이다.
- `display_approved: false`: 원점수는 사용할 수 있지만 `84% 확률`처럼 표시하면 안 된다.
- 어떤 경우에도 자동 신고·삭제로 연결하지 않고 사람이 후보를 확인한다.
"""
    ),
]


def build_notebook() -> dict[str, object]:
    return {
        "cells": CELLS,
        "metadata": {
            "kaggle": {
                "name": OUTPUT.name,
                "is_private": True,
                "accelerator": "gpu",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
