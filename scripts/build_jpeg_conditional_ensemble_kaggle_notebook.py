#!/usr/bin/env python3
"""JPEG 조건부 EfficientNet-B4·Xception 앙상블 Kaggle 노트북을 생성한다."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "celebdf_jpeg_conditional_ensemble_kaggle.ipynb"


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
        ROOT / "configs" / "deepfake" / "jpeg_conditional_ensemble.json",
        ROOT / "scripts" / "celebdf_deepfake.py",
        ROOT / "scripts" / "run_celebdf_deepfake.py",
        ROOT / "scripts" / "optimize_deepfake_score_ensemble.py",
    ]
    embedded = {
        str(path.relative_to(ROOT)): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }
    fingerprint = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    source = r'''# 3. 실행 코드 준비 — GitHub 권한이 필요 없는 내장 코드가 기본값
import base64
import os
from pathlib import Path
import subprocess

EMBEDDED_FILES_B64 = ''' + repr(embedded) + r'''
EMBEDDED_CODE_SHA256 = "''' + fingerprint + r'''"

if IN_KAGGLE and CODE_SOURCE == "github":
    REPO_DIR = Path("/kaggle/temp/face-image")
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
# JPEG 압축 조건부 두 모델 결합 실험 — Kaggle 무료 GPU

이 노트북은 이미 학습한 EfficientNet-B4와 Xception을 다시 학습하지 않는다. 같은 Validation 얼굴 프레임에 두 모델을 실행한 뒤 다음 세 방법을 비교한다.

1. EfficientNet-B4만 사용
2. 모든 입력에서 두 모델 점수를 결합
3. JPEG 압축이 심한 조건에서만 Xception 점수를 결합

결합 가중치와 판정 기준은 Validation에서만 고르며 공식 Test는 실행하지 않는다. 효과가 없으면 기존 EfficientNet-B4 단독을 유지한다. 얼굴 crop, 프레임 점수와 checkpoint는 `/kaggle/temp`에서만 사용하고 Output에는 비식별 집계 결과만 남긴다.
"""
    ),
    code(
        """
# 1. 사용자가 확인할 설정
REPO_URL = "https://github.com/Chunbae-A/face-image.git"
BRANCH = "agent/jpeg-conditional-ensemble"
CODE_SOURCE = "embedded"  # 권한 문제를 피하려면 embedded 유지
PREPROCESS_ARCHIVE_PATH = ""  # 비우면 Kaggle Input에서 자동 탐색
PRIVATE_MODELS_ZIP_PATH = ""  # 비우면 Kaggle Input에서 자동 탐색

I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED = False
I_CONFIRM_PRIVATE_MODEL_OUTPUT_MAY_BE_USED = False

import os
from pathlib import Path
import sys

IN_KAGGLE = Path("/kaggle").exists() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if not I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED:
    raise PermissionError("비공개 얼굴 crop Output 사용 권한을 확인하세요.")
if not I_CONFIRM_PRIVATE_MODEL_OUTPUT_MAY_BE_USED:
    raise PermissionError("비공개 EfficientNet-B4·Xception 모델 Output 사용 권한을 확인하세요.")
print({"kaggle": IN_KAGGLE, "selection_split": "validation", "official_test": "locked"})
"""
    ),
    code(
        """
# 2. 모델 추론 의존성 — Kaggle의 torch/torchvision은 교체하지 않음
%pip install -q --no-cache-dir "timm==1.0.28" "Pillow==11.3.0"
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
# 4. 고정 설정과 GPU 확인
import json
import torch
import timm
import torchvision

CONFIG_PATH = REPO_DIR / "configs" / "deepfake" / "jpeg_conditional_ensemble.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
assert CONFIG["selection_split"] == "validation"
assert CONFIG["official_test_used_for_selection"] is False
assert CONFIG["specialist_conditions"] == ["jpeg_q30"]

cuda_available = torch.cuda.is_available()
print({
    "experiment": CONFIG["experiment_id"],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "cuda_available": cuda_available,
    "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
})
if IN_KAGGLE and not cuda_available:
    raise RuntimeError("Kaggle Settings에서 GPU를 선택하고 다시 시작하세요.")
"""
    ),
    code(
        """
# 5. 비공개 얼굴 crop과 기존 두 모델 복원
import shutil
import tarfile
import zipfile

def find_one(explicit_path, filename):
    candidates = (
        [Path(explicit_path).expanduser()]
        if explicit_path.strip()
        else sorted(Path("/kaggle/input").rglob(filename))
    )
    exact = [path for path in candidates if path.is_file()]
    if len(exact) != 1:
        raise FileNotFoundError(f"{filename} 하나가 필요합니다: {exact}")
    return exact[0]

def safe_extract_zip(archive_path, output_dir):
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if output_dir not in target.parents and target != output_dir:
                raise RuntimeError("모델 ZIP에 안전하지 않은 경로가 있습니다.")
        archive.extractall(output_dir)

PREPROCESS_ARCHIVE = find_one(
    PREPROCESS_ARCHIVE_PATH,
    "celebdf_deepfake_preprocess_private.tar",
)
PRIVATE_MODELS_ZIP = find_one(
    PRIVATE_MODELS_ZIP_PATH,
    "effb4_xception_private_models.zip",
)

WORK_ROOT = Path("/kaggle/temp/jpeg_conditional_ensemble") if IN_KAGGLE else REPO_DIR / "outputs" / "jpeg_conditional_ensemble"
CROP_STAGE = WORK_ROOT / "preprocess"
MODEL_STAGE = WORK_ROOT / "models"
PRIVATE_SCORE_ROOT = WORK_ROOT / "private_scores"
for directory in (CROP_STAGE, MODEL_STAGE, PRIVATE_SCORE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

with tarfile.open(PREPROCESS_ARCHIVE) as archive:
    archive.extractall(CROP_STAGE, filter="data")
safe_extract_zip(PRIVATE_MODELS_ZIP, MODEL_STAGE)

CROP_ROOT = CROP_STAGE / "crops"
CROP_MANIFEST = CROP_STAGE / "crop_private_manifest.csv"
CHECKPOINTS = {
    "efficientnet_b4": MODEL_STAGE / "efficientnet_b4" / "best.pt",
    "xception": MODEL_STAGE / "xception" / "best.pt",
}
if not CROP_ROOT.is_dir() or not CROP_MANIFEST.is_file():
    raise RuntimeError("비공개 얼굴 crop 복원에 실패했습니다.")
if not all(path.is_file() for path in CHECKPOINTS.values()):
    raise RuntimeError(f"두 모델 checkpoint 복원에 실패했습니다: {CHECKPOINTS}")

OUTPUT_ROOT = Path("/kaggle/working/jpeg_conditional_ensemble") if IN_KAGGLE else WORK_ROOT / "sanitized"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
ENSEMBLE_REPORT = OUTPUT_ROOT / "jpeg_conditional_ensemble_validation.json"
FIGURE = OUTPUT_ROOT / "jpeg_conditional_ensemble_validation.png"
SUMMARY = OUTPUT_ROOT / "README.md"
RESULT_ZIP = (Path("/kaggle/working") if IN_KAGGLE else OUTPUT_ROOT) / "jpeg_conditional_ensemble_sanitized_results.zip"

print({
    "preprocess_input": PREPROCESS_ARCHIVE.name,
    "model_input": PRIVATE_MODELS_ZIP.name,
    "crop_files": sum(1 for _ in CROP_ROOT.rglob("*.jpg")),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
})
"""
    ),
    code(
        """
# 6. 두 모델 Validation 5조건 추론 — 학습·공식 Test 없음
SCORE_PATHS = {}
METRIC_PATHS = {}
for model_id, checkpoint in CHECKPOINTS.items():
    score_path = PRIVATE_SCORE_ROOT / f"{model_id}_validation_private.csv"
    metric_path = WORK_ROOT / f"{model_id}_validation_aggregate.json"
    SCORE_PATHS[model_id] = score_path
    METRIC_PATHS[model_id] = metric_path
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "evaluate",
        "--crop-manifest", str(CROP_MANIFEST),
        "--crop-root", str(CROP_ROOT),
        "--checkpoint", str(checkpoint),
        "--private-scores", str(score_path),
        "--metrics", str(metric_path),
        "--input-size", "256",
        "--batch-size", "16",
        "--seed", "20260808",
        "--target-fpr", "0.01",
        "--validation-only",
        "--frame-counts", "16",
        "--aggregation-methods", "mean",
        "--conditions", *CONFIG["conditions"],
    ], check=True)
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    assert metrics["evaluation_scope"] == "validation_only"
    assert metrics["official_test_inference_performed"] is False
    print({
        "model": model_id,
        "validation_auc": metrics["validation_video"]["roc_auc"],
        "private_frame_scores": score_path.stat().st_size,
    })
"""
    ),
    code(
        """
# 7. Validation 점수 결합 정책 자동 선택
subprocess.run([
    sys.executable,
    "scripts/optimize_deepfake_score_ensemble.py",
    "--primary-scores", str(SCORE_PATHS["efficientnet_b4"]),
    "--specialist-scores", str(SCORE_PATHS["xception"]),
    "--config", str(CONFIG_PATH),
    "--output", str(ENSEMBLE_REPORT),
], check=True)

RESULT = json.loads(ENSEMBLE_REPORT.read_text(encoding="utf-8"))
assert RESULT["official_test_used_for_selection"] is False
assert RESULT["privacy"]["contains_frame_scores"] is False
print({
    "selected_policy": RESULT["selected_policy"],
    "ensemble_selected": RESULT["ensemble_selected"],
    "paired_frame_count": RESULT["paired_frame_count"],
    "paired_video_count": RESULT["paired_video_count"],
    "selection_fingerprint": RESULT["selection_fingerprint_sha256"],
})
"""
    ),
    code(
        """
# 8. 비식별 비교 그래프와 한국어 요약
import matplotlib.pyplot as plt

policy_by_id = {row["policy_id"]: row for row in RESULT["candidate_policies"]}
baseline = policy_by_id["primary_only"]
selected = policy_by_id[RESULT["selected_policy"]]
condition_names = CONFIG["conditions"]
baseline_auc = [baseline["condition_validation"][name]["video"]["roc_auc"] for name in condition_names]
selected_auc = [selected["condition_validation"][name]["video"]["roc_auc"] for name in condition_names]
baseline_fpr = [baseline["condition_validation"][name]["operating_point_at_target_recall"]["fpr"] for name in condition_names]
selected_fpr = [selected["condition_validation"][name]["operating_point_at_target_recall"]["fpr"] for name in condition_names]

x = range(len(condition_names))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].plot(x, baseline_auc, marker="o", label="EfficientNet-B4 only")
axes[0].plot(x, selected_auc, marker="o", label="Selected policy")
axes[0].set_title("Validation ROC-AUC")
axes[0].set_xticks(list(x), condition_names, rotation=25, ha="right")
axes[0].legend()
axes[1].plot(x, baseline_fpr, marker="o", label="EfficientNet-B4 only")
axes[1].plot(x, selected_fpr, marker="o", label="Selected policy")
axes[1].set_title("FPR at Recall ≥ 95%")
axes[1].set_xticks(list(x), condition_names, rotation=25, ha="right")
axes[1].legend()
fig.suptitle(f"Selected: {RESULT['selected_policy']}")
fig.tight_layout()
fig.savefig(FIGURE, dpi=160, bbox_inches="tight")
plt.show()

jpeg = selected["comparison_to_primary"]
SUMMARY.write_text(f'''# JPEG 조건부 두 모델 결합 Validation 결과

- 선택 정책: `{RESULT['selected_policy']}`
- 두 모델 결합 채택: `{RESULT['ensemble_selected']}`
- JPEG ROC-AUC 개선량: `{jpeg['specialist_auc_improvement']:.8f}`
- JPEG Recall 95% 지점 FPR 개선량: `{jpeg['specialist_fpr_improvement_at_target_recall']:.8f}`
- 공식 Test 사용: `False`
- 운영 승인: `False`
- 선택 fingerprint: `{RESULT['selection_fingerprint_sha256']}`

이 결과는 Celeb-DF Validation에서 결합 방식을 고른 연구 결과다. 외부 웹 영상 검증 전에는 자동 차단이나 딥페이크 확정에 사용하지 않는다. 결합이 선택되지 않았다면 기존 EfficientNet-B4 단독을 유지한다.
''', encoding="utf-8")
print(SUMMARY.read_text(encoding="utf-8"))
"""
    ),
    code(
        """
# 9. 공개 가능한 결과만 ZIP으로 묶고 비공개 임시 파일 삭제
import zipfile

with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in (ENSEMBLE_REPORT, FIGURE, SUMMARY):
        archive.write(path, arcname=path.name)

with zipfile.ZipFile(RESULT_ZIP) as archive:
    names = set(archive.namelist())
    assert not any(name.endswith((".csv", ".jpg", ".mp4", ".pt", ".onnx", ".tar")) for name in names)
    assert {ENSEMBLE_REPORT.name, FIGURE.name, SUMMARY.name}.issubset(names)

shutil.rmtree(WORK_ROOT)
assert not PRIVATE_SCORE_ROOT.exists()
assert not MODEL_STAGE.exists()
assert not CROP_STAGE.exists()
print({
    "sanitized_result": str(RESULT_ZIP),
    "selected_policy": RESULT["selected_policy"],
    "ensemble_selected": RESULT["ensemble_selected"],
    "private_runtime_deleted": True,
    "warning": "결과가 좋아도 외부 검증 전에는 운영 모델이 아닙니다.",
})
"""
    ),
    markdown(
        """
## 결과를 어떻게 읽나요?

- `ensemble_selected=true`: JPEG 조건에서 Xception을 보조로 실행할 근거가 Validation에서 확인됐다.
- `ensemble_selected=false`: 두 모델을 섞어도 채택 기준을 못 넘었으므로 EfficientNet-B4 단독을 유지한다.
- `specialist_auc_improvement`: JPEG 조건에서 점수 순서 구분이 얼마나 좋아졌는지 나타낸다.
- `specialist_fpr_improvement_at_target_recall`: 딥페이크 95% 이상을 잡는 조건에서 실제 영상을 잘못 경고하는 비율이 얼마나 줄었는지 나타낸다.
- 어느 결과든 외부 실제 웹 영상과 여러 seed 검증이 남아 있어 운영 승인이 아니다.
"""
    ),
]


def build_notebook() -> dict[str, object]:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"name": OUTPUT.name, "is_private": True},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
