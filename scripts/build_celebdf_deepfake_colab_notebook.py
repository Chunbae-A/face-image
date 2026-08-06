#!/usr/bin/env python3
"""Generate the Celeb-DF-v2 EfficientNet-B4 Colab notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "celebdf_efficientnet_b4_colab.ipynb"


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
    embedded_paths = [
        ROOT / "scripts" / "celebdf_deepfake.py",
        ROOT / "scripts" / "run_celebdf_deepfake.py",
    ]
    embedded = {
        str(path.relative_to(ROOT)): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in embedded_paths
    }
    fingerprint = hashlib.sha256(
        b"".join(path.read_bytes() for path in embedded_paths)
    ).hexdigest()
    source = r'''#@title 3. 실행 코드 준비
from pathlib import Path
import base64
import os
import subprocess

EMBEDDED_FILES_B64 = ''' + repr(embedded) + r'''
EMBEDDED_CODE_SHA256 = "''' + fingerprint + r'''"

if IN_HOSTED_COLAB and CODE_SOURCE == "github":
    REPO_DIR = Path("/content/face-image")
    if not REPO_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
    CODE_VERSION = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
elif IN_HOSTED_COLAB:
    REPO_DIR = Path("/content/face-image")
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
# Celeb-DF-v2 전체 딥페이크 판별 — EfficientNet-B4 기준선

이 노트북은 기존 **같은 사람 비교(ArcFace)**가 아니라, 영상 속 얼굴이 **실제인지 딥페이크인지** 판별하는 별도 모델을 학습한다.

- 전체 영상: 6,529개 (`실제 890`, `딥페이크 5,639`)
- 공식 Test: 518개를 학습·설정 선택에서 잠금
- 모델: ImageNet 사전학습 EfficientNet-B4, 입력 380×380
- 검증 비교: 영상당 8/16/32프레임, 평균/중앙값/상위 25% 평균
- 최종 수치: Video ROC-AUC, FPR/FNR, Recall, F1, AP, EER, p50/p95
- 열화 평가: JPEG, 흐림, 저조도, 해상도 축소

코드 준비와 전체 모델 실행은 다른 단계다. 모든 셀을 끝내기 전에는 모델 정확도가 확인됐다고 말하지 않는다.
"""
    ),
    code(
        """
#@title 1. 실행 설정과 이용 조건 확인
REPO_URL = "https://github.com/Chunbae-A/face-image.git" #@param {type:"string"}
BRANCH = "exp/15-celebdf-deepfake-baseline" #@param {type:"string"}
CODE_SOURCE = "embedded" #@param ["embedded", "github"]

SOURCE_ZIP_PATH = "/content/drive/MyDrive/Celeb-DF-v2.zip" #@param {type:"string"}
EXPECTED_SOURCE_ZIP_BYTES = 9952957051 #@param {type:"integer"}
DRIVE_PRIVATE_ROOT = "/content/drive/MyDrive/face-image-deepfake-private" #@param {type:"string"}
PERSIST_CROP_CACHE_TO_DRIVE = True #@param {type:"boolean"}

# 공식 신청·승인 파일이며 약관상 Colab/Drive 처리가 허용되는지 확인한 경우에만 True
I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED = False #@param {type:"boolean"}
# InsightFace 제공 검출 가중치의 비상업 연구 조건을 확인한 경우에만 True
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = False #@param {type:"boolean"}

RUN_PREPROCESS_SMOKE = True #@param {type:"boolean"}
RUN_FULL_PREPROCESS = True #@param {type:"boolean"}
RUN_TRAINING = True #@param {type:"boolean"}
RUN_FINAL_OFFICIAL_TEST = True #@param {type:"boolean"}
ALLOW_REPEAT_OFFICIAL_TEST = False #@param {type:"boolean"}
SEED = 20260807 #@param {type:"integer"}
EPOCHS = 8 #@param {type:"integer"}
BATCH_SIZE = 8 #@param {type:"integer"}

import sys
IN_HOSTED_COLAB = "google.colab" in sys.modules
if IN_HOSTED_COLAB and not I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED:
    raise PermissionError("Celeb-DF의 Colab/Drive 처리가 허용되는지 확인한 뒤 설정을 True로 바꾸세요.")
if not I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE:
    raise PermissionError("InsightFace 제공 가중치의 비상업 연구 조건을 확인한 뒤 설정을 True로 바꾸세요.")
if EXPECTED_SOURCE_ZIP_BYTES <= 0:
    raise ValueError("원본 ZIP의 정확한 바이트 크기가 필요합니다.")

print({
    "hosted_colab": IN_HOSTED_COLAB,
    "seed": SEED,
    "epochs": EPOCHS,
    "maximum_face_detections": 6529 * 32,
    "official_test_locked": True,
})
"""
    ),
    markdown(
        """
## GPU 환경

Colab 메뉴에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택한다. 설치 후 런타임을 재시작했다면 1번 셀부터 다시 실행하되 설치 셀은 다시 실행하지 않는다.
"""
    ),
    code(
        """
#@title 2. 라이브러리 설치
%pip uninstall -y -q onnxruntime onnxruntime-gpu
%pip install -q --no-cache-dir "insightface==1.0.1" "onnxruntime-gpu==1.23.2" "onnx==1.18.0" "numpy==2.0.2" "opencv-python-headless==4.12.0.88" "Pillow==12.3.0"
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
#@title 4. Drive 연결, 원본 확인, 경로 준비
import json
import shutil

if IN_HOSTED_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

SOURCE_ZIP = Path(SOURCE_ZIP_PATH).expanduser()
if not SOURCE_ZIP.exists():
    raise FileNotFoundError(f"Drive에서 Celeb-DF-v2.zip을 찾지 못했습니다: {SOURCE_ZIP}")
if SOURCE_ZIP.stat().st_size != EXPECTED_SOURCE_ZIP_BYTES:
    raise IOError(
        f"ZIP 크기가 다릅니다: {SOURCE_ZIP.stat().st_size} != {EXPECTED_SOURCE_ZIP_BYTES}"
    )

WORK_ROOT = Path("/content/celebdf_deepfake") if IN_HOSTED_COLAB else REPO_DIR / "outputs" / "celebdf_deepfake"
VIDEO_ROOT = WORK_ROOT / "videos"
CROP_ROOT = WORK_ROOT / "crops"
MANIFEST = WORK_ROOT / "celebdf_private_manifest.csv"
INVENTORY = WORK_ROOT / "inventory_aggregate.json"
CROP_MANIFEST = WORK_ROOT / "crop_private_manifest.csv"
PREPROCESS_REPORT = WORK_ROOT / "preprocess_aggregate.json"
REJECTS = WORK_ROOT / "preprocess_rejects_private.csv"

DRIVE_ROOT = Path(DRIVE_PRIVATE_ROOT)
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
CROP_CACHE_ARCHIVE = DRIVE_ROOT / "celebdf_aligned_crops.tar"
CHECKPOINT = DRIVE_ROOT / "efficientnet_b4_best.pt"
TRAIN_REPORT = DRIVE_ROOT / "train_aggregate.json"
PRIVATE_SCORES = DRIVE_ROOT / "frame_scores_private.csv"
METRICS = DRIVE_ROOT / "aggregate_metrics.json"
ONNX_MODEL = DRIVE_ROOT / "efficientnet_b4.onnx"
ONNX_EXPORT_REPORT = DRIVE_ROOT / "onnx_export.json"
ONNX_SMOKE_REPORT = DRIVE_ROOT / "onnx_cpu_smoke.json"

WORK_ROOT.mkdir(parents=True, exist_ok=True)
print({
    "zip_gb": round(SOURCE_ZIP.stat().st_size / 1e9, 3),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
    "private_drive_root": str(DRIVE_ROOT),
    "crop_cache_exists": CROP_CACHE_ARCHIVE.exists(),
})
"""
    ),
    code(
        """
#@title 5. GPU, PyTorch, ONNX Runtime 확인
import subprocess
import torch
import torchvision
import onnxruntime as ort

providers = ort.get_available_providers()
print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.cuda.is_available(),
    "onnxruntime": ort.__version__,
    "providers": providers,
})
if IN_HOSTED_COLAB and not torch.cuda.is_available():
    raise RuntimeError("PyTorch에서 GPU를 찾지 못했습니다. T4 GPU 런타임으로 다시 연결하세요.")
if IN_HOSTED_COLAB and "CUDAExecutionProvider" not in providers:
    raise RuntimeError("얼굴 검출용 CUDAExecutionProvider가 없습니다. 설치 후 런타임을 재시작하세요.")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
))
"""
    ),
    code(
        """
#@title 6. 전체 목록 검사와 누수 없는 분할
subprocess.run([
    sys.executable, "scripts/celebdf_deepfake.py", "inventory", str(SOURCE_ZIP),
    "--manifest", str(MANIFEST), "--summary", str(INVENTORY),
    "--validation-fraction", "0.15", "--seed", str(SEED),
], check=True)
inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
assert inventory["video_count"] == 6529, inventory
assert inventory["official_test_count"] == 518, inventory
assert inventory["leakage_audit"]["train_validation_video_overlap"] == 0, inventory
assert inventory["leakage_audit"]["train_validation_group_overlap"] == 0, inventory
assert inventory["leakage_audit"]["official_test_outside_test_split"] == 0, inventory
print({
    "전체 영상": inventory["video_count"],
    "실제": inventory["real_video_count"],
    "딥페이크": inventory["fake_video_count"],
    "공식 Test": inventory["official_test_count"],
    "내부 Train/Validation 누수": 0,
})
"""
    ),
    code(
        """
#@title 7. 얼굴 전처리 Smoke — 분할별 실제/가짜 각 1개
if RUN_PREPROCESS_SMOKE:
    SMOKE_ROOT = WORK_ROOT / "smoke"
    subprocess.run([
        sys.executable, "scripts/celebdf_deepfake.py", "extract", str(SOURCE_ZIP),
        "--manifest", str(MANIFEST), "--output", str(VIDEO_ROOT), "--split", "all",
        "--mode", "smoke", "--smoke-videos-per-class-per-split", "1",
    ], check=True)
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "preprocess",
        "--manifest", str(MANIFEST), "--video-root", str(VIDEO_ROOT),
        "--crop-root", str(SMOKE_ROOT / "crops"),
        "--crop-manifest", str(SMOKE_ROOT / "crops.csv"),
        "--rejects", str(SMOKE_ROOT / "rejects.csv"),
        "--run-report", str(SMOKE_ROOT / "report.json"),
        "--mode", "smoke", "--frames-per-video", "8", "--minimum-valid-frames", "2",
        "--accept-noncommercial-detector-license", "--fail-fast",
    ], check=True)
    smoke = json.loads((SMOKE_ROOT / "report.json").read_text(encoding="utf-8"))
    print({
        "smoke_videos": smoke["successful_video_count_total"],
        "smoke_crops": smoke["crop_count_total"],
        "detector_device": smoke["device"],
    })
else:
    print("Smoke 전처리를 건너뛰었습니다.")
"""
    ),
    code(
        """
#@title 8. 전체 6,529개 영상 얼굴 전처리 또는 Drive 캐시 복원
if CROP_CACHE_ARCHIVE.exists():
    print("Drive의 전처리 캐시를 복원합니다.")
    subprocess.run(["tar", "-xf", str(CROP_CACHE_ARCHIVE), "-C", str(WORK_ROOT)], check=True)
elif RUN_FULL_PREPROCESS:
    if not VIDEO_ROOT.exists() or len(list(VIDEO_ROOT.rglob("*.mp4"))) != 6529:
        subprocess.run([
            sys.executable, "scripts/celebdf_deepfake.py", "extract", str(SOURCE_ZIP),
            "--manifest", str(MANIFEST), "--output", str(VIDEO_ROOT), "--split", "all",
        ], check=True)
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "preprocess",
        "--manifest", str(MANIFEST), "--video-root", str(VIDEO_ROOT),
        "--crop-root", str(CROP_ROOT), "--crop-manifest", str(CROP_MANIFEST),
        "--rejects", str(REJECTS), "--run-report", str(PREPROCESS_REPORT),
        "--mode", "full", "--frames-per-video", "32", "--minimum-valid-frames", "4",
        "--checkpoint-every-videos", "25", "--progress-every", "25",
        "--accept-noncommercial-detector-license",
    ], check=True)
    if PERSIST_CROP_CACHE_TO_DRIVE:
        local_archive = WORK_ROOT / "celebdf_aligned_crops.tar"
        subprocess.run([
            "tar", "-cf", str(local_archive), "-C", str(WORK_ROOT),
            CROP_ROOT.name, CROP_MANIFEST.name, PREPROCESS_REPORT.name, REJECTS.name,
        ], check=True)
        copying = CROP_CACHE_ARCHIVE.with_suffix(".tar.copying")
        shutil.copyfile(local_archive, copying)
        copying.replace(CROP_CACHE_ARCHIVE)
        print({"drive_crop_cache_gb": round(CROP_CACHE_ARCHIVE.stat().st_size / 1e9, 3)})
else:
    raise FileNotFoundError("전체 crop 캐시가 없고 RUN_FULL_PREPROCESS=False입니다.")

if not CROP_MANIFEST.exists() or not CROP_ROOT.exists():
    raise RuntimeError("전체 얼굴 crop 캐시 복원 또는 생성에 실패했습니다.")
print({
    "crop_manifest_mb": round(CROP_MANIFEST.stat().st_size / 1e6, 2),
    "crop_files": sum(1 for _ in CROP_ROOT.rglob("*.jpg")),
})
"""
    ),
    code(
        """
#@title 9. EfficientNet-B4 학습
training_complete = False
if CHECKPOINT.exists() and TRAIN_REPORT.exists():
    previous_train = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    training_complete = previous_train.get("status") == "completed"

if RUN_TRAINING and not training_complete:
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "train",
        "--crop-manifest", str(CROP_MANIFEST), "--crop-root", str(CROP_ROOT),
        "--checkpoint", str(CHECKPOINT), "--train-report", str(TRAIN_REPORT),
        "--input-size", "380", "--train-frames-per-video", "16",
        "--batch-size", str(BATCH_SIZE), "--gradient-accumulation-steps", "2",
        "--epochs", str(EPOCHS), "--early-stopping-patience", "3",
        "--seed", str(SEED), "--require-cuda",
    ], check=True)
elif training_complete:
    print("완료된 Drive checkpoint를 재사용합니다.")
else:
    raise FileNotFoundError("완료된 checkpoint가 없고 RUN_TRAINING=False입니다.")

train_report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
print({
    "epochs_completed": train_report["epochs_completed"],
    "best_validation_video_auc": train_report["best_validation_video_auc"],
    "checkpoint_sha256": train_report["checkpoint_sha256"],
})
"""
    ),
    code(
        """
#@title 10. Validation 선택 후 공식 Test·열화 평가
if METRICS.exists() and not ALLOW_REPEAT_OFFICIAL_TEST:
    print("기존 공식 Test 결과가 있어 반복 실행하지 않습니다.")
elif RUN_FINAL_OFFICIAL_TEST:
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "evaluate",
        "--crop-manifest", str(CROP_MANIFEST), "--crop-root", str(CROP_ROOT),
        "--checkpoint", str(CHECKPOINT), "--private-scores", str(PRIVATE_SCORES),
        "--metrics", str(METRICS), "--input-size", "380", "--batch-size", "16",
        "--seed", str(SEED), "--target-fpr", "0.01",
        "--frame-counts", "8", "16", "32",
        "--conditions", "clean", "jpeg_q30", "gaussian_blur_sigma2", "low_light_gamma2", "downscale_0_25",
    ], check=True)
else:
    raise FileNotFoundError("공식 Test 결과가 없고 RUN_FINAL_OFFICIAL_TEST=False입니다.")

metrics = json.loads(METRICS.read_text(encoding="utf-8"))
print({
    "selected_frames_per_video": metrics["selected_frames_per_video"],
    "selected_aggregation": metrics["selected_aggregation"],
    "selected_threshold": metrics["selected_threshold"],
    "official_test_video_auc": metrics["test_video"]["roc_auc"],
    "official_test_real_fpr": metrics["test_video"]["fpr"],
    "official_test_fake_recall": metrics["test_video"]["recall"],
    "coverage": metrics["coverage"]["official_test_coverage"],
    "research_gate_pass": metrics["research_gate"]["overall_pass"],
})
"""
    ),
    code(
        """
#@title 11. API 연결용 ONNX 내보내기와 CPU 스모크
subprocess.run([
    sys.executable, "scripts/run_celebdf_deepfake.py", "export-onnx",
    "--checkpoint", str(CHECKPOINT), "--output", str(ONNX_MODEL),
    "--report", str(ONNX_EXPORT_REPORT),
], check=True)
subprocess.run([
    sys.executable, "scripts/run_celebdf_deepfake.py", "smoke-onnx",
    "--model", str(ONNX_MODEL), "--crop-manifest", str(CROP_MANIFEST),
    "--crop-root", str(CROP_ROOT), "--report", str(ONNX_SMOKE_REPORT),
    "--input-size", "380",
], check=True)
smoke = json.loads(ONNX_SMOKE_REPORT.read_text(encoding="utf-8"))
print({
    "onnx_cpu_status": smoke["status"],
    "provider": smoke["provider"],
    "processing_ms": smoke["processing_ms"],
    "model_sha256": smoke["model_sha256"],
})
"""
    ),
    code(
        """
#@title 12. GitHub에 올릴 수 있는 비식별 집계 결과 묶음
import zipfile

SANITIZED_ROOT = WORK_ROOT / "sanitized"
SANITIZED_ROOT.mkdir(parents=True, exist_ok=True)
public_files = {
    INVENTORY: "inventory_aggregate.json",
    PREPROCESS_REPORT: "preprocess_aggregate.json",
    TRAIN_REPORT: "train_aggregate.json",
    METRICS: "aggregate_metrics.json",
    ONNX_EXPORT_REPORT: "onnx_export.json",
    ONNX_SMOKE_REPORT: "onnx_cpu_smoke.json",
}
for source, name in public_files.items():
    if source.exists():
        shutil.copyfile(source, SANITIZED_ROOT / name)

bundle = WORK_ROOT / "celebdf_deepfake_sanitized_results.zip"
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(SANITIZED_ROOT.glob("*.json")):
        archive.write(path, arcname=path.name)

for forbidden in (PRIVATE_SCORES, CHECKPOINT, ONNX_MODEL, CROP_MANIFEST, REJECTS):
    assert forbidden.name not in {item.name for item in SANITIZED_ROOT.iterdir()}

print({
    "download_bundle": str(bundle),
    "files": sorted(path.name for path in SANITIZED_ROOT.iterdir()),
    "excluded": ["원본 영상", "얼굴 crop", "영상/인물 ID", "frame score", "checkpoint", "ONNX"],
})
if IN_HOSTED_COLAB:
    from google.colab import files
    files.download(str(bundle))
"""
    ),
    markdown(
        """
## 완료 판단

마지막 출력의 `research_gate_pass`가 참인지와 별개로 결과를 그대로 보고한다.

- AUC 0.90 미만이면 판별력이 부족하다.
- 실제 영상 FPR 1% 초과면 즉시경보에 사용하지 않는다.
- 두 기준을 통과해도 Celeb-DF 연구 기준선일 뿐 운영 승인이 아니다.
- 다운로드한 비식별 ZIP만 Issue #15 결과 보고에 사용한다.
"""
    ),
]


def build_notebook() -> dict[str, object]:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
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
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
