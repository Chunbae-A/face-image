#!/usr/bin/env python3
"""Generate two private Kaggle notebooks for the Celeb-DF deepfake baseline."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_OUTPUT = ROOT / "notebooks" / "celebdf_deepfake_preprocess_kaggle.ipynb"
TRAIN_OUTPUT = ROOT / "notebooks" / "celebdf_deepfake_train_kaggle.ipynb"


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


PREPROCESS_CELLS = [
    markdown(
        """
# 1단계 — Celeb-DF 전체 얼굴 전처리 (Kaggle 무료 GPU)

이 노트북은 전체 6,529개 영상에서 최대 32개 얼굴 프레임을 탐지·정렬한다. 모델 학습은 하지 않는다. 완료 후 비공개 얼굴 crop 묶음을 Kaggle Output으로 저장하고, 2단계 학습 노트북이 그 결과를 입력으로 사용한다.

## 실행 전

1. 승인받은 전체 `Celeb-DF-v2.zip`을 **Private Kaggle Dataset**으로 연결한다.
2. Notebook도 `Private` 상태를 유지한다.
3. 오른쪽 `Settings`에서 GPU를 선택하고 Internet을 켠다.
4. 이용 조건 확인값 두 개를 `True`로 바꾼다.

`/kaggle/working/celebdf_deepfake_preprocess_private.tar`에는 얼굴 crop과 비공개 manifest가 포함된다. 절대 공개하거나 GitHub에 올리지 않는다.
"""
    ),
    code(
        """
# 1. 설정과 이용 조건 확인
REPO_URL = "https://github.com/Chunbae-A/face-image.git"
BRANCH = "exp/15-celebdf-deepfake-baseline"
CODE_SOURCE = "embedded"  # "embedded" 권장

SOURCE_ZIP_PATH = ""  # 비우면 /kaggle/input에서 자동 탐색
EXPECTED_SOURCE_ZIP_BYTES = 9952957051
EXPECTED_VIDEO_BYTES = 10156083187
I_CONFIRM_CELEBDF_KAGGLE_PRIVATE_PROCESSING_IS_ALLOWED = False
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = False
RUN_SMOKE = True
RUN_FULL_PREPROCESS = True
SEED = 20260807

import os
from pathlib import Path
import sys

IN_KAGGLE = Path("/kaggle").exists() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if IN_KAGGLE and not I_CONFIRM_CELEBDF_KAGGLE_PRIVATE_PROCESSING_IS_ALLOWED:
    raise PermissionError("Celeb-DF의 Kaggle 비공개 처리가 허용되는지 확인하세요.")
if not I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE:
    raise PermissionError("InsightFace 제공 가중치의 비상업 연구 조건을 확인하세요.")
print({
    "kaggle": IN_KAGGLE,
    "seed": SEED,
    "maximum_face_detections": 6529 * 32,
    "raw_and_crops_must_stay_private": True,
})
"""
    ),
    markdown(
        """
## GPU 환경

Kaggle Settings에서 GPU를 선택한다. 공식 GPU 안내는 P100 무료 사용과 주간 quota를 설명한다. 전처리가 끝나면 `Save Version`으로 Output을 보존한다.
"""
    ),
    code(
        """
# 2. 얼굴 탐지 라이브러리 설치
%pip uninstall -y -q onnxruntime onnxruntime-gpu
%pip install -q --no-cache-dir "insightface==1.0.1" "onnxruntime-gpu==1.23.2" "numpy==2.0.2" "opencv-python-headless==4.12.0.88" "Pillow==12.3.0"
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
# 4. 비공개 Kaggle Input에서 전체 데이터 찾기
import json
import shutil

WORK_ROOT = Path("/kaggle/temp/celebdf_deepfake_preprocess") if IN_KAGGLE else REPO_DIR / "outputs" / "celebdf_deepfake_preprocess"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = WORK_ROOT / "celebdf_private_manifest.csv"
INVENTORY = WORK_ROOT / "inventory_aggregate.json"
VIDEO_ROOT = WORK_ROOT / "videos"
CROP_ROOT = WORK_ROOT / "crops"
CROP_MANIFEST = WORK_ROOT / "crop_private_manifest.csv"
PREPROCESS_REPORT = WORK_ROOT / "preprocess_aggregate.json"
REJECTS = WORK_ROOT / "preprocess_rejects_private.csv"

INPUT_MODE = None
SOURCE_ZIP = None
DATASET_ROOT = None
if IN_KAGGLE:
    zip_candidates = (
        [Path(SOURCE_ZIP_PATH).expanduser()]
        if SOURCE_ZIP_PATH.strip()
        else sorted(Path("/kaggle/input").rglob("Celeb-DF-v2.zip"))
    )
    exact_zips = [
        path for path in zip_candidates
        if path.is_file() and path.stat().st_size == EXPECTED_SOURCE_ZIP_BYTES
    ]
    extracted_roots = []
    for test_list in sorted(Path("/kaggle/input").rglob("List_of_testing_videos.txt")):
        root = test_list.parent
        if all((root / name).is_dir() for name in ("Celeb-real", "YouTube-real", "Celeb-synthesis")):
            video_count = sum(1 for _ in root.glob("*/*.mp4"))
            video_bytes = sum(path.stat().st_size for path in root.glob("*/*.mp4"))
            if video_count == 6529 and video_bytes == EXPECTED_VIDEO_BYTES:
                extracted_roots.append(root)
    if len(exact_zips) == 1:
        INPUT_MODE = "zip"
        SOURCE_ZIP = exact_zips[0]
    elif len(extracted_roots) == 1:
        INPUT_MODE = "kaggle_auto_extracted"
        DATASET_ROOT = extracted_roots[0]
        VIDEO_ROOT = DATASET_ROOT
    else:
        raise FileNotFoundError(
            "정확한 전체 Celeb-DF 입력 하나를 찾지 못했습니다. "
            f"zip={exact_zips}, extracted={extracted_roots}"
        )
else:
    SOURCE_ZIP = Path(SOURCE_ZIP_PATH).expanduser()
    INPUT_MODE = "zip"

print({
    "input_mode": INPUT_MODE,
    "source": str(SOURCE_ZIP or DATASET_ROOT),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
})
"""
    ),
    code(
        """
# 5. GPU와 ONNX Runtime 확인
import onnxruntime as ort

providers = ort.get_available_providers()
print({"onnxruntime": ort.__version__, "providers": providers})
if IN_KAGGLE and "CUDAExecutionProvider" not in providers:
    raise RuntimeError("Kaggle GPU를 선택하고 세션을 다시 시작하세요.")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
))
"""
    ),
    code(
        """
# 6. 전체 목록·라벨·분할 검사
if INPUT_MODE == "zip":
    inventory_command = [
        sys.executable, "scripts/celebdf_deepfake.py", "inventory", str(SOURCE_ZIP),
    ]
else:
    inventory_command = [
        sys.executable, "scripts/celebdf_deepfake.py", "inventory-directory", str(DATASET_ROOT),
    ]
subprocess.run(inventory_command + [
    "--manifest", str(MANIFEST), "--summary", str(INVENTORY),
    "--validation-fraction", "0.15", "--seed", str(SEED),
], check=True)
inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
assert inventory["video_count"] == 6529, inventory
assert inventory["official_test_count"] == 518, inventory
assert inventory["leakage_audit"]["train_validation_video_overlap"] == 0, inventory
assert inventory["leakage_audit"]["train_validation_group_overlap"] == 0, inventory
print({
    "videos": inventory["video_count"],
    "real": inventory["real_video_count"],
    "fake": inventory["fake_video_count"],
    "official_test": inventory["official_test_count"],
    "train_validation_leakage": 0,
})
"""
    ),
    code(
        """
# 7. 분할별 실제/가짜 각 1개 Smoke
if RUN_SMOKE:
    SMOKE_ROOT = WORK_ROOT / "smoke"
    SMOKE_VIDEO_ROOT = VIDEO_ROOT
    if INPUT_MODE == "zip":
        SMOKE_VIDEO_ROOT = SMOKE_ROOT / "videos"
        subprocess.run([
            sys.executable, "scripts/celebdf_deepfake.py", "extract", str(SOURCE_ZIP),
            "--manifest", str(MANIFEST), "--output", str(SMOKE_VIDEO_ROOT),
            "--mode", "smoke", "--smoke-videos-per-class-per-split", "1",
        ], check=True)
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "preprocess",
        "--manifest", str(MANIFEST), "--video-root", str(SMOKE_VIDEO_ROOT),
        "--crop-root", str(SMOKE_ROOT / "crops"),
        "--crop-manifest", str(SMOKE_ROOT / "crops.csv"),
        "--rejects", str(SMOKE_ROOT / "rejects.csv"),
        "--run-report", str(SMOKE_ROOT / "report.json"),
        "--mode", "smoke", "--frames-per-video", "8", "--minimum-valid-frames", "2",
        "--accept-noncommercial-detector-license", "--fail-fast",
    ], check=True)
    smoke = json.loads((SMOKE_ROOT / "report.json").read_text(encoding="utf-8"))
    print({
        "successful_videos": smoke["successful_video_count_total"],
        "face_crops": smoke["crop_count_total"],
        "detector_device": smoke["device"],
    })
"""
    ),
    code(
        """
# 8. 전체 얼굴 전처리
if not RUN_FULL_PREPROCESS:
    raise ValueError("1단계 노트북은 RUN_FULL_PREPROCESS=True로 실행해야 합니다.")
if INPUT_MODE == "zip":
    subprocess.run([
        sys.executable, "scripts/celebdf_deepfake.py", "extract", str(SOURCE_ZIP),
        "--manifest", str(MANIFEST), "--output", str(VIDEO_ROOT), "--mode", "full",
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
preprocess_report = json.loads(PREPROCESS_REPORT.read_text(encoding="utf-8"))
print({
    "successful_videos": preprocess_report["successful_video_count_total"],
    "face_crops": preprocess_report["crop_count_total"],
    "reject_reasons": preprocess_report["reject_reasons_this_run"],
    "p95_seconds_per_video": preprocess_report["preprocess_video_seconds_p95"],
})
"""
    ),
    code(
        """
# 9. 비공개 2단계 입력과 비식별 사전검사 결과 저장
import hashlib
import zipfile

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if IN_KAGGLE:
    PRIVATE_ARCHIVE = Path("/kaggle/working/celebdf_deepfake_preprocess_private.tar")
    SANITIZED_BUNDLE = Path("/kaggle/working/celebdf_deepfake_preflight_sanitized.zip")
else:
    PRIVATE_ARCHIVE = WORK_ROOT / "celebdf_deepfake_preprocess_private.tar"
    SANITIZED_BUNDLE = WORK_ROOT / "celebdf_deepfake_preflight_sanitized.zip"

subprocess.run([
    "tar", "-cf", str(PRIVATE_ARCHIVE), "-C", str(WORK_ROOT),
    CROP_ROOT.name, CROP_MANIFEST.name, MANIFEST.name,
    INVENTORY.name, PREPROCESS_REPORT.name, REJECTS.name,
], check=True)
with zipfile.ZipFile(SANITIZED_BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(INVENTORY, arcname=INVENTORY.name)
    archive.write(PREPROCESS_REPORT, arcname=PREPROCESS_REPORT.name)

print({
    "private_stage2_input": str(PRIVATE_ARCHIVE),
    "private_stage2_input_gb": round(PRIVATE_ARCHIVE.stat().st_size / 1e9, 3),
    "private_stage2_input_sha256": file_sha256(PRIVATE_ARCHIVE),
    "sanitized_preflight": str(SANITIZED_BUNDLE),
    "warning": "private TAR contains aligned faces and must never be published",
})
"""
    ),
    markdown(
        """
## 1단계 완료 후

`Save Version`으로 이 비공개 Notebook의 Output을 저장한다. 다음 학습 Notebook의 `Add Input`에서 이 Notebook Output을 연결한다. Private TAR는 얼굴 데이터를 포함하므로 공개 Dataset으로 만들지 않는다.
"""
    ),
]


TRAIN_CELLS = [
    markdown(
        """
# 2단계 — EfficientNet-B4 학습·공식 Test·ONNX (Kaggle 무료 GPU)

1단계 비공개 Notebook Output의 얼굴 crop으로 모델을 학습한다. Validation에서 프레임 수·통합 방식·기준값을 먼저 고정하고, 그 뒤 공식 Test와 촬영 열화 조건을 평가한다.

## 실행 전

1. 이 Notebook을 `Private`로 유지한다.
2. `Add Input`에서 1단계 Notebook Output을 연결한다.
3. GPU와 Internet을 켠다. Internet은 ImageNet 사전학습 가중치 다운로드에 필요하다.
4. 확인값을 `True`로 변경한다.
"""
    ),
    code(
        """
# 1. 설정과 비공개 입력 확인
REPO_URL = "https://github.com/Chunbae-A/face-image.git"
BRANCH = "exp/15-celebdf-deepfake-baseline"
CODE_SOURCE = "embedded"
PREPROCESS_ARCHIVE_PATH = ""  # 비우면 /kaggle/input에서 자동 탐색
I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED = False
RUN_TRAINING = True
RUN_FINAL_OFFICIAL_TEST = True
SEED = 20260807
EPOCHS = 8
BATCH_SIZE = 8

import os
from pathlib import Path
import sys
IN_KAGGLE = Path("/kaggle").exists() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if not I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED:
    raise PermissionError("1단계 얼굴 crop Output을 비공개 연구에 사용할 수 있는지 확인하세요.")
print({"kaggle": IN_KAGGLE, "seed": SEED, "epochs": EPOCHS})
"""
    ),
    code(
        """
# 2. ONNX 내보내기·CPU 시험 의존성
# Kaggle 기본 torchvision은 Pillow 12.x와 충돌할 수 있으므로 호환 버전을 고정한다.
%pip install -q --no-cache-dir "onnx==1.18.0" "onnxruntime==1.23.2" "Pillow==11.3.0"
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
# 4. 1단계 비공개 Output 복원
import json
import shutil

if IN_KAGGLE:
    candidates = (
        [Path(PREPROCESS_ARCHIVE_PATH).expanduser()]
        if PREPROCESS_ARCHIVE_PATH.strip()
        else sorted(Path("/kaggle/input").rglob("celebdf_deepfake_preprocess_private.tar"))
    )
else:
    candidates = [Path(PREPROCESS_ARCHIVE_PATH).expanduser()]
exact = [path for path in candidates if path.is_file()]
if len(exact) != 1:
    raise FileNotFoundError(f"1단계 private TAR 하나가 필요합니다: {exact}")

WORK_ROOT = Path("/kaggle/temp/celebdf_deepfake_train") if IN_KAGGLE else REPO_DIR / "outputs" / "celebdf_deepfake_train"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "-xf", str(exact[0]), "-C", str(WORK_ROOT)], check=True)
CROP_ROOT = WORK_ROOT / "crops"
CROP_MANIFEST = WORK_ROOT / "crop_private_manifest.csv"
INVENTORY = WORK_ROOT / "inventory_aggregate.json"
PREPROCESS_REPORT = WORK_ROOT / "preprocess_aggregate.json"
if not CROP_ROOT.is_dir() or not CROP_MANIFEST.is_file():
    raise RuntimeError("1단계 얼굴 crop 또는 manifest 복원에 실패했습니다.")

OUTPUT_ROOT = Path("/kaggle/working") if IN_KAGGLE else WORK_ROOT / "output"
# Kaggle에서는 학습 직후 체크포인트를 결과 영역에 저장한다. 이후 ONNX 변환이나
# 결과 포장이 실패해도 세션 결과에서 모델을 복구해 재학습을 피할 수 있다.
PRIVATE_MODEL_ROOT = (OUTPUT_ROOT if IN_KAGGLE else WORK_ROOT) / "private_model"
SANITIZED_ROOT = OUTPUT_ROOT / "sanitized"
PRIVATE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
SANITIZED_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = PRIVATE_MODEL_ROOT / "efficientnet_b4_best.pt"
ONNX_MODEL = PRIVATE_MODEL_ROOT / "efficientnet_b4.onnx"
TRAIN_REPORT = SANITIZED_ROOT / "train_aggregate.json"
METRICS = SANITIZED_ROOT / "aggregate_metrics.json"
ONNX_EXPORT_REPORT = SANITIZED_ROOT / "onnx_export.json"
ONNX_SMOKE_REPORT = SANITIZED_ROOT / "onnx_cpu_smoke.json"
PRIVATE_SCORES = WORK_ROOT / "frame_scores_private.csv"
FIGURE = SANITIZED_ROOT / "deepfake_baseline_summary.png"
MODEL_CARD = SANITIZED_ROOT / "MODEL_CARD.md"

print({
    "private_stage1_input": str(exact[0]),
    "crop_files": sum(1 for _ in CROP_ROOT.rglob("*.jpg")),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
})
"""
    ),
    code(
        """
# 5. PyTorch GPU 확인
import torch
import torchvision

cuda_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
cuda_capability = torch.cuda.get_device_capability(0) if cuda_available else None
compiled_arches = torch.cuda.get_arch_list() if cuda_available else []
required_arch = (
    f"sm_{cuda_capability[0]}{cuda_capability[1]}" if cuda_capability else None
)
print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": cuda_available,
    "gpu": gpu_name,
    "cuda_capability": cuda_capability,
    "compiled_arches": compiled_arches,
})
if IN_KAGGLE and not cuda_available:
    raise RuntimeError("Kaggle Settings에서 GPU를 선택하고 다시 시작하세요.")
if IN_KAGGLE and compiled_arches and required_arch not in compiled_arches:
    raise RuntimeError(
        f"현재 GPU({gpu_name}, {required_arch})는 설치된 PyTorch가 지원하지 않습니다. "
        "Kaggle Settings에서 GPU T4 x2를 선택하세요."
    )
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
))
"""
    ),
    code(
        """
# 6. EfficientNet-B4 학습 — 공식 Test는 사용하지 않음
if not RUN_TRAINING:
    raise ValueError("완료 checkpoint 입력을 별도로 연결하지 않았으므로 RUN_TRAINING=True가 필요합니다.")
subprocess.run([
    sys.executable, "scripts/run_celebdf_deepfake.py", "train",
    "--crop-manifest", str(CROP_MANIFEST), "--crop-root", str(CROP_ROOT),
    "--checkpoint", str(CHECKPOINT), "--train-report", str(TRAIN_REPORT),
    "--input-size", "380", "--train-frames-per-video", "16",
    "--batch-size", str(BATCH_SIZE), "--gradient-accumulation-steps", "2",
    "--epochs", str(EPOCHS), "--early-stopping-patience", "3",
    "--seed", str(SEED), "--require-cuda",
], check=True)
train_report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
assert train_report["official_test_used_for_training"] is False
print({
    "epochs_completed": train_report["epochs_completed"],
    "best_validation_video_auc": train_report["best_validation_video_auc"],
    "checkpoint_sha256": train_report["checkpoint_sha256"],
})
"""
    ),
    code(
        """
# 7. Validation 선택을 고정한 뒤 공식 Test·열화 평가
if not RUN_FINAL_OFFICIAL_TEST:
    raise ValueError("공식 Test 실행을 승인할 때 RUN_FINAL_OFFICIAL_TEST=True로 바꾸세요.")
subprocess.run([
    sys.executable, "scripts/run_celebdf_deepfake.py", "evaluate",
    "--crop-manifest", str(CROP_MANIFEST), "--crop-root", str(CROP_ROOT),
    "--checkpoint", str(CHECKPOINT), "--private-scores", str(PRIVATE_SCORES),
    "--metrics", str(METRICS), "--input-size", "380", "--batch-size", "16",
    "--seed", str(SEED), "--target-fpr", "0.01",
    "--frame-counts", "8", "16", "32",
    "--conditions", "clean", "jpeg_q30", "gaussian_blur_sigma2", "low_light_gamma2", "downscale_0_25",
], check=True)
metrics = json.loads(METRICS.read_text(encoding="utf-8"))
print({
    "selected_frames": metrics["selected_frames_per_video"],
    "selected_aggregation": metrics["selected_aggregation"],
    "threshold": metrics["selected_threshold"],
    "official_test_auc": metrics["test_video"]["roc_auc"],
    "official_test_fpr": metrics["test_video"]["fpr"],
    "official_test_recall": metrics["test_video"]["recall"],
    "coverage": metrics["coverage"]["official_test_coverage"],
    "research_gate_pass": metrics["research_gate"]["overall_pass"],
})
"""
    ),
    code(
        """
# 8. ONNX 내보내기와 CPU 추론 연결 시험
subprocess.run([
    sys.executable, "scripts/run_celebdf_deepfake.py", "export-onnx",
    "--checkpoint", str(CHECKPOINT), "--output", str(ONNX_MODEL),
    "--report", str(ONNX_EXPORT_REPORT),
], check=True)
subprocess.run([
    sys.executable, "scripts/run_celebdf_deepfake.py", "smoke-onnx",
    "--model", str(ONNX_MODEL), "--crop-manifest", str(CROP_MANIFEST),
    "--crop-root", str(CROP_ROOT), "--report", str(ONNX_SMOKE_REPORT),
    "--export-report", str(ONNX_EXPORT_REPORT),
], check=True)
onnx_smoke = json.loads(ONNX_SMOKE_REPORT.read_text(encoding="utf-8"))
print({
    "onnx_cpu_status": onnx_smoke["status"],
    "processing_ms": onnx_smoke["processing_ms"],
    "model_sha256": onnx_smoke["model_sha256"],
})
"""
    ),
    code(
        """
# 9. ROC·PR·혼동행렬·열화 결과 그래프와 모델 카드
import matplotlib.pyplot as plt
import numpy as np

curves = metrics["test_video_curves"]
test = metrics["test_video"]
conditions = metrics["condition_test"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(curves["roc_fpr"], curves["roc_tpr"], label=f"AUC={test['roc_auc']:.4f}")
axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
axes[0, 0].set(title="Official Test ROC", xlabel="FPR", ylabel="TPR")
axes[0, 0].legend()

axes[0, 1].plot(curves["pr_recall"], curves["pr_precision"], label=f"AP={test['average_precision']:.4f}")
axes[0, 1].set(title="Official Test Precision-Recall", xlabel="Recall", ylabel="Precision")
axes[0, 1].legend()

matrix = np.asarray([
    [test["true_negative"], test["false_positive"]],
    [test["false_negative"], test["true_positive"]],
])
axes[1, 0].imshow(matrix, cmap="Blues")
for row in range(2):
    for column in range(2):
        axes[1, 0].text(column, row, str(matrix[row, column]), ha="center", va="center")
axes[1, 0].set(title="Video Confusion Matrix", xlabel="Predicted", ylabel="Actual")
axes[1, 0].set_xticks([0, 1], ["real", "fake"])
axes[1, 0].set_yticks([0, 1], ["real", "fake"])

names = list(conditions)
auc_values = [conditions[name]["video"]["roc_auc"] for name in names]
fpr_values = [conditions[name]["video"]["fpr"] for name in names]
x = np.arange(len(names))
axes[1, 1].bar(x - 0.18, auc_values, width=0.36, label="ROC-AUC")
axes[1, 1].bar(x + 0.18, fpr_values, width=0.36, label="FPR")
axes[1, 1].set_xticks(x, names, rotation=25, ha="right")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title("촬영 열화 조건")
axes[1, 1].legend()

fig.tight_layout()
fig.savefig(FIGURE, dpi=160, bbox_inches="tight")
plt.show()

MODEL_CARD.write_text(f'''# EfficientNet-B4 Celeb-DF 연구 모델 카드

- 상태: 연구용 기준선, 운영 미승인
- 입력: 정렬 얼굴 RGB 380×380
- 출력: fake logit; sigmoid 적용값이 딥페이크 점수
- 선택 프레임: {metrics['selected_frames_per_video']}
- 점수 통합: {metrics['selected_aggregation']}
- 연구 기준값: {metrics['selected_threshold']}
- 공식 Test Video ROC-AUC: {test['roc_auc']}
- 공식 Test 실제영상 FPR: {test['fpr']}
- 공식 Test 딥페이크 Recall: {test['recall']}
- 공식 Test 커버리지: {metrics['coverage']['official_test_coverage']}
- Gate 통과: {metrics['research_gate']['overall_pass']}

이 모델은 Celeb-DF와 비상업 연구 조건의 InsightFace 얼굴 검출 전처리를 사용했다. 한국인·최신 생성 방식·실제 웹 검색 영상에 대한 운영 성능과 상용 사용권을 증명하지 않는다. 사람 검토 없이 차단·삭제를 자동 실행하지 않는다.
''', encoding="utf-8")
print({"figure": str(FIGURE), "model_card": str(MODEL_CARD)})
"""
    ),
    code(
        """
# 10. 비식별 결과와 비공개 API 모델 묶음 저장
import hashlib
import shutil
import zipfile

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

SANITIZED_BUNDLE = OUTPUT_ROOT / "celebdf_deepfake_sanitized_results.zip"
PRIVATE_MODEL_BUNDLE = OUTPUT_ROOT / "celebdf_deepfake_private_model.zip"
sanitized_files = (
    INVENTORY,
    PREPROCESS_REPORT,
    TRAIN_REPORT,
    METRICS,
    ONNX_EXPORT_REPORT,
    ONNX_SMOKE_REPORT,
    FIGURE,
    MODEL_CARD,
)
with zipfile.ZipFile(SANITIZED_BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sanitized_files:
        archive.write(path, arcname=path.name)
with zipfile.ZipFile(PRIVATE_MODEL_BUNDLE, "w", compression=zipfile.ZIP_STORED) as archive:
    archive.write(CHECKPOINT, arcname=CHECKPOINT.name)
    archive.write(ONNX_MODEL, arcname=ONNX_MODEL.name)
    archive.write(MODEL_CARD, arcname=MODEL_CARD.name)

with zipfile.ZipFile(SANITIZED_BUNDLE) as archive:
    names = set(archive.namelist())
    assert PRIVATE_SCORES.name not in names
    assert CROP_MANIFEST.name not in names
    assert not any(name.endswith((".jpg", ".mp4", ".pt", ".onnx")) for name in names)

# ZIP 검증까지 성공한 경우에만 중복 원본을 지운다. 중간 단계가 실패하면
# /kaggle/working/private_model이 남아 체크포인트를 복구할 수 있다.
if IN_KAGGLE:
    shutil.rmtree(PRIVATE_MODEL_ROOT)

print({
    "sanitized_results": str(SANITIZED_BUNDLE),
    "sanitized_sha256": file_sha256(SANITIZED_BUNDLE),
    "private_api_model": str(PRIVATE_MODEL_BUNDLE),
    "private_model_sha256": file_sha256(PRIVATE_MODEL_BUNDLE),
    "raw_faces_in_sanitized_bundle": False,
    "warning": "private model bundle must not be published or committed to GitHub",
})
"""
    ),
    markdown(
        """
## 결과 판단

- `official_test_auc ≥ 0.90`
- `official_test_fpr ≤ 0.01`
- 내부 Train/Validation 누수 0건

통과하더라도 연구 기준선이다. 비식별 결과 ZIP만 GitHub Issue/PR에 첨부하고, 얼굴 crop·frame score·checkpoint·ONNX는 비공개로 유지한다.
"""
    ),
]


def build_notebook(cells: list[dict[str, object]], name: str) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"name": name, "is_private": True},
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


def build_preprocess_notebook() -> dict[str, object]:
    return build_notebook(PREPROCESS_CELLS, PREPROCESS_OUTPUT.name)


def build_train_notebook() -> dict[str, object]:
    return build_notebook(TRAIN_CELLS, TRAIN_OUTPUT.name)


def main() -> int:
    for path, notebook in (
        (PREPROCESS_OUTPUT, build_preprocess_notebook()),
        (TRAIN_OUTPUT, build_train_notebook()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
