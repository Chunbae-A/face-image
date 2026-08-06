#!/usr/bin/env python3
"""Generate the multi-seed Celeb-DF ArcFace audit Colab notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from build_celebdf_colab_notebook import ROOT, code, markdown


OUTPUT = ROOT / "notebooks" / "celebdf_arcface_audit_colab.ipynb"


def repository_bootstrap_cell() -> dict[str, object]:
    embedded_paths = [
        ROOT / "scripts" / "celebdf_faceguard.py",
        ROOT / "scripts" / "run_celebdf_arcface.py",
        ROOT / "scripts" / "audit_celebdf_baseline.py",
    ]
    embedded = {
        str(path.relative_to(ROOT)): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in embedded_paths
    }
    fingerprint = hashlib.sha256(
        b"".join(path.read_bytes() for path in embedded_paths)
    ).hexdigest()
    source = r'''#@title 3. 실행 코드 준비 — 기본값은 PR 코드가 내장된 embedded 모드
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
    try:
        CODE_VERSION = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        CODE_VERSION = f"local:{EMBEDDED_CODE_SHA256[:12]}"

os.chdir(REPO_DIR)
print({"repo": str(REPO_DIR), "code_source": CODE_SOURCE, "code_version": CODE_VERSION})
'''
    return code(source)


CELLS = [
    markdown(
        """
# Celeb-DF ArcFace baseline 재현성·누수 감사

이 노트북은 GitHub Issue #4의 고정 프로토콜을 실행한다.

- `frames/video`: 1, 5, 10
- 등록 영상: 1, 3, 5개
- subject/protocol seed: 5개
- 모든 등록 프로토콜의 query는 공통으로 6번째 영상부터 사용
- validation/test identity와 registration/query video의 교집합을 0으로 검증
- 원본, frame, crop, 개별 score, embedding은 runtime 밖으로 내보내지 않음
- Drive에는 집계 JSON/CSV/PNG와 hash만 포함한 ZIP만 저장

이 결과는 **Celeb-real 동일인 검증 baseline**이며 딥페이크 탐지 성능이 아니다.
"""
    ),
    code(
        """
#@title 1. 실행 설정과 권한 확인
REPO_URL = "https://github.com/Chunbae-A/face-image.git" #@param {type:"string"}
BRANCH = "exp/4-celebdf-baseline-audit" #@param {type:"string"}
CODE_SOURCE = "embedded" #@param ["embedded", "github"]
SOURCE_ZIP_PATH = "/content/drive/MyDrive/Celeb-DF-v2.zip" #@param {type:"string"}
# Drive/Drive API가 막혀 로컬 ZIP을 /content에 분할 업로드한 경우만 True.
ASSEMBLE_RUNTIME_UPLOAD_PARTS = False #@param {type:"boolean"}
# 0이면 생략. 분할 전 로컬 ZIP의 실제 바이트를 입력하면 결합 후 검증한다.
EXPECTED_SOURCE_ZIP_BYTES = 0 #@param {type:"integer"}
# DriveFS mount가 반복 실패할 때만 Drive 웹의 파일 ID를 입력한다. Git/결과 bundle에는 기록되지 않는다.
DRIVE_SOURCE_FILE_ID = "" #@param {type:"string"}
DRIVE_RESULT_DIR = "/content/drive/MyDrive/face-image-celebdf-audit" #@param {type:"string"}
PERSIST_SANITIZED_RESULTS_TO_DRIVE = True #@param {type:"boolean"}

# 공식 신청·승인 파일이며 약관상 Hosted Colab/Drive 처리가 허용됨을 직접 확인한 경우만 True.
I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED = False #@param {type:"boolean"}
# InsightFace 제공 buffalo_l 가중치는 비상업 연구 전용.
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = False #@param {type:"boolean"}

FRAMES_PER_VIDEO_VALUES = "1,5,10" #@param {type:"string"}
REFERENCE_COUNTS = "1,3,5" #@param {type:"string"}
SEEDS = "20260805,20260806,20260807,20260808,20260809" #@param {type:"string"}
BOOTSTRAP_REPEATS = 500 #@param {type:"integer"}
RUN_SMOKE_BEFORE_FULL = True #@param {type:"boolean"}

import sys
IN_HOSTED_COLAB = "google.colab" in sys.modules
if IN_HOSTED_COLAB and not I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED:
    raise PermissionError("Confirm Celeb-DF Hosted Colab/Drive processing permission first.")
if not I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE:
    raise PermissionError("Accept the InsightFace non-commercial research weight license first.")

FRAME_VALUES = tuple(int(value) for value in FRAMES_PER_VIDEO_VALUES.split(","))
REFERENCE_VALUES = tuple(int(value) for value in REFERENCE_COUNTS.split(","))
SEED_VALUES = tuple(int(value) for value in SEEDS.split(","))
if FRAME_VALUES != (1, 5, 10):
    raise ValueError("Issue #4 protocol requires frames 1,5,10.")
if REFERENCE_VALUES != (1, 3, 5):
    raise ValueError("Issue #4 protocol requires references 1,3,5.")
if len(SEED_VALUES) != 5 or len(set(SEED_VALUES)) != 5:
    raise ValueError("Issue #4 protocol requires five unique seeds.")
print({
    "hosted_colab": IN_HOSTED_COLAB,
    "frames": FRAME_VALUES,
    "references": REFERENCE_VALUES,
    "seeds": SEED_VALUES,
    "maximum_frame_inferences": 590 * sum(FRAME_VALUES),
})
"""
    ),
    markdown(
        """
## 실행 환경

Colab에서 GPU runtime을 선택한다. 설치 셀은 Colab CUDA 사용자 라이브러리와 호환되는 `onnxruntime-gpu==1.23.2`를 고정한다. 설치 후 runtime을 재시작했다면 2번 설치 셀을 건너뛰고 1번과 3번 이후 셀을 다시 실행한다.
"""
    ),
    code(
        """
#@title 2. 라이브러리 설치
%pip uninstall -y -q onnxruntime onnxruntime-gpu
%pip install -q --no-cache-dir "insightface==1.0.1" "onnxruntime-gpu==1.23.2" opencv-python-headless pandas matplotlib seaborn tqdm
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
#@title 4. Drive 연결과 runtime 작업 경로
import json
import shutil

source_zip = Path(SOURCE_ZIP_PATH).expanduser()
source_transport = "configured_path"
runtime_upload_zip = Path("/content/Celeb-DF-v2.zip")
runtime_upload_parts = sorted(Path("/content").glob("Celeb-DF-v2.zip.part-*"))

if IN_HOSTED_COLAB and ASSEMBLE_RUNTIME_UPLOAD_PARTS:
    if not runtime_upload_parts:
        raise FileNotFoundError("No /content/Celeb-DF-v2.zip.part-* uploads were found.")
    expected_joined_bytes = sum(part.stat().st_size for part in runtime_upload_parts)
    temporary_joined_zip = runtime_upload_zip.with_suffix(".zip.joining")
    with temporary_joined_zip.open("wb") as sink:
        for part in runtime_upload_parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, sink, length=64 * 1024 * 1024)
    if temporary_joined_zip.stat().st_size != expected_joined_bytes:
        raise IOError(
            f"runtime upload join size mismatch: "
            f"{temporary_joined_zip.stat().st_size} != {expected_joined_bytes}"
        )
    temporary_joined_zip.replace(runtime_upload_zip)
    source_zip = runtime_upload_zip
    source_transport = "runtime_upload_parts"
elif IN_HOSTED_COLAB and runtime_upload_zip.exists():
    source_zip = runtime_upload_zip
    source_transport = "runtime_upload"
elif IN_HOSTED_COLAB and DRIVE_SOURCE_FILE_ID.strip():
    from google.colab import auth
    from google.auth import default
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    auth.authenticate_user()
    credentials, _ = default()
    drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    metadata = drive_service.files().get(
        fileId=DRIVE_SOURCE_FILE_ID.strip(), fields="id,name,size"
    ).execute()
    expected_size = int(metadata["size"])
    source_zip = Path("/content/Celeb-DF-v2.zip")
    if not source_zip.exists() or source_zip.stat().st_size != expected_size:
        temporary_zip = source_zip.with_suffix(".zip.part")
        request = drive_service.files().get_media(fileId=DRIVE_SOURCE_FILE_ID.strip())
        with temporary_zip.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=64 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print({"drive_api_download_percent": round(status.progress() * 100, 1)})
        if temporary_zip.stat().st_size != expected_size:
            raise IOError(
                f"Drive API download size mismatch: {temporary_zip.stat().st_size} != {expected_size}"
            )
        temporary_zip.replace(source_zip)
    print({"drive_api_source": metadata["name"], "bytes": expected_size})
    source_transport = "drive_api"
elif IN_HOSTED_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    source_transport = "drivefs"

if not source_zip.exists():
    raise FileNotFoundError(f"Celeb-DF ZIP not found: {source_zip}")
if EXPECTED_SOURCE_ZIP_BYTES and source_zip.stat().st_size != EXPECTED_SOURCE_ZIP_BYTES:
    raise IOError(
        f"Celeb-DF ZIP size mismatch: {source_zip.stat().st_size} != {EXPECTED_SOURCE_ZIP_BYTES}"
    )

DATA_ROOT = Path("/content/celebdf_faceguard") if IN_HOSTED_COLAB else REPO_DIR / "outputs" / "celebdf_faceguard"
AUDIT_ROOT = Path("/content/celebdf_baseline_audit") if IN_HOSTED_COLAB else REPO_DIR / "outputs" / "celebdf_baseline_audit"
VIDEO_ROOT = DATA_ROOT / "videos"
MANIFEST = DATA_ROOT / "celeb_real_manifest.csv"
INVENTORY_JSON = DATA_ROOT / "celeb_real_inventory.json"
SANITIZED_ROOT = AUDIT_ROOT / "sanitized"
for path in (DATA_ROOT, AUDIT_ROOT, SANITIZED_ROOT):
    path.mkdir(parents=True, exist_ok=True)

print({
    "source_zip_gb": round(source_zip.stat().st_size / 1e9, 3),
    "source_transport": source_transport,
    "runtime_free_gb": round(shutil.disk_usage(AUDIT_ROOT).free / 1e9, 2),
    "audit_root": str(AUDIT_ROOT),
    "sanitized_drive_dir": DRIVE_RESULT_DIR if PERSIST_SANITIZED_RESULTS_TO_DRIVE else None,
})
"""
    ),
    code(
        """
#@title 5. ZIP inventory와 Celeb-real 590개 추출
subprocess.run([
    sys.executable, "scripts/celebdf_faceguard.py", "inventory", str(source_zip),
    "--manifest", str(MANIFEST), "--summary", str(INVENTORY_JSON),
], check=True)
inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
assert inventory["video_count"] == 590, inventory
assert inventory["subject_count"] == 59, inventory
assert inventory["eligible_subjects_ge_8_videos"] == 56, inventory

subprocess.run([
    sys.executable, "scripts/celebdf_faceguard.py", "extract", str(source_zip),
    "--manifest", str(MANIFEST), "--output", str(VIDEO_ROOT), "--mode", "full",
], check=True)
extracted = sorted((VIDEO_ROOT / "Celeb-real").glob("*.mp4"))
if len(extracted) != 590:
    raise RuntimeError(f"Expected 590 videos, found {len(extracted)}")
print({"videos": len(extracted), "eligible_subjects": 56})
"""
    ),
    code(
        """
#@title 6. GPU/ONNX Runtime 확인
import subprocess
import onnxruntime as ort

providers = ort.get_available_providers()
print({"onnxruntime": ort.__version__, "providers": providers})
if IN_HOSTED_COLAB and "CUDAExecutionProvider" not in providers:
    raise RuntimeError("CUDAExecutionProvider is unavailable. Restart a GPU runtime without rerunning cell 2.")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
))
"""
    ),
    code(
        """
#@title 7. frames 1/5/10 전체 ArcFace 추론
EMBEDDING_RUNS = {}
RUN_REPORTS = {}
REJECT_FILES = {}

for frames in FRAME_VALUES:
    run_root = AUDIT_ROOT / f"frames_{frames}"
    run_root.mkdir(parents=True, exist_ok=True)
    embeddings = run_root / "video_embeddings.npz"
    rejects = run_root / "rejects.csv"
    run_report = run_root / "run.json"
    minimum_valid_frames = min(3, frames)
    command = [
        sys.executable, "scripts/run_celebdf_arcface.py",
        "--manifest", str(MANIFEST),
        "--video-root", str(VIDEO_ROOT),
        "--output", str(embeddings),
        "--rejects", str(rejects),
        "--run-report", str(run_report),
        "--frames-per-video", str(frames),
        "--minimum-valid-frames", str(minimum_valid_frames),
        "--checkpoint-every", "25",
        "--progress-every", "25",
        "--model-name", "buffalo_l",
        "--accept-noncommercial-model-license",
    ]
    if RUN_SMOKE_BEFORE_FULL and frames == FRAME_VALUES[0] and not embeddings.exists():
        subprocess.run(
            command + ["--mode", "smoke", "--smoke-subjects", "2", "--smoke-videos-per-subject", "1"],
            check=True,
        )
    subprocess.run(command + ["--mode", "full"], check=True)
    EMBEDDING_RUNS[frames] = embeddings
    RUN_REPORTS[frames] = run_report
    REJECT_FILES[frames] = rejects

print({
    "completed_frame_runs": sorted(EMBEDDING_RUNS),
    "embedding_files_stay_in_runtime": True,
})
"""
    ),
    code(
        """
#@title 8. 다중 seed·reference 감사와 누수 검사
audit_command = [
    sys.executable, "scripts/audit_celebdf_baseline.py",
    "--output-dir", str(SANITIZED_ROOT),
    "--seeds", ",".join(str(value) for value in SEED_VALUES),
    "--reference-counts", ",".join(str(value) for value in REFERENCE_VALUES),
    "--bootstrap-repeats", str(BOOTSTRAP_REPEATS),
    "--max-reference-count", "5",
]
for frames in FRAME_VALUES:
    audit_command.extend(["--embedding-run", f"{frames}={EMBEDDING_RUNS[frames]}"])
    audit_command.extend(["--run-report", f"{frames}={RUN_REPORTS[frames]}"])
    if REJECT_FILES[frames].exists():
        audit_command.extend(["--rejects", f"{frames}={REJECT_FILES[frames]}"])
subprocess.run(audit_command, check=True)

AUDIT_JSON = SANITIZED_ROOT / "celebdf_baseline_audit.json"
METRICS_CSV = SANITIZED_ROOT / "celebdf_baseline_audit_metrics.csv"
SUMMARY_CSV = SANITIZED_ROOT / "celebdf_baseline_audit_summary.csv"
audit_report = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
assert all(item["validation_test_identity_overlap"] == 0 for item in audit_report["leakage_checks"])
assert all(item["registration_query_video_overlap"] == 0 for item in audit_report["leakage_checks"])
print(json.dumps(audit_report["decisions"], ensure_ascii=False, indent=2))
"""
    ),
    code(
        """
#@title 9. seed 변동과 운영점 결과 확인
import pandas as pd

metrics = pd.read_csv(METRICS_CSV)
summary = pd.read_csv(SUMMARY_CSV)
display(summary[[
    "frames_per_video", "reference_count",
    "test_roc_auc_mean", "test_roc_auc_min",
    "test_eer_mean", "test_eer_max",
    "far_0.001_test_tar_mean", "far_0.001_test_far_mean",
    "far_0.001_threshold_mean", "far_0.001_threshold_std",
]])
display(pd.DataFrame([
    {
        "frames_per_video": item["frames_per_video"],
        **item["quality"],
        "reject_reason_counts": item["reject_reason_counts"],
    }
    for item in audit_report["input_runs"]
]))
"""
    ),
    code(
        """
#@title 10. 감사 그래프 생성
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.lineplot(
    data=summary,
    x="frames_per_video",
    y="far_0.001_test_tar_mean",
    hue="reference_count",
    marker="o",
    palette="viridis",
    ax=axes[0],
)
axes[0].set(title="Mean TAR at validation-selected FAR=0.001", ylabel="Test TAR")
sns.lineplot(
    data=summary,
    x="frames_per_video",
    y="far_0.001_test_far_mean",
    hue="reference_count",
    marker="o",
    palette="viridis",
    ax=axes[1],
    legend=False,
)
axes[1].axhline(0.001, color="red", linestyle="--", linewidth=1, label="target FAR")
axes[1].set(title="Observed mean Test FAR", ylabel="Test FAR")
axes[1].legend()
fig.tight_layout()
FIGURE_PNG = SANITIZED_ROOT / "celebdf_baseline_audit.png"
fig.savefig(FIGURE_PNG, dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """
#@title 11. 비식별 결과 ZIP 생성과 Drive 보존
import zipfile

RUNTIME_CONFIG = SANITIZED_ROOT / "audit_runtime_config.json"
RUNTIME_CONFIG.write_text(json.dumps({
    "code_version": CODE_VERSION,
    "frames_per_video": FRAME_VALUES,
    "reference_counts": REFERENCE_VALUES,
    "seeds": SEED_VALUES,
    "bootstrap_repeats": BOOTSTRAP_REPEATS,
    "raw_data_in_bundle": False,
    "embeddings_in_bundle": False,
}, ensure_ascii=False, indent=2), encoding="utf-8")

RESULT_BUNDLE = SANITIZED_ROOT / "celebdf_baseline_audit_results.zip"
with zipfile.ZipFile(RESULT_BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in (AUDIT_JSON, METRICS_CSV, SUMMARY_CSV, FIGURE_PNG, RUNTIME_CONFIG):
        archive.write(path, arcname=path.name)

saved_to = None
if PERSIST_SANITIZED_RESULTS_TO_DRIVE and not DRIVE_SOURCE_FILE_ID.strip():
    drive_result_dir = Path(DRIVE_RESULT_DIR)
    drive_result_dir.mkdir(parents=True, exist_ok=True)
    saved_to = shutil.copy2(RESULT_BUNDLE, drive_result_dir / RESULT_BUNDLE.name)
elif IN_HOSTED_COLAB:
    from google.colab import files
    files.download(str(RESULT_BUNDLE))
print({
    "result_bundle": str(RESULT_BUNDLE),
    "size_mb": round(RESULT_BUNDLE.stat().st_size / 1e6, 2),
    "saved_to_drive": str(saved_to) if saved_to else None,
})
"""
    ),
    markdown(
        """
## 해석 제한

- 완벽한 ROC-AUC가 반복돼도 Celeb-real이 쉬운 내부 benchmark일 가능성을 먼저 고려한다.
- validation에서 고른 threshold의 test FAR 변동을 ROC-AUC보다 우선 확인한다.
- 결과 ZIP에는 집계값과 hash만 있으며 NPZ embedding은 포함하지 않는다.
- AI-Hub 승인 후 한국인 얼굴 데이터에는 동일 프로토콜을 별도 Issue로 적용한다.
"""
    ),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "celebdf_arcface_audit_colab.ipynb",
                "provenance": [],
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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
