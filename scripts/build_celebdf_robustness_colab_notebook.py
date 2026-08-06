#!/usr/bin/env python3
"""Issue #6 Celeb-DF 촬영 열화 강건성 평가 Colab 노트북을 생성한다."""

from __future__ import annotations

import base64
import hashlib
import json

from build_celebdf_colab_notebook import ROOT, code, markdown


OUTPUT = ROOT / "notebooks" / "celebdf_arcface_robustness_colab.ipynb"


def repository_bootstrap_cell() -> dict[str, object]:
    embedded_paths = [
        ROOT / "scripts" / "celebdf_faceguard.py",
        ROOT / "scripts" / "run_celebdf_arcface.py",
        ROOT / "scripts" / "audit_celebdf_robustness.py",
    ]
    embedded = {
        str(path.relative_to(ROOT)): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in embedded_paths
    }
    fingerprint = hashlib.sha256(
        b"".join(path.read_bytes() for path in embedded_paths)
    ).hexdigest()
    source = r'''#@title 3. 실행 코드 준비 — GitHub 권한이 필요 없는 내장 코드가 기본값
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
# 얼굴가드 촬영 열화 강건성 평가 — Issue #6

쉽게 말하면 **깨끗한 영상에서 등록한 얼굴이 실제 촬영처럼 어둡거나 흐려져도 같은 사람으로 잘 인식되는지** 확인하는 노트북이다.

- 기준 영상과 다섯 가지 열화 조건을 영상당 5프레임으로 처리한다.
- 등록 얼굴은 항상 깨끗한 영상만 사용한다.
- 모든 조건에서 얼굴 추론에 성공한 공통 query 영상만 비교한다.
- 깨끗한 영상에서 정한 판정 기준값과 조건별로 다시 정한 기준값을 모두 평가한다.
- 원본·얼굴 이미지·개별 점수·임베딩은 Colab 세션 밖으로 내보내지 않는다.
- Drive에는 사람을 식별할 수 없는 집계 JSON·CSV·PNG와 설정만 담은 ZIP을 저장한다.

이 실험은 얼굴 동일인 검증이며 딥페이크 탐지 정확도가 아니다.
"""
    ),
    code(
        """
#@title 1. 실행 설정과 권한 확인
REPO_URL = "https://github.com/Chunbae-A/face-image.git" #@param {type:"string"}
BRANCH = "exp/6-faceguard-robustness" #@param {type:"string"}
CODE_SOURCE = "embedded" #@param ["embedded", "github"]
SOURCE_ZIP_PATH = "/content/drive/MyDrive/face-image-data/Celeb-DF-v2.zip" #@param {type:"string"}
EXPECTED_SOURCE_ZIP_BYTES = 928989923 #@param {type:"integer"}
DRIVE_RESULT_DIR = "/content/drive/MyDrive/face-image-results/celebdf-robustness" #@param {type:"string"}
PERSIST_SANITIZED_RESULTS_TO_DRIVE = True #@param {type:"boolean"}

# 공식 신청·승인 파일이며 약관상 Hosted Colab/Drive 처리가 허용됨을 확인한 경우만 True.
I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED = False #@param {type:"boolean"}
# InsightFace 제공 buffalo_l 가중치는 비상업 연구 전용이다.
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = False #@param {type:"boolean"}

FRAMES_PER_VIDEO = 5
MINIMUM_VALID_FRAMES = 3
CONDITIONS = (
    "clean",
    "jpeg_q30",
    "gaussian_blur_sigma2",
    "low_light_gamma2",
    "downscale_0_25",
    "combined_mobile_stress",
)
SEEDS = (20260805, 20260806, 20260807, 20260808, 20260809)
BOOTSTRAP_REPEATS = 500
RUN_SMOKE_BEFORE_FULL = True #@param {type:"boolean"}

import sys
IN_HOSTED_COLAB = "google.colab" in sys.modules
if IN_HOSTED_COLAB and not I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED:
    raise PermissionError("Celeb-DF의 Hosted Colab/Drive 처리 허용 여부를 먼저 확인하세요.")
if not I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE:
    raise PermissionError("InsightFace 비상업 연구용 가중치 조건을 확인하세요.")
if EXPECTED_SOURCE_ZIP_BYTES <= 0:
    raise ValueError("Drive 원본 ZIP의 정확한 바이트를 입력해야 합니다.")

print({
    "hosted_colab": IN_HOSTED_COLAB,
    "frames_per_video": FRAMES_PER_VIDEO,
    "conditions": CONDITIONS,
    "seeds": SEEDS,
    "maximum_frame_inferences": 590 * FRAMES_PER_VIDEO * len(CONDITIONS),
})
"""
    ),
    markdown(
        """
## 실행 환경

Colab에서 GPU runtime을 선택한다. 설치 후 runtime 재시작 안내가 나오면 재시작하고, 2번 설치 셀은 건너뛴 채 1번과 3번 이후 셀을 다시 실행한다.
"""
    ),
    code(
        """
#@title 2. 라이브러리 설치
%pip uninstall -y -q onnxruntime onnxruntime-gpu
%pip install -q --no-cache-dir "insightface==1.0.1" "onnxruntime-gpu==1.23.2" "Pillow==12.3.0" opencv-python-headless pandas matplotlib seaborn tqdm
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
#@title 4. Drive 원본 확인과 세션 저장소 복사
import json
import shutil

if IN_HOSTED_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

source_zip = Path(SOURCE_ZIP_PATH).expanduser()
if not source_zip.exists():
    raise FileNotFoundError(f"Drive에서 Celeb-DF ZIP을 찾지 못했습니다: {source_zip}")
if source_zip.stat().st_size != EXPECTED_SOURCE_ZIP_BYTES:
    raise IOError(
        f"Celeb-DF ZIP 크기가 다릅니다: {source_zip.stat().st_size} "
        f"!= {EXPECTED_SOURCE_ZIP_BYTES}"
    )

WORK_ROOT = Path("/content/celebdf_robustness") if IN_HOSTED_COLAB else REPO_DIR / "outputs" / "celebdf_robustness"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
runtime_zip = WORK_ROOT / "Celeb-DF-v2.zip"
if not runtime_zip.exists() or runtime_zip.stat().st_size != EXPECTED_SOURCE_ZIP_BYTES:
    temporary_zip = runtime_zip.with_suffix(".zip.copying")
    shutil.copyfile(source_zip, temporary_zip)
    if temporary_zip.stat().st_size != EXPECTED_SOURCE_ZIP_BYTES:
        raise IOError("세션 저장소로 복사한 ZIP의 크기가 다릅니다.")
    temporary_zip.replace(runtime_zip)

VIDEO_ROOT = WORK_ROOT / "videos"
MANIFEST = WORK_ROOT / "celeb_real_manifest.csv"
INVENTORY_JSON = WORK_ROOT / "celeb_real_inventory.json"
CONDITION_ROOT = WORK_ROOT / "conditions"
SANITIZED_ROOT = WORK_ROOT / "sanitized"
for path in (VIDEO_ROOT, CONDITION_ROOT, SANITIZED_ROOT):
    path.mkdir(parents=True, exist_ok=True)

print({
    "source_zip_gb": round(source_zip.stat().st_size / 1e9, 3),
    "runtime_zip": str(runtime_zip),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
    "result_drive_dir": DRIVE_RESULT_DIR if PERSIST_SANITIZED_RESULTS_TO_DRIVE else None,
})
"""
    ),
    code(
        """
#@title 5. Celeb-real 590개 확인과 추출
subprocess.run([
    sys.executable, "scripts/celebdf_faceguard.py", "inventory", str(runtime_zip),
    "--manifest", str(MANIFEST), "--summary", str(INVENTORY_JSON),
], check=True)
inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
assert inventory["video_count"] == 590, inventory
assert inventory["subject_count"] == 59, inventory
assert inventory["eligible_subjects_ge_8_videos"] == 56, inventory

subprocess.run([
    sys.executable, "scripts/celebdf_faceguard.py", "extract", str(runtime_zip),
    "--manifest", str(MANIFEST), "--output", str(VIDEO_ROOT), "--mode", "full",
], check=True)
extracted = sorted((VIDEO_ROOT / "Celeb-real").glob("*.mp4"))
if len(extracted) != 590:
    raise RuntimeError(f"590개 영상이 필요하지만 {len(extracted)}개를 찾았습니다.")
print({"videos": len(extracted), "eligible_subjects": 56})
"""
    ),
    code(
        """
#@title 6. GPU와 ONNX Runtime 확인
import onnxruntime as ort

providers = ort.get_available_providers()
print({"onnxruntime": ort.__version__, "providers": providers})
if IN_HOSTED_COLAB and "CUDAExecutionProvider" not in providers:
    raise RuntimeError(
        "GPU 실행기가 연결되지 않았습니다. GPU runtime을 재시작하고 2번 설치 셀은 다시 실행하지 마세요."
    )
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
))
"""
    ),
    code(
        """
#@title 7. 깨끗한 영상과 다섯 가지 촬영 열화 조건 추론
EMBEDDING_RUNS = {}
RUN_REPORTS = {}
REJECT_FILES = {}

for condition in CONDITIONS:
    run_root = CONDITION_ROOT / condition
    run_root.mkdir(parents=True, exist_ok=True)
    embeddings = run_root / "video_embeddings.npz"
    rejects = run_root / "rejects.csv"
    run_report = run_root / "run.json"
    command = [
        sys.executable, "scripts/run_celebdf_arcface.py",
        "--manifest", str(MANIFEST),
        "--video-root", str(VIDEO_ROOT),
        "--output", str(embeddings),
        "--rejects", str(rejects),
        "--run-report", str(run_report),
        "--frames-per-video", str(FRAMES_PER_VIDEO),
        "--minimum-valid-frames", str(MINIMUM_VALID_FRAMES),
        "--input-condition", condition,
        "--checkpoint-every", "25",
        "--progress-every", "25",
        "--model-name", "buffalo_l",
        "--accept-noncommercial-model-license",
    ]
    if RUN_SMOKE_BEFORE_FULL and condition == "clean" and not embeddings.exists():
        subprocess.run(
            command + [
                "--mode", "smoke", "--smoke-subjects", "2",
                "--smoke-videos-per-subject", "1",
            ],
            check=True,
        )
    subprocess.run(command + ["--mode", "full"], check=True)
    completed = json.loads(run_report.read_text(encoding="utf-8"))
    if completed["status"] != "completed" or completed["input_condition"] != condition:
        raise RuntimeError(f"조건 실행이 완료되지 않았습니다: {condition}")
    EMBEDDING_RUNS[condition] = embeddings
    RUN_REPORTS[condition] = run_report
    REJECT_FILES[condition] = rejects
    print({
        "completed_condition": condition,
        "successful_videos": completed["successful_video_count_total"],
    })

print({
    "completed_conditions": tuple(EMBEDDING_RUNS),
    "embeddings_stay_in_colab_runtime": True,
})
"""
    ),
    code(
        """
#@title 8. 공통 query 평가와 판정 기준값 보정
audit_command = [
    sys.executable, "scripts/audit_celebdf_robustness.py",
    "--output-dir", str(SANITIZED_ROOT),
    "--seeds", ",".join(str(seed) for seed in SEEDS),
    "--reference-count", "3",
    "--bootstrap-repeats", str(BOOTSTRAP_REPEATS),
]
for condition in CONDITIONS:
    audit_command.extend(["--embedding-run", f"{condition}={EMBEDDING_RUNS[condition]}"])
    audit_command.extend(["--run-report", f"{condition}={RUN_REPORTS[condition]}"])
    if REJECT_FILES[condition].exists():
        audit_command.extend(["--rejects", f"{condition}={REJECT_FILES[condition]}"])
subprocess.run(audit_command, check=True)

AUDIT_JSON = SANITIZED_ROOT / "celebdf_robustness_audit.json"
METRICS_CSV = SANITIZED_ROOT / "celebdf_robustness_metrics.csv"
SUMMARY_CSV = SANITIZED_ROOT / "celebdf_robustness_summary.csv"
audit_report = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
assert all(item["validation_test_identity_overlap"] == 0 for item in audit_report["leakage_checks"])
assert all(item["registration_query_video_overlap"] == 0 for item in audit_report["leakage_checks"])
print(json.dumps(audit_report["decisions"], ensure_ascii=False, indent=2))
"""
    ),
    code(
        """
#@title 9. 조건별 결과 표 확인
import pandas as pd

summary = pd.read_csv(SUMMARY_CSV)
display(summary[[
    "condition",
    "test_roc_auc_mean",
    "test_eer_mean",
    "far_0.001_clean_locked_test_tar_mean",
    "far_0.001_clean_locked_test_far_mean",
    "far_0.001_condition_calibrated_test_tar_mean",
    "far_0.001_condition_calibrated_test_far_mean",
    "far_0.001_threshold_shift_mean",
]])
display(pd.DataFrame([
    {
        "condition": item["condition"],
        **item["quality"],
        "reject_reason_counts": item["reject_reason_counts"],
    }
    for item in audit_report["input_runs"]
]))
"""
    ),
    code(
        """
#@title 10. 촬영 열화별 성능 그래프 생성
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
order = list(CONDITIONS)
figure, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(
    data=summary,
    x="condition",
    y="far_0.001_clean_locked_test_tar_mean",
    order=order,
    color="#4c78a8",
    ax=axes[0],
)
axes[0].set(title="Recognition rate with clean-locked threshold", ylabel="Test TAR", xlabel="")
axes[0].tick_params(axis="x", rotation=35)
sns.barplot(
    data=summary,
    x="condition",
    y="far_0.001_clean_locked_test_far_mean",
    order=order,
    color="#f58518",
    ax=axes[1],
)
axes[1].axhline(0.001, color="red", linestyle="--", linewidth=1, label="target FAR")
axes[1].set(title="False acceptance with clean-locked threshold", ylabel="Test FAR", xlabel="")
axes[1].tick_params(axis="x", rotation=35)
axes[1].legend()
figure.tight_layout()
FIGURE_PNG = SANITIZED_ROOT / "celebdf_robustness_audit.png"
figure.savefig(FIGURE_PNG, dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """
#@title 11. 얼굴 없는 결과 ZIP만 Drive에 저장
import hashlib
import zipfile

RUNTIME_CONFIG = SANITIZED_ROOT / "robustness_runtime_config.json"
RUNTIME_CONFIG.write_text(json.dumps({
    "environment": "colab",
    "code_version": CODE_VERSION,
    "source_zip_bytes": EXPECTED_SOURCE_ZIP_BYTES,
    "frames_per_video": FRAMES_PER_VIDEO,
    "minimum_valid_frames": MINIMUM_VALID_FRAMES,
    "conditions": CONDITIONS,
    "reference_count": 3,
    "reserved_registration_count": 5,
    "seeds": SEEDS,
    "bootstrap_repeats": BOOTSTRAP_REPEATS,
    "raw_data_in_bundle": False,
    "embeddings_in_bundle": False,
}, ensure_ascii=False, indent=2), encoding="utf-8")

RESULT_BUNDLE = SANITIZED_ROOT / "celebdf_robustness_results.zip"
allowed_artifacts = (AUDIT_JSON, METRICS_CSV, SUMMARY_CSV, FIGURE_PNG, RUNTIME_CONFIG)
with zipfile.ZipFile(RESULT_BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in allowed_artifacts:
        archive.write(path, arcname=path.name)

with zipfile.ZipFile(RESULT_BUNDLE) as archive:
    names = set(archive.namelist())
    if names != {path.name for path in allowed_artifacts}:
        raise AssertionError(f"결과 ZIP 허용 목록이 다릅니다: {sorted(names)}")
    if any(name.endswith(".npz") for name in names):
        raise AssertionError("결과 ZIP에 임베딩이 포함되었습니다.")

bundle_sha256 = hashlib.sha256(RESULT_BUNDLE.read_bytes()).hexdigest()
saved_to = None
if PERSIST_SANITIZED_RESULTS_TO_DRIVE:
    drive_result_dir = Path(DRIVE_RESULT_DIR)
    drive_result_dir.mkdir(parents=True, exist_ok=True)
    saved_to = shutil.copy2(RESULT_BUNDLE, drive_result_dir / RESULT_BUNDLE.name)
elif IN_HOSTED_COLAB:
    from google.colab import files
    files.download(str(RESULT_BUNDLE))

print({
    "result_bundle": str(RESULT_BUNDLE),
    "bundle_sha256": bundle_sha256,
    "size_mb": round(RESULT_BUNDLE.stat().st_size / 1e6, 2),
    "saved_to_drive": str(saved_to) if saved_to else None,
    "raw_or_embedding_files_in_bundle": False,
})
"""
    ),
    markdown(
        """
## 결과를 읽는 순서

1. `success_rate`가 0.98보다 낮으면 그 촬영 조건은 얼굴 검출 단계부터 불안정하다는 뜻이다.
2. `clean_locked_test_tar`가 깨끗한 조건보다 0.05 이상 낮아지면 촬영 품질 안내나 재촬영 유도가 필요하다.
3. `clean_locked_test_far`가 0.001보다 높으면 깨끗한 영상에서 정한 하나의 판정 기준값을 모든 조건에 쓰면 안 된다.
4. 조건별 보정 후에도 `condition_calibrated_test_far`가 0.001보다 높으면 이 결과만으로 운영 승인을 내리지 않는다.

Celeb-real 내부 결과는 한국인 얼굴·실제 휴대전화·운영 트래픽의 성능을 대신하지 않는다. AI-Hub 승인 데이터나 동의받은 실제 촬영 데이터에서 같은 절차를 다시 수행해야 한다.
"""
    ),
]


def build_notebook() -> dict[str, object]:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "celebdf_arcface_robustness_colab.ipynb",
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
