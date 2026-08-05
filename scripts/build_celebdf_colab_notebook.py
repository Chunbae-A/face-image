#!/usr/bin/env python3
"""Generate the full Celeb-DF-v2 ArcFace Colab notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "celebdf_arcface_full_colab.ipynb"


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
        ROOT / "scripts" / "celebdf_faceguard.py",
        ROOT / "scripts" / "run_celebdf_arcface.py",
    ]
    embedded = {
        str(path.relative_to(ROOT)): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in embedded_paths
    }
    fingerprint = hashlib.sha256(
        b"".join(path.read_bytes() for path in embedded_paths)
    ).hexdigest()
    source = r'''#@title 3. 실행 코드 준비 — 기본값은 GitHub 권한이 필요 없는 내장 모드
from pathlib import Path
import base64
import os
import subprocess

EMBEDDED_FILES_B64 = ''' + repr(embedded) + r'''
EMBEDDED_CODE_SHA256 = "''' + fingerprint + r'''"

if IN_HOSTED_COLAB and CODE_SOURCE == "github":
    REPO_DIR = Path("/content/deepsogak")
    if not REPO_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
    GIT_COMMIT = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
elif IN_HOSTED_COLAB:
    REPO_DIR = Path("/content/deepsogak")
    for relative_path, encoded in EMBEDDED_FILES_B64.items():
        target = REPO_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded))
    GIT_COMMIT = f"embedded:{EMBEDDED_CODE_SHA256[:12]}"
else:
    REPO_DIR = Path.cwd()
    try:
        GIT_COMMIT = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        GIT_COMMIT = f"local:{EMBEDDED_CODE_SHA256[:12]}"

os.chdir(REPO_DIR)
print({"repo": str(REPO_DIR), "code_source": CODE_SOURCE, "code_version": GIT_COMMIT})
'''
    return code(source)


CELLS = [
    markdown(
        """
# Celeb-DF-v2 전체 ArcFace 얼굴인식 평가 — 딥소각 얼굴가드

이 노트북은 **Celeb-real 590개 영상 전체**를 처리한다. 각 영상에서 10개 프레임을 균등 추출하고, 얼굴 탐지·정렬·ArcFace 추론 후 프레임 임베딩을 평균하여 **영상당 하나의 임베딩**을 만든다. 같은 영상의 프레임이 등록과 테스트에 동시에 들어가지 않는다.

- 원본 규모: 590개 영상, 59명, 약 946.5MB
- 기본 처리량: 최대 5,900개 프레임
- 최종 평가: 등록 영상 5개 + 테스트 영상 3개 이상이 가능한 56명
- 프로토콜: 등록 3개 영상 / 등록 5개 영상, query는 두 프로토콜 모두 6번째 영상부터 사용
- 검증/테스트: 인물 ID가 겹치지 않는 30% / 70% subject-disjoint 분할
- 지표: ROC-AUC, EER, validation에서 고정한 threshold의 TAR/FAR/FRR, 95% subject bootstrap CI

이 실험은 **일반 얼굴 동일인 검증**이며 딥페이크 탐지 정확도가 아니다. AI-Hub 데이터는 승인 전 사용하지 않는다.

InsightFace 코드는 MIT이지만 제공 사전학습 모델은 비상업 연구 용도이다. 해커톤 연구 검증에만 사용하고 제품에 그대로 탑재하지 않는다.
"""
    ),
    code(
        """
#@title 1. 실행 설정과 권한 확인
REPO_URL = "https://github.com/Chunbae-A/deepsogak.git" #@param {type:"string"}
BRANCH = "feat/faceguard-experiment-plan" #@param {type:"string"}
CODE_SOURCE = "embedded" #@param ["embedded", "github"]
USE_GOOGLE_DRIVE = True #@param {type:"boolean"}
SOURCE_ZIP_PATH = "/content/drive/MyDrive/Celeb-DF-v2.zip" #@param {type:"string"}
COPY_ZIP_TO_RUNTIME = False #@param {type:"boolean"}
PERSIST_DERIVED_RESULTS_TO_DRIVE = False #@param {type:"boolean"}
DRIVE_RESULT_DIR = "/content/drive/MyDrive/deepsogak-celebdf-results" #@param {type:"string"}

# 공식 신청·승인으로 받은 파일이며, 해당 약관상 Colab/Drive 처리가 허용되는지 직접 확인 후 True로 변경합니다.
I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED = False #@param {type:"boolean"}
# InsightFace 제공 buffalo_l 가중치는 비상업 연구 전용입니다.
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = False #@param {type:"boolean"}

RUN_SMOKE_BEFORE_FULL = True #@param {type:"boolean"}
RUN_FULL_590_VIDEOS = True #@param {type:"boolean"}
FRAMES_PER_VIDEO = 10 #@param {type:"integer"}
MINIMUM_VALID_FRAMES = 3 #@param {type:"integer"}
BOOTSTRAP_REPEATS = 500 #@param {type:"integer"}
SEED = 20260805 #@param {type:"integer"}

import sys
IN_HOSTED_COLAB = "google.colab" in sys.modules
if IN_HOSTED_COLAB and not I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED:
    raise PermissionError(
        "Hosted Colab/Drive processing is blocked until the Celeb-DF terms are checked. "
        "After checking them, set I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED=True."
    )
if not I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE:
    raise PermissionError(
        "Review the InsightFace model license, then set "
        "I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE=True."
    )
if not RUN_FULL_590_VIDEOS:
    raise ValueError("This notebook is configured for the requested full 590-video run.")
print({
    "hosted_colab": IN_HOSTED_COLAB,
    "run_full": RUN_FULL_590_VIDEOS,
    "frames_per_video": FRAMES_PER_VIDEO,
    "maximum_frame_inferences": 590 * FRAMES_PER_VIDEO,
})
"""
    ),
    markdown(
        """
## 권장 실행 환경

Colab 메뉴에서 **런타임 → 런타임 유형 변경 → GPU**를 선택한다. 아래 설치는 2026-08-05 기준 공식 PyPI의 `insightface==1.0.1`을 사용한다. 설치 후 ONNX Runtime import 오류가 나면 런타임을 한 번 다시 시작하고 1번 셀부터 재실행한다.
"""
    ),
    code(
        """
#@title 2. 라이브러리 설치
%pip uninstall -y -q onnxruntime onnxruntime-gpu
%pip install -q --no-cache-dir "insightface==1.0.1" "onnxruntime-gpu==1.23.2" opencv-python-headless pandas scikit-learn matplotlib seaborn tqdm
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
#@title 4. Drive 연결, ZIP 확인, 작업 경로 설정
import shutil

if USE_GOOGLE_DRIVE:
    if not IN_HOSTED_COLAB:
        print("Local runtime: Google Drive mount cell is skipped.")
    else:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)

source_zip = Path(SOURCE_ZIP_PATH).expanduser()
if not source_zip.exists():
    raise FileNotFoundError(
        f"ZIP not found: {source_zip}. Upload the official Celeb-DF-v2.zip or correct SOURCE_ZIP_PATH."
    )

WORK_ROOT = Path("/content/celebdf_faceguard") if IN_HOSTED_COLAB else REPO_DIR / "outputs" / "celebdf_faceguard"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT = WORK_ROOT / "videos"
MANIFEST = WORK_ROOT / "celeb_real_manifest.csv"
INVENTORY_JSON = WORK_ROOT / "celeb_real_inventory.json"

if PERSIST_DERIVED_RESULTS_TO_DRIVE:
    if not I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED:
        raise PermissionError("Derived embedding persistence also requires cloud-processing permission.")
    RESULT_ROOT = Path(DRIVE_RESULT_DIR)
else:
    RESULT_ROOT = WORK_ROOT / "results"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

runtime_zip = source_zip
if COPY_ZIP_TO_RUNTIME:
    required = int(source_zip.stat().st_size * 1.25 + 3_000_000_000)
    free = shutil.disk_usage(WORK_ROOT).free
    if free < required:
        raise OSError(f"Not enough runtime disk: free={free}, required={required}")
    runtime_zip = WORK_ROOT / "Celeb-DF-v2.zip"
    if not runtime_zip.exists() or runtime_zip.stat().st_size != source_zip.stat().st_size:
        from tqdm.auto import tqdm
        temporary = runtime_zip.with_suffix(".zip.part")
        with source_zip.open("rb") as src, temporary.open("wb") as dst, tqdm(
            total=source_zip.stat().st_size, unit="B", unit_scale=True, desc="Copy ZIP"
        ) as progress:
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)
                progress.update(len(chunk))
        temporary.replace(runtime_zip)

print({
    "source_zip": str(source_zip),
    "zip_gb": round(source_zip.stat().st_size / 1e9, 3),
    "runtime_zip": str(runtime_zip),
    "work_root": str(WORK_ROOT),
    "result_root": str(RESULT_ROOT),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
})
"""
    ),
    code(
        """
#@title 5. ZIP 중앙 디렉터리 검사와 전체 590개 manifest 생성
import json

inventory_command = [
    sys.executable,
    "scripts/celebdf_faceguard.py",
    "inventory",
    str(runtime_zip),
    "--manifest", str(MANIFEST),
    "--summary", str(INVENTORY_JSON),
]
subprocess.run(inventory_command, check=True)
inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
assert inventory["video_count"] == 590, inventory
assert inventory["subject_count"] == 59, inventory
assert inventory["eligible_subjects_ge_8_videos"] == 56, inventory
print(json.dumps({
    key: inventory[key] for key in [
        "video_count", "subject_count", "uncompressed_bytes",
        "eligible_subjects_ge_8_videos", "excluded_subjects_lt_8_videos"
    ]
}, ensure_ascii=False, indent=2))
"""
    ),
    code(
        """
#@title 6. Smoke 2개 영상 추출 후 Celeb-real 590개 전체 추출
if RUN_SMOKE_BEFORE_FULL:
    subprocess.run([
        sys.executable, "scripts/celebdf_faceguard.py", "extract", str(runtime_zip),
        "--manifest", str(MANIFEST), "--output", str(VIDEO_ROOT),
        "--mode", "smoke", "--smoke-subjects", "2", "--smoke-videos-per-subject", "1",
    ], check=True)

subprocess.run([
    sys.executable, "scripts/celebdf_faceguard.py", "extract", str(runtime_zip),
    "--manifest", str(MANIFEST), "--output", str(VIDEO_ROOT), "--mode", "full",
], check=True)
extracted_videos = sorted((VIDEO_ROOT / "Celeb-real").glob("*.mp4"))
if len(extracted_videos) != 590:
    raise RuntimeError(f"Expected 590 extracted videos, found {len(extracted_videos)}")
print({
    "extracted_videos": len(extracted_videos),
    "extracted_gb": round(sum(path.stat().st_size for path in extracted_videos) / 1e9, 3),
})
"""
    ),
    code(
        """
#@title 7. GPU/ONNX Runtime 확인
import subprocess
import onnxruntime as ort

providers = ort.get_available_providers()
print({"onnxruntime": ort.__version__, "providers": providers})
if IN_HOSTED_COLAB and "CUDAExecutionProvider" not in providers:
    raise RuntimeError(
        "CUDAExecutionProvider is unavailable. Select a GPU runtime, then restart and rerun."
    )
try:
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True))
except (FileNotFoundError, subprocess.SubprocessError):
    print("nvidia-smi unavailable; CPU/local runtime may be active.")
"""
    ),
    code(
        """
#@title 8. Smoke 추론 — 설치·모델 다운로드·얼굴 탐지 확인
EMBEDDINGS_NPZ = RESULT_ROOT / "celeb_real_video_embeddings.npz"
REJECTS_CSV = RESULT_ROOT / "celeb_real_rejects.csv"
RUN_REPORT_JSON = RESULT_ROOT / "celeb_real_arcface_run.json"

base_runner_command = [
    sys.executable, "scripts/run_celebdf_arcface.py",
    "--manifest", str(MANIFEST),
    "--video-root", str(VIDEO_ROOT),
    "--output", str(EMBEDDINGS_NPZ),
    "--rejects", str(REJECTS_CSV),
    "--run-report", str(RUN_REPORT_JSON),
    "--frames-per-video", str(FRAMES_PER_VIDEO),
    "--minimum-valid-frames", str(MINIMUM_VALID_FRAMES),
    "--checkpoint-every", "25",
    "--model-name", "buffalo_l",
    "--accept-noncommercial-model-license",
]
if RUN_SMOKE_BEFORE_FULL:
    subprocess.run(
        base_runner_command + [
            "--mode", "smoke", "--smoke-subjects", "2", "--smoke-videos-per-subject", "1"
        ],
        check=True,
    )
print("Smoke inference completed. The full cell below resumes from the same NPZ checkpoint.")
"""
    ),
    code(
        """
#@title 9. Celeb-real 590개 전체 ArcFace 실행
if RUN_FULL_590_VIDEOS:
    subprocess.run(base_runner_command + ["--mode", "full"], check=True)
else:
    raise RuntimeError("Full run was unexpectedly disabled.")
print(json.dumps(json.loads(RUN_REPORT_JSON.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
"""
    ),
    code(
        """
#@title 10. 처리 품질 확인
import numpy as np
import pandas as pd

with np.load(EMBEDDINGS_NPZ, allow_pickle=False) as payload:
    quality = pd.DataFrame({
        "subject_id": payload["subject_ids"],
        "video_id": payload["video_ids"],
        "sampled_frames": payload["sampled_frames"],
        "valid_frames": payload["valid_frames"],
        "mean_detection_score": payload["mean_detection_scores"],
        "mean_face_area_ratio": payload["mean_face_area_ratios"],
        "decode_seconds": payload["decode_seconds"],
        "inference_seconds": payload["inference_seconds"],
    })
print({
    "successful_videos": len(quality),
    "success_rate": len(quality) / 590,
    "eligible_subjects_ge_8_successful_videos": int((quality.groupby("subject_id").size() >= 8).sum()),
})
display(quality.describe(include="all"))
display(quality.groupby("subject_id").size().sort_values().rename("successful_videos").to_frame())
if len(quality) < 560:
    print("WARNING: success rate is below the expected guardrail; inspect reject reasons before evaluation.")
"""
    ),
    code(
        """
#@title 11. 등록 3장/5장 평가와 95% CI 생성
METRICS_JSON = RESULT_ROOT / "celeb_real_arcface_metrics.json"
subprocess.run([
    sys.executable, "scripts/celebdf_faceguard.py", "evaluate",
    "--embeddings", str(EMBEDDINGS_NPZ),
    "--output", str(METRICS_JSON),
    "--seed", str(SEED),
    "--bootstrap-repeats", str(BOOTSTRAP_REPEATS),
], check=True)
metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))

metric_rows = []
for protocol_name, protocol in metrics["protocols"].items():
    row = {
        "protocol": protocol_name,
        "test_roc_auc": protocol["test_roc_auc"],
        "test_eer": protocol["test_eer"],
        "roc_auc_ci_low": protocol["roc_auc_95ci"][0],
        "roc_auc_ci_high": protocol["roc_auc_95ci"][1],
        "eer_ci_low": protocol["eer_95ci"][0],
        "eer_ci_high": protocol["eer_95ci"][1],
        "positive_pairs": protocol["test_positive_pairs"],
        "negative_pairs": protocol["test_negative_pairs"],
    }
    for far_key, point in protocol["operating_points"].items():
        row[f"{far_key}_threshold"] = point["threshold_selected_on_validation"]
        row[f"{far_key}_test_tar"] = point["test"]["tar"]
        row[f"{far_key}_test_far"] = point["test"]["far"]
        row[f"{far_key}_test_frr"] = point["test"]["frr"]
    metric_rows.append(row)
metrics_table = pd.DataFrame(metric_rows)
METRICS_CSV = RESULT_ROOT / "celeb_real_arcface_metrics.csv"
metrics_table.to_csv(METRICS_CSV, index=False)
display(metrics_table.T)
print({
    "eligible_subjects": metrics["eligible_subject_count"],
    "validation_subjects": metrics["validation_subject_count"],
    "test_subjects": metrics["test_subject_count"],
})
"""
    ),
    code(
        """
#@title 12. ROC와 score 분포 그래프
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, str(REPO_DIR / "scripts"))
from celebdf_faceguard import (
    auc_eer, build_pair_scores, group_eligible_records,
    load_video_embeddings, roc_curve, split_subjects,
)

records = load_video_embeddings(EMBEDDINGS_NPZ)
grouped = group_eligible_records(records, seed=SEED)
validation_subjects, test_subjects = split_subjects(grouped, seed=SEED)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for reference_count, color in [(3, "#2E74B5"), (5, "#E07A2D")]:
    pairs = build_pair_scores(grouped, test_subjects, reference_count=reference_count)
    fpr, tpr, _ = roc_curve(pairs.labels, pairs.scores)
    auc, eer = auc_eer(pairs.labels, pairs.scores)
    axes[0].semilogx(np.clip(fpr, 1e-5, 1), tpr, color=color, label=f"ref {reference_count} | AUC={auc:.4f}, EER={eer:.4f}")
    sample_negative = pairs.scores[pairs.labels == 0]
    if len(sample_negative) > 20000:
        rng = np.random.default_rng(SEED + reference_count)
        sample_negative = rng.choice(sample_negative, 20000, replace=False)
    sns.kdeplot(pairs.scores[pairs.labels == 1], ax=axes[1], color=color, linestyle="-", label=f"ref {reference_count} positive")
    sns.kdeplot(sample_negative, ax=axes[1], color=color, linestyle="--", label=f"ref {reference_count} negative")
axes[0].set(xlabel="False Accept Rate (log)", ylabel="True Accept Rate", title="Celeb-real identity verification ROC", xlim=(1e-5, 1), ylim=(0, 1.01))
axes[0].grid(True, alpha=0.25)
axes[0].legend()
axes[1].set(xlabel="Cosine similarity", ylabel="Density", title="Test score distributions")
axes[1].legend()
fig.tight_layout()
FIGURE_PNG = RESULT_ROOT / "celeb_real_arcface_roc_scores.png"
fig.savefig(FIGURE_PNG, dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """
#@title 13. 비식별 결과 묶음 저장
import zipfile

RESULT_BUNDLE = RESULT_ROOT / "celeb_real_arcface_results.zip"
bundle_files = [
    INVENTORY_JSON, RUN_REPORT_JSON, METRICS_JSON, METRICS_CSV, FIGURE_PNG,
]
if REJECTS_CSV.exists():
    bundle_files.append(REJECTS_CSV)
with zipfile.ZipFile(RESULT_BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in bundle_files:
        archive.write(path, arcname=path.name)
print({"result_bundle": str(RESULT_BUNDLE), "size_mb": round(RESULT_BUNDLE.stat().st_size / 1e6, 2)})

if IN_HOSTED_COLAB and not PERSIST_DERIVED_RESULTS_TO_DRIVE:
    from google.colab import files
    files.download(str(RESULT_BUNDLE))
"""
    ),
    markdown(
        """
## 결과 해석 시 주의사항

1. 이 결과는 사전학습 `buffalo_l` ArcFace baseline의 **Celeb-real 동일인 검증 성능**이다.
2. threshold는 validation 인물에서 고정한 뒤 겹치지 않는 test 인물에 적용한다. 운영 임계값은 실제 서비스 데이터로 다시 검증해야 한다.
3. Celeb-real은 한국인 전용 데이터가 아니다. AI-Hub 승인이 나면 같은 프로토콜을 한국인 안면 이미지에 재실행하여 일반화 차이를 비교한다.
4. 실제 얼굴 이미지나 프레임은 결과 묶음에 포함하지 않는다. `.npz` 임베딩은 생체정보로 취급하며 외부 공개·Git 커밋을 금지한다.
5. `buffalo_l` 제공 가중치는 비상업 연구 전용이다. 제품 배포 전 상업 사용 가능한 별도 모델·가중치를 선택한다.

공식 참고: [InsightFace PyPI](https://pypi.org/project/insightface/), [InsightFace GitHub](https://github.com/deepinsight/insightface)
"""
    ),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "celebdf_arcface_full_colab.ipynb",
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
