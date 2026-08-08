#!/usr/bin/env python3
"""Generate the private Kaggle EfficientNet-B4 versus Xception notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "celebdf_effb4_xception_compare_kaggle.ipynb"


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
        ROOT / "configs" / "deepfake" / "effb4_xception_comparison.json",
        ROOT / "scripts" / "celebdf_deepfake.py",
        ROOT / "scripts" / "run_celebdf_deepfake.py",
        ROOT / "scripts" / "compare_deepfake_model_candidates.py",
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


CELLS = [
    markdown(
        """
# EfficientNet-B4와 Xception 공정 비교 — Kaggle 무료 GPU

이 노트북은 기존의 **같은 비공개 얼굴 crop과 분할**로 EfficientNet-B4와 Xception을 각각 학습한다. 두 모델 모두 `256×256`, 정규화 `0.5`, 같은 seed·프레임·증강·optimizer·epoch 예산을 사용한다.

실행 순서는 다음과 같다.

1. 두 모델을 공식 Test 없이 학습한다.
2. Validation의 clean·압축·흐림·저조도·축소 조건만 비교한다.
3. `Recall ≥ 95%`에서 FPR이 낮은 모델 하나를 자동으로 고정한다.
4. 고정된 모델 하나만 공식 Test에 평가한다.
5. 선택 모델을 ONNX로 내보내 CPU 연결 시험을 한다.

새 결과가 나오기 전까지 두 모델 중 어느 것이 더 좋다고 결론 내리지 않는다. Notebook과 Output은 반드시 Private으로 유지한다.
"""
    ),
    code(
        """
# 1. 사용자가 확인할 설정
REPO_URL = "https://github.com/Chunbae-A/face-image.git"
BRANCH = "main"
CODE_SOURCE = "embedded"  # 권한 문제를 피하려면 embedded 유지
PREPROCESS_ARCHIVE_PATH = ""  # 비우면 Kaggle Input에서 자동 탐색

I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED = False
RUN_TRAINING = True
RUN_FINAL_OFFICIAL_TEST_FOR_FROZEN_WINNER = True

import os
from pathlib import Path
import sys

IN_KAGGLE = Path("/kaggle").exists() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if not I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED:
    raise PermissionError("1단계 얼굴 crop Output을 비공개 연구에 사용할 수 있는지 확인하세요.")
print({"kaggle": IN_KAGGLE, "official_test_for_one_frozen_winner": RUN_FINAL_OFFICIAL_TEST_FOR_FROZEN_WINNER})
"""
    ),
    code(
        """
# 2. 모델·ONNX 의존성 — Kaggle의 torch/torchvision은 교체하지 않음
%pip install -q --no-cache-dir "timm==1.0.28" "onnx==1.18.0" "onnxruntime==1.23.2" "Pillow==11.3.0"
"""
    ),
    repository_bootstrap_cell(),
    code(
        """
# 4. 고정 비교 설정 읽기
import json

CONFIG_PATH = REPO_DIR / "configs" / "deepfake" / "effb4_xception_comparison.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
assert CONFIG["selection"]["official_test_used_for_selection"] is False
assert CONFIG["selection"]["official_test_candidates_evaluated"] == 1
assert len({candidate["id"] for candidate in CONFIG["candidates"]}) == 2
print({
    "experiment": CONFIG["experiment_id"],
    "candidates": [candidate["id"] for candidate in CONFIG["candidates"]],
    "seed": CONFIG["seed"],
    "input_size": CONFIG["input_size"],
    "normalization": CONFIG["normalization"],
    "effective_batch_size": CONFIG["effective_batch_size"],
})
"""
    ),
    code(
        """
# 5. 1단계 비공개 얼굴 crop Output 복원
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

WORK_ROOT = Path("/kaggle/temp/effb4_xception_compare") if IN_KAGGLE else REPO_DIR / "outputs" / "effb4_xception_compare"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "-xf", str(exact[0]), "-C", str(WORK_ROOT)], check=True)
CROP_ROOT = WORK_ROOT / "crops"
CROP_MANIFEST = WORK_ROOT / "crop_private_manifest.csv"
INVENTORY = WORK_ROOT / "inventory_aggregate.json"
PREPROCESS_REPORT = WORK_ROOT / "preprocess_aggregate.json"
if not CROP_ROOT.is_dir() or not CROP_MANIFEST.is_file():
    raise RuntimeError("1단계 얼굴 crop 또는 manifest 복원에 실패했습니다.")

OUTPUT_ROOT = Path("/kaggle/working") if IN_KAGGLE else WORK_ROOT / "output"
PRIVATE_ROOT = OUTPUT_ROOT / "private_candidates"
SANITIZED_ROOT = OUTPUT_ROOT / "sanitized_comparison"
PRIVATE_SCORE_ROOT = WORK_ROOT / "private_scores"
for directory in (PRIVATE_ROOT, SANITIZED_ROOT, PRIVATE_SCORE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

COMPARISON_REPORT = SANITIZED_ROOT / "validation_candidate_comparison.json"
FINAL_METRICS = SANITIZED_ROOT / "frozen_winner_official_test_metrics.json"
ONNX_MODEL = PRIVATE_ROOT / "frozen_winner.onnx"
ONNX_EXPORT_REPORT = SANITIZED_ROOT / "frozen_winner_onnx_export.json"
ONNX_SMOKE_REPORT = SANITIZED_ROOT / "frozen_winner_onnx_cpu_smoke.json"
FIGURE = SANITIZED_ROOT / "effb4_xception_validation_comparison.png"
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
# 6. GPU와 라이브러리 확인
import timm
import torch
import torchvision

cuda_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
cuda_capability = torch.cuda.get_device_capability(0) if cuda_available else None
compiled_arches = torch.cuda.get_arch_list() if cuda_available else []
required_arch = f"sm_{cuda_capability[0]}{cuda_capability[1]}" if cuda_capability else None
print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "cuda_available": cuda_available,
    "gpu": gpu_name,
    "compiled_arches": compiled_arches,
})
if IN_KAGGLE and not cuda_available:
    raise RuntimeError("Kaggle Settings에서 GPU를 선택하고 다시 시작하세요.")
if IN_KAGGLE and compiled_arches and required_arch not in compiled_arches:
    raise RuntimeError(f"현재 GPU({gpu_name}, {required_arch})를 설치된 PyTorch가 지원하지 않습니다.")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
))
"""
    ),
    code(
        """
# 7. 두 후보를 같은 예산으로 학습 — 공식 Test 사용 안 함
if not RUN_TRAINING:
    raise ValueError("첫 비교 실행에서는 RUN_TRAINING=True가 필요합니다.")

CANDIDATE_PATHS = {}
for candidate in CONFIG["candidates"]:
    candidate_id = candidate["id"]
    candidate_private = PRIVATE_ROOT / candidate_id
    candidate_sanitized = SANITIZED_ROOT / candidate_id
    candidate_private.mkdir(parents=True, exist_ok=True)
    candidate_sanitized.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_private / "best.pt"
    train_report = candidate_sanitized / "train_aggregate.json"
    validation_metrics = candidate_sanitized / "validation_metrics.json"
    validation_scores = PRIVATE_SCORE_ROOT / f"{candidate_id}_validation_scores.csv"
    CANDIDATE_PATHS[candidate_id] = {
        "checkpoint": checkpoint,
        "train_report": train_report,
        "validation_metrics": validation_metrics,
        "validation_scores": validation_scores,
    }
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "train",
        "--crop-manifest", str(CROP_MANIFEST), "--crop-root", str(CROP_ROOT),
        "--checkpoint", str(checkpoint), "--train-report", str(train_report),
        "--architecture", candidate_id,
        "--normalization", CONFIG["normalization"],
        "--input-size", str(CONFIG["input_size"]),
        "--train-frames-per-video", str(CONFIG["train_frames_per_video"]),
        "--batch-size", str(CONFIG["batch_size"]),
        "--gradient-accumulation-steps", str(CONFIG["gradient_accumulation_steps"]),
        "--epochs", str(CONFIG["epochs"]),
        "--early-stopping-patience", str(CONFIG["early_stopping_patience"]),
        "--learning-rate", str(CONFIG["learning_rate"]),
        "--weight-decay", str(CONFIG["weight_decay"]),
        "--seed", str(CONFIG["seed"]), "--require-cuda",
    ], check=True)
    report = json.loads(train_report.read_text(encoding="utf-8"))
    assert report["official_test_used_for_training"] is False
    assert report["architecture_id"] == candidate_id
    print({
        "candidate": candidate_id,
        "epochs_completed": report["epochs_completed"],
        "best_validation_video_auc": report["best_validation_video_auc"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    })
"""
    ),
    code(
        """
# 8. 두 후보 Validation-only 평가 — 공식 Test 추론 0건
for candidate in CONFIG["candidates"]:
    candidate_id = candidate["id"]
    paths = CANDIDATE_PATHS[candidate_id]
    command = [
        sys.executable, "scripts/run_celebdf_deepfake.py", "evaluate",
        "--crop-manifest", str(CROP_MANIFEST), "--crop-root", str(CROP_ROOT),
        "--checkpoint", str(paths["checkpoint"]),
        "--private-scores", str(paths["validation_scores"]),
        "--metrics", str(paths["validation_metrics"]),
        "--input-size", str(CONFIG["input_size"]),
        "--batch-size", str(CONFIG["batch_size"] * 2),
        "--seed", str(CONFIG["seed"]),
        "--target-fpr", str(CONFIG["target_validation_fpr"]),
        "--validation-only",
        "--frame-counts", *[str(value) for value in CONFIG["evaluation_frame_counts"]],
        "--aggregation-methods", *CONFIG["aggregation_methods"],
        "--conditions", *CONFIG["conditions"],
    ]
    subprocess.run(command, check=True)
    metrics = json.loads(paths["validation_metrics"].read_text(encoding="utf-8"))
    assert metrics["evaluation_scope"] == "validation_only"
    assert metrics["official_test_inference_performed"] is False
    print({
        "candidate": candidate_id,
        "validation_auc": metrics["validation_video"]["roc_auc"],
        "fpr_at_recall_0_95": metrics["validation_operating_point_at_recall_0_95"]["fpr"],
        "latency_p95_ms": metrics["validation_video_latency"]["p95_ms"],
    })
"""
    ),
    code(
        """
# 9. Validation 결과만으로 승자 고정
compare_command = [
    sys.executable,
    "scripts/compare_deepfake_model_candidates.py",
]
for candidate in CONFIG["candidates"]:
    candidate_id = candidate["id"]
    compare_command.extend([
        "--candidate",
        f"{candidate_id}={CANDIDATE_PATHS[candidate_id]['validation_metrics']}",
    ])
compare_command.extend(["--output", str(COMPARISON_REPORT)])
subprocess.run(compare_command, check=True)

comparison = json.loads(COMPARISON_REPORT.read_text(encoding="utf-8"))
FROZEN_CANDIDATE = comparison["selected_candidate"]
SELECTION_FINGERPRINT = comparison["selection_fingerprint_sha256"]
assert comparison["official_test_used_for_selection"] is False
assert comparison["official_test_inference_performed_before_freeze"] is False
assert comparison["selected_candidate_frozen_before_official_test"] is True
assert len(SELECTION_FINGERPRINT) == 64
print({
    "frozen_candidate": FROZEN_CANDIDATE,
    "selection_fingerprint": SELECTION_FINGERPRINT,
    "official_test_seen_before_freeze": False,
})
"""
    ),
    code(
        """
# 10. 고정된 승자 하나만 공식 Test와 촬영 열화 평가
FINAL_RESULT = None
if RUN_FINAL_OFFICIAL_TEST_FOR_FROZEN_WINNER:
    selected = CANDIDATE_PATHS[FROZEN_CANDIDATE]
    final_private_scores = PRIVATE_SCORE_ROOT / f"{FROZEN_CANDIDATE}_official_test_scores.csv"
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "evaluate",
        "--crop-manifest", str(CROP_MANIFEST), "--crop-root", str(CROP_ROOT),
        "--checkpoint", str(selected["checkpoint"]),
        "--private-scores", str(final_private_scores),
        "--metrics", str(FINAL_METRICS),
        "--input-size", str(CONFIG["input_size"]),
        "--batch-size", str(CONFIG["batch_size"] * 2),
        "--seed", str(CONFIG["seed"]),
        "--target-fpr", str(CONFIG["target_validation_fpr"]),
        "--frame-counts", *[str(value) for value in CONFIG["evaluation_frame_counts"]],
        "--aggregation-methods", *CONFIG["aggregation_methods"],
        "--conditions", *CONFIG["conditions"],
    ], check=True)
    FINAL_RESULT = json.loads(FINAL_METRICS.read_text(encoding="utf-8"))
    assert FINAL_RESULT["architecture_id"] == FROZEN_CANDIDATE
    assert FINAL_RESULT["official_test_inference_performed"] is True
    FINAL_RESULT["selection_fingerprint_sha256"] = SELECTION_FINGERPRINT
    FINAL_METRICS.write_text(json.dumps(FINAL_RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
    print({
        "candidate": FROZEN_CANDIDATE,
        "official_test_auc": FINAL_RESULT["test_video"]["roc_auc"],
        "official_test_fpr": FINAL_RESULT["test_video"]["fpr"],
        "official_test_recall": FINAL_RESULT["test_video"]["recall"],
        "research_gate_pass": FINAL_RESULT["research_gate"]["overall_pass"],
    })
else:
    print("공식 Test를 실행하지 않았습니다. Validation 비교와 checkpoint만 저장합니다.")
"""
    ),
    code(
        """
# 11. 고정 승자 ONNX 내보내기와 CPU 연결 시험
if RUN_FINAL_OFFICIAL_TEST_FOR_FROZEN_WINNER:
    selected_checkpoint = CANDIDATE_PATHS[FROZEN_CANDIDATE]["checkpoint"]
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "export-onnx",
        "--checkpoint", str(selected_checkpoint),
        "--output", str(ONNX_MODEL),
        "--report", str(ONNX_EXPORT_REPORT),
    ], check=True)
    subprocess.run([
        sys.executable, "scripts/run_celebdf_deepfake.py", "smoke-onnx",
        "--model", str(ONNX_MODEL),
        "--crop-manifest", str(CROP_MANIFEST),
        "--crop-root", str(CROP_ROOT),
        "--report", str(ONNX_SMOKE_REPORT),
        "--input-size", str(CONFIG["input_size"]),
        "--architecture", FROZEN_CANDIDATE,
        "--normalization", CONFIG["normalization"],
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
# 12. 비식별 비교 그래프와 모델 카드
import matplotlib.pyplot as plt
import numpy as np

rows = comparison["candidates"]
names = [row["candidate_id"] for row in rows]
fpr_values = [row["validation_fpr_at_recall_0_95"] for row in rows]
robust_auc = [row["validation_robustness_macro_roc_auc"] for row in rows]
latencies = [row["validation_latency_p95_ms"] for row in rows]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(names, fpr_values, color=["#4C78A8", "#F58518"])
axes[0].set_title("Validation FPR at Recall ≥ 95%")
axes[0].set_ylabel("낮을수록 좋음")
axes[1].bar(names, robust_auc, color=["#4C78A8", "#F58518"])
axes[1].set_title("열화 Validation Macro AUC")
axes[1].set_ylabel("높을수록 좋음")
axes[2].bar(names, latencies, color=["#4C78A8", "#F58518"])
axes[2].set_title("Validation p95 추론시간")
axes[2].set_ylabel("ms, 낮을수록 좋음")
fig.suptitle(f"Frozen candidate: {FROZEN_CANDIDATE}")
fig.tight_layout()
fig.savefig(FIGURE, dpi=160, bbox_inches="tight")
plt.show()

official_lines = "- 공식 Test: 실행하지 않음"
if FINAL_RESULT is not None:
    official_lines = f'''- 공식 Test Video ROC-AUC: {FINAL_RESULT['test_video']['roc_auc']}
- 공식 Test 실제영상 FPR: {FINAL_RESULT['test_video']['fpr']}
- 공식 Test 딥페이크 Recall: {FINAL_RESULT['test_video']['recall']}
- 내부 연구 Gate 통과: {FINAL_RESULT['research_gate']['overall_pass']}'''
MODEL_CARD.write_text(f'''# EfficientNet-B4 vs Xception 연구 비교 모델 카드

- 선택 후보: {FROZEN_CANDIDATE}
- 선택 split: Validation only
- 선택 fingerprint: {SELECTION_FINGERPRINT}
- 입력: 정렬 얼굴 RGB {CONFIG['input_size']}×{CONFIG['input_size']}
- 정규화: mean/std 0.5
- 영상당 학습 프레임: {CONFIG['train_frames_per_video']}
{official_lines}
- 외부 데이터 검증: 아직 필요
- 운영 승인: 아님

두 모델은 같은 crop manifest, seed, 입력 크기, 정규화, 프레임 수, 증강, optimizer와 epoch 예산으로 비교했다. 공식 Test는 Validation으로 후보를 고정한 뒤 선택 모델 하나에만 사용했다. 원본 얼굴·개별 영상 점수·checkpoint·ONNX는 공개하지 않는다.
''', encoding="utf-8")
print({"figure": str(FIGURE), "model_card": str(MODEL_CARD)})
"""
    ),
    code(
        """
# 13. 비식별 결과 ZIP과 비공개 모델 ZIP 저장
import hashlib
import zipfile

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

SANITIZED_BUNDLE = OUTPUT_ROOT / "effb4_xception_sanitized_results.zip"
PRIVATE_MODEL_BUNDLE = OUTPUT_ROOT / "effb4_xception_private_models.zip"
sanitized_files = [INVENTORY, PREPROCESS_REPORT, COMPARISON_REPORT, FIGURE, MODEL_CARD]
for paths in CANDIDATE_PATHS.values():
    sanitized_files.extend([paths["train_report"], paths["validation_metrics"]])
if FINAL_RESULT is not None:
    sanitized_files.extend([FINAL_METRICS, ONNX_EXPORT_REPORT, ONNX_SMOKE_REPORT])

with zipfile.ZipFile(SANITIZED_BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sanitized_files:
        archive.write(path, arcname=str(path.relative_to(SANITIZED_ROOT)) if path.is_relative_to(SANITIZED_ROOT) else path.name)
with zipfile.ZipFile(PRIVATE_MODEL_BUNDLE, "w", compression=zipfile.ZIP_STORED) as archive:
    for candidate_id, paths in CANDIDATE_PATHS.items():
        archive.write(paths["checkpoint"], arcname=f"{candidate_id}/best.pt")
    if ONNX_MODEL.is_file():
        archive.write(ONNX_MODEL, arcname=f"{FROZEN_CANDIDATE}/frozen_winner.onnx")
    archive.writestr("selection_fingerprint.txt", SELECTION_FINGERPRINT + "\\n")

with zipfile.ZipFile(SANITIZED_BUNDLE) as archive:
    names = set(archive.namelist())
    assert CROP_MANIFEST.name not in names
    assert not any(name.endswith((".jpg", ".mp4", ".pt", ".onnx", ".csv")) for name in names)
with zipfile.ZipFile(PRIVATE_MODEL_BUNDLE) as archive:
    assert sum(name.endswith(".pt") for name in archive.namelist()) == 2

if IN_KAGGLE:
    shutil.rmtree(PRIVATE_ROOT)

print({
    "sanitized_results": str(SANITIZED_BUNDLE),
    "sanitized_sha256": file_sha256(SANITIZED_BUNDLE),
    "private_models": str(PRIVATE_MODEL_BUNDLE),
    "private_models_sha256": file_sha256(PRIVATE_MODEL_BUNDLE),
    "selected_candidate": FROZEN_CANDIDATE,
    "selection_fingerprint": SELECTION_FINGERPRINT,
    "warning": "private model ZIP은 공개하거나 GitHub에 올리지 마세요.",
})
"""
    ),
    markdown(
        """
## 결과를 어떻게 읽나요?

- 먼저 `validation_candidate_comparison.json`의 `selected_candidate`를 확인한다.
- 선택 근거는 `validation_fpr_at_recall_0_95`가 가장 낮은 후보이며, 동률이면 열화 AUC와 속도로 결정한다.
- 공식 Test 결과는 선택 모델 한 개에 대해서만 생성된다.
- FPR `≤ 1%`, Recall `≥ 95%`, coverage `≥ 99%`를 모두 확인한다.
- 통과하더라도 Celeb-DF 내부 연구 후보일 뿐이다. 외부 조작 방식과 실제 웹 영상 검증 전에는 운영 모델로 교체하지 않는다.
- GitHub에는 `effb4_xception_sanitized_results.zip`의 비식별 내용만 정리한다. `effb4_xception_private_models.zip`은 비공개로 보관한다.
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
