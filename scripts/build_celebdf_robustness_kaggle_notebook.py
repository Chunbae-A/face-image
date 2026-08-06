#!/usr/bin/env python3
"""Issue #6 촬영 열화 강건성 평가 Kaggle 노트북을 생성한다."""

from __future__ import annotations

import copy
import json

from build_celebdf_colab_notebook import ROOT, code, markdown
from build_celebdf_robustness_colab_notebook import (
    CELLS as COLAB_CELLS,
    repository_bootstrap_cell as colab_repository_bootstrap_cell,
)


OUTPUT = ROOT / "notebooks" / "celebdf_arcface_robustness_kaggle.ipynb"


def cloned_colab_cell(index: int, *replacements: tuple[str, str]) -> dict[str, object]:
    """검증된 공통 실험 셀을 복제하고 환경 표현만 바꾼다."""
    cell = copy.deepcopy(COLAB_CELLS[index])
    source = "".join(cell["source"])
    for old, new in replacements:
        source = source.replace(old, new)
    cell["source"] = source.splitlines(keepends=True)
    return cell


def repository_bootstrap_cell() -> dict[str, object]:
    """Colab과 동일한 실행 코드를 Kaggle 작업 공간에 복원한다."""
    cell = copy.deepcopy(colab_repository_bootstrap_cell())
    source = "".join(cell["source"])
    replacements = (
        ("#@title 3. 실행 코드 준비 — GitHub 권한이 필요 없는 내장 코드가 기본값", "# 3. 실행 코드 준비 — GitHub 권한이 필요 없는 내장 코드가 기본값"),
        ("IN_HOSTED_COLAB", "IN_KAGGLE"),
        ('Path("/content/face-image")', 'Path("/kaggle/temp/face-image")'),
    )
    for old, new in replacements:
        source = source.replace(old, new)
    cell["source"] = source.splitlines(keepends=True)
    return cell


CELLS = [
    markdown(
        """
# 얼굴가드 촬영 열화 강건성 평가 — Kaggle 무료 GPU

쉽게 말하면 **깨끗한 영상에서 등록한 얼굴이 압축·흐림·어두움·저해상도에서도 같은 사람으로 인식되는지** 확인하는 노트북이다.

## 실행 전 딱 세 가지

1. 오른쪽 `Settings`에서 `Accelerator`를 `GPU P100`으로 선택한다.
2. 오른쪽 `Input`에서 승인받은 929MB ZIP을 담은 **비공개 Kaggle Dataset**을 연결한다.
3. `Internet`을 켠 뒤 아래 셀을 위에서부터 실행한다. 모델 파일 최초 다운로드에만 필요하다.

원본 영상·얼굴 이미지·사람별 점수·임베딩은 `/kaggle/temp`에서만 처리한다. 최종 `/kaggle/working`에는 사람을 식별할 수 없는 집계 결과 ZIP만 남긴다. 이 실험은 얼굴 동일인 검증이며 딥페이크 탐지 정확도가 아니다.
"""
    ),
    code(
        """
# 1. 실행 설정과 이용 조건 확인
REPO_URL = "https://github.com/Chunbae-A/face-image.git"
BRANCH = "exp/6-faceguard-robustness"
CODE_SOURCE = "embedded"  # "embedded" 권장, 필요하면 "github"

# 비워두면 /kaggle/input 아래에서 정확한 크기의 Celeb-DF-v2.zip을 자동으로 찾는다.
SOURCE_ZIP_PATH = ""
EXPECTED_SOURCE_ZIP_BYTES = 928989923
# Kaggle Dataset이 ZIP을 자동으로 풀었을 때 검증할 590개 MP4의 정확한 합계다.
EXPECTED_EXTRACTED_VIDEO_BYTES = 946501150

# Celeb-DF 이용 조건에서 Kaggle 비공개 데이터 처리도 허용되는지 확인한 경우만 True.
I_CONFIRM_CELEBDF_KAGGLE_PRIVATE_PROCESSING_IS_ALLOWED = False
# InsightFace buffalo_l 가중치는 비상업 연구 전용이다.
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = False

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
RUN_SMOKE_BEFORE_FULL = True

import os
from pathlib import Path
import sys

IN_KAGGLE = Path("/kaggle").exists() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if IN_KAGGLE and not I_CONFIRM_CELEBDF_KAGGLE_PRIVATE_PROCESSING_IS_ALLOWED:
    raise PermissionError("Celeb-DF의 Kaggle 비공개 처리 허용 여부를 먼저 확인하세요.")
if not I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE:
    raise PermissionError("InsightFace 비상업 연구용 가중치 조건을 확인하세요.")
if EXPECTED_SOURCE_ZIP_BYTES <= 0:
    raise ValueError("승인 ZIP의 정확한 바이트를 입력해야 합니다.")

print({
    "kaggle": IN_KAGGLE,
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

Kaggle 오른쪽 설정에서 GPU와 Internet을 켠다. 이 노트북은 민감한 중간 산출물을 저장 결과에 포함하지 않도록 `/kaggle/temp`와 `/kaggle/working`을 분리한다.
"""
    ),
    cloned_colab_cell(
        3,
        ("#@title 2. 라이브러리 설치", "# 2. 라이브러리 설치"),
    ),
    repository_bootstrap_cell(),
    code(
        """
# 4. 비공개 Kaggle Dataset 원본 확인
import json
import shutil
import zipfile

# /kaggle/temp는 Save Version 결과에 포함하지 않는 민감정보 임시 처리 영역이다.
WORK_ROOT = Path("/kaggle/temp/celebdf_robustness") if IN_KAGGLE else REPO_DIR / "outputs" / "celebdf_robustness"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

input_mode = "zip"
source_input = None
runtime_zip = None

if IN_KAGGLE:
    if SOURCE_ZIP_PATH.strip():
        zip_candidates = [Path(SOURCE_ZIP_PATH).expanduser()]
    else:
        zip_candidates = sorted(Path("/kaggle/input").rglob("Celeb-DF-v2.zip"))
    exact_zips = [
        path for path in zip_candidates
        if path.is_file() and path.stat().st_size == EXPECTED_SOURCE_ZIP_BYTES
    ]

    if len(exact_zips) == 1:
        source_input = exact_zips[0]
        runtime_zip = exact_zips[0]
    elif exact_zips:
        raise FileExistsError(f"정확한 Celeb-DF ZIP이 여러 개입니다: {exact_zips}")
    else:
        # Kaggle은 업로드한 ZIP을 자동으로 풀 수 있다. 이 경우 590개 파일 수와
        # 정확한 총 바이트를 확인한 뒤 /kaggle/temp에 저장 방식 ZIP을 재구성한다.
        celeb_real_dirs = sorted({
            path.parent
            for path in Path("/kaggle/input").rglob("Celeb-real/*.mp4")
            if path.is_file()
        })
        exact_dirs = []
        for directory in celeb_real_dirs:
            videos = sorted(directory.glob("*.mp4"))
            total_bytes = sum(path.stat().st_size for path in videos)
            if len(videos) == 590 and total_bytes == EXPECTED_EXTRACTED_VIDEO_BYTES:
                exact_dirs.append((directory, videos))
        if len(exact_dirs) != 1:
            found = [
                {
                    "directory": str(directory),
                    "video_count": len(videos),
                    "total_bytes": sum(path.stat().st_size for path in videos),
                }
                for directory, videos in exact_dirs
            ]
            raise FileNotFoundError(
                "정확한 Celeb-real MP4 590개 묶음 하나를 찾지 못했습니다. "
                f"Kaggle Input 연결을 확인하세요. 일치={found}"
            )

        celeb_real_dir, videos = exact_dirs[0]
        source_input = celeb_real_dir
        input_mode = "kaggle_auto_extracted_mp4"
        repacked_dir = WORK_ROOT / "source_repacked"
        repacked_dir.mkdir(parents=True, exist_ok=True)
        runtime_zip = repacked_dir / "Celeb-DF-v2.zip"
        partial_zip = runtime_zip.with_suffix(".zip.partial")
        if not runtime_zip.exists():
            partial_zip.unlink(missing_ok=True)
            with zipfile.ZipFile(partial_zip, "w", compression=zipfile.ZIP_STORED) as archive:
                for video in videos:
                    archive.write(video, arcname=f"Celeb-real/{video.name}")
            partial_zip.replace(runtime_zip)

        with zipfile.ZipFile(runtime_zip) as archive:
            repacked_members = [
                item for item in archive.infolist()
                if not item.is_dir() and item.filename.startswith("Celeb-real/")
            ]
        if len(repacked_members) != 590:
            raise IOError(f"재구성 ZIP 영상 수가 다릅니다: {len(repacked_members)} != 590")
else:
    source_input = Path(SOURCE_ZIP_PATH).expanduser()
    if not source_input.is_file():
        raise FileNotFoundError(f"Celeb-DF ZIP을 찾지 못했습니다: {source_input}")
    if source_input.stat().st_size != EXPECTED_SOURCE_ZIP_BYTES:
        raise IOError(
            f"Celeb-DF ZIP 크기가 다릅니다: {source_input.stat().st_size} "
            f"!= {EXPECTED_SOURCE_ZIP_BYTES}"
        )
    runtime_zip = source_input

assert source_input is not None
assert runtime_zip is not None

VIDEO_ROOT = WORK_ROOT / "videos"
MANIFEST = WORK_ROOT / "celeb_real_manifest.csv"
INVENTORY_JSON = WORK_ROOT / "celeb_real_inventory.json"
CONDITION_ROOT = WORK_ROOT / "conditions"
SANITIZED_ROOT = WORK_ROOT / "sanitized"
for path in (VIDEO_ROOT, CONDITION_ROOT, SANITIZED_ROOT):
    path.mkdir(parents=True, exist_ok=True)

print({
    "input_mode": input_mode,
    "source_input": str(source_input),
    "runtime_zip_gb": round(runtime_zip.stat().st_size / 1e9, 3),
    "private_work_root": str(WORK_ROOT),
    "runtime_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 2),
})
"""
    ),
    cloned_colab_cell(6, ("#@title", "#")),
    code(
        """
# 6. GPU와 ONNX Runtime 확인
import onnxruntime as ort

providers = ort.get_available_providers()
print({"onnxruntime": ort.__version__, "providers": providers})
if IN_KAGGLE and "CUDAExecutionProvider" not in providers:
    raise RuntimeError(
        "Kaggle GPU가 연결되지 않았습니다. 오른쪽 Settings에서 Accelerator를 GPU로 바꾸고 세션을 재시작하세요."
    )
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
))
"""
    ),
    cloned_colab_cell(
        8,
        ("#@title", "#"),
        ('"embeddings_stay_in_colab_runtime": True', '"embeddings_stay_in_kaggle_temp": True'),
    ),
    cloned_colab_cell(9, ("#@title", "#")),
    cloned_colab_cell(10, ("#@title", "#")),
    cloned_colab_cell(11, ("#@title", "#")),
    code(
        """
# 11. 얼굴 없는 집계 결과만 Kaggle Output에 저장
import hashlib
import zipfile

RUNTIME_CONFIG = SANITIZED_ROOT / "robustness_runtime_config.json"
RUNTIME_CONFIG.write_text(json.dumps({
    "environment": "kaggle",
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

temporary_bundle = SANITIZED_ROOT / "celebdf_robustness_results.zip"
allowed_artifacts = (AUDIT_JSON, METRICS_CSV, SUMMARY_CSV, FIGURE_PNG, RUNTIME_CONFIG)
with zipfile.ZipFile(temporary_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in allowed_artifacts:
        archive.write(path, arcname=path.name)

with zipfile.ZipFile(temporary_bundle) as archive:
    names = set(archive.namelist())
    if names != {path.name for path in allowed_artifacts}:
        raise AssertionError(f"결과 ZIP 허용 목록이 다릅니다: {sorted(names)}")
    if any(name.endswith(".npz") for name in names):
        raise AssertionError("결과 ZIP에 임베딩이 포함되었습니다.")

RESULT_BUNDLE = Path("/kaggle/working/celebdf_robustness_results.zip") if IN_KAGGLE else temporary_bundle
if IN_KAGGLE:
    shutil.copy2(temporary_bundle, RESULT_BUNDLE)
bundle_sha256 = hashlib.sha256(RESULT_BUNDLE.read_bytes()).hexdigest()

# 원본에서 파생된 영상, ID manifest, 실패 ID, 임베딩을 모두 지운다.
if IN_KAGGLE:
    shutil.rmtree(WORK_ROOT)
    forbidden_suffixes = {".mp4", ".npz"}
    leaked = [
        str(path) for path in Path("/kaggle/working").rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if leaked:
        raise AssertionError(f"Kaggle Output에 민감한 중간 파일이 남았습니다: {leaked}")

print({
    "result_bundle": str(RESULT_BUNDLE),
    "bundle_sha256": bundle_sha256,
    "size_mb": round(RESULT_BUNDLE.stat().st_size / 1e6, 2),
    "private_work_root_deleted": IN_KAGGLE,
    "raw_or_embedding_files_in_bundle": False,
})
"""
    ),
    markdown(
        """
## 완료 후 해야 할 일

1. 마지막 셀에 `private_work_root_deleted: True`가 표시됐는지 확인한다.
2. 오른쪽 `Output` 또는 실행 결과에서 `celebdf_robustness_results.zip`만 내려받는다.
3. 원본 영상이나 `.npz` 임베딩은 GitHub에 올리지 않는다.

결과는 얼굴 검출 성공률 → 깨끗한 기준값의 TAR/FAR → 조건별 보정 결과 순서로 읽는다. Celeb-real 결과는 한국인 얼굴·실제 휴대전화·운영 트래픽의 성능을 대신하지 않는다.
"""
    ),
]


def build_notebook() -> dict[str, object]:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
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
