# FaceGuard 스크립트

## 실험 사전 검사

```bash
# CPU/RAM/GPU/디스크 inventory
python3 scripts/faceguard_plan.py env --path .

# 다운로드 소요 시간과 저장공간 추정
python3 scripts/faceguard_plan.py download --size-gb 100 --efficiency 0.8
python3 scripts/faceguard_plan.py storage \
  --compressed-gb 100 --unpacked-gb 150 --preprocessed-gb 30 \
  --images 1000000

# subject/source/hash 누수와 Test 증강 금지 검사
python3 scripts/validate_faceguard_manifest.py examples/faceguard_manifest.csv
```

## Celeb-DF-v2 Celeb-real baseline

```bash
# ZIP을 풀지 않고 590개 영상 manifest 생성
python3 scripts/celebdf_faceguard.py inventory /path/to/Celeb-DF-v2.zip \
  --manifest outputs/celebdf_faceguard/celeb_real_manifest.csv \
  --summary outputs/celebdf_faceguard/celeb_real_inventory.json

# Celeb-real 590개만 안전하게 추출
python3 scripts/celebdf_faceguard.py extract /path/to/Celeb-DF-v2.zip \
  --manifest outputs/celebdf_faceguard/celeb_real_manifest.csv \
  --output outputs/celebdf_faceguard/videos --mode full

# 영상별 ArcFace 임베딩 평가
python3 scripts/celebdf_faceguard.py evaluate \
  --embeddings outputs/celebdf_faceguard/results/celeb_real_video_embeddings.npz \
  --output outputs/celebdf_faceguard/results/celeb_real_arcface_metrics.json
```

GPU 추론은 `notebooks/celebdf_arcface_full_colab.ipynb` 또는 `scripts/run_celebdf_arcface.py`를 사용한다. smoke 2개 영상은 환경 확인용이며, 동일 checkpoint에서 590개 전체 실행을 이어간다.

## Celeb-DF-v2 딥페이크 판별 기준선

기존 ArcFace 실험은 같은 사람인지 비교하는 기능이다. 실제/딥페이크 판별은 아래 별도 파이프라인으로 실행한다.

```bash
# 전체 6,529개 목록 확인, 공식 Test 518개 잠금, Train/Validation 분할
python3 scripts/celebdf_deepfake.py inventory /path/to/Celeb-DF-v2.zip \
  --manifest outputs/celebdf_deepfake/private_manifest.csv \
  --summary outputs/celebdf_deepfake/inventory_aggregate.json

# GPU 얼굴 전처리, EfficientNet-B4 학습·평가·ONNX 내보내기
python3 scripts/run_celebdf_deepfake.py --help
```

내부 라벨은 `실제=0`, `딥페이크=1`이다. 공식 목록의 반대 표기를 자동 변환하고 경로와 대조한다. 8/16/32프레임과 평균/중앙값/상위평균은 Validation에서만 선택하고, 선택 후 공식 Test를 평가한다. 자세한 순서와 결과 해석은 [딥페이크 기준선 실행 안내](../docs/experiments/deepfake-baseline.md)에 있다.

## 딥페이크 모델 고도화 계획 검사

Xception, SBI, hard negative 보강, FTCN을 비교하기 전에 데이터 누수 방지와 공정 비교 규칙이 유지되는지 확인한다.

```bash
python3 scripts/validate_deepfake_model_improvement_plan.py
```

검사 대상은 [`configs/deepfake/model_improvement_plan.json`](../configs/deepfake/model_improvement_plan.json)이다. 공식 Test를 학습·기준값 선택에 사용하거나 Test 오류를 hard negative로 되돌리는 설정, 단계 의존성 오류와 비공개 산출물 공개 설정을 발견하면 실패한다. 전체 실행 순서는 [딥페이크 모델 고도화 계획](../docs/experiments/model-improvement.md)에 있다.

## EfficientNet-B4와 Xception 공정 비교

아래 생성기는 두 모델을 같은 `256×256`, 정규화 `0.5`, seed·프레임·학습 예산으로 실행하는 비공개 Kaggle 노트북을 만든다.

```bash
python3 scripts/build_effb4_xception_comparison_kaggle_notebook.py
```

노트북은 모든 후보를 Validation-only로 먼저 평가한 뒤 [`compare_deepfake_model_candidates.py`](compare_deepfake_model_candidates.py)가 후보 하나를 고정한다. 그 후보에만 공식 Test와 ONNX 변환을 실행한다. 클릭 순서는 [EfficientNet-B4·Xception 비교](../docs/experiments/xception-comparison.md)에 있다.

## JPEG 조건부 두 모델 결합

전체 비교에서 Xception이 JPEG q30 조건에서만 우세했기 때문에, 아래 생성기는 기존 두 checkpoint를 다시 학습하지 않고 Validation 프레임 점수만 조건부로 결합하는 비공개 Kaggle 노트북을 만든다.

```bash
python3 scripts/build_jpeg_conditional_ensemble_kaggle_notebook.py
```

[`optimize_deepfake_score_ensemble.py`](optimize_deepfake_score_ensemble.py)는 두 CSV의 영상·프레임 키와 라벨이 정확히 같고 split이 Validation일 때만 실행한다. EfficientNet-B4 단독, 항상 결합, JPEG 조건부 결합을 비교하고 식별자가 없는 집계 JSON만 저장한다. 실행 순서는 [JPEG 조건부 두 모델 결합](../docs/experiments/jpeg-conditional-ensemble.md)에 있다.

## 노트북 재생성

```bash
python3 scripts/build_celebdf_colab_notebook.py
python3 scripts/build_celebdf_audit_colab_notebook.py
python3 scripts/build_celebdf_robustness_colab_notebook.py
python3 scripts/build_celebdf_robustness_kaggle_notebook.py
python3 scripts/build_celebdf_deepfake_colab_notebook.py
python3 scripts/build_celebdf_deepfake_kaggle_notebooks.py
python3 scripts/build_effb4_xception_comparison_kaggle_notebook.py
python3 scripts/build_jpeg_conditional_ensemble_kaggle_notebook.py
```

노트북은 실행 스크립트를 내장하므로 생성 후 ArcFace와 딥페이크 관련 원본 스크립트의 변경이 정확히 포함됐는지 테스트로 검증한다.

## Baseline 감사

`audit_celebdf_baseline.py`는 frame 수별 NPZ를 runtime에서 읽고 다중 seed·reference 지표와 누수 검사를 생성한다. 출력에는 subject/video ID 대신 fingerprint와 reject reason 집계만 남긴다.

```bash
python3 scripts/audit_celebdf_baseline.py \
  --embedding-run 1=/trusted/frames_1/video_embeddings.npz \
  --embedding-run 5=/trusted/frames_5/video_embeddings.npz \
  --embedding-run 10=/trusted/frames_10/video_embeddings.npz \
  --run-report 1=/trusted/frames_1/run.json \
  --run-report 5=/trusted/frames_5/run.json \
  --run-report 10=/trusted/frames_10/run.json \
  --output-dir outputs/celebdf_baseline_audit/sanitized
```

## 촬영 열화 강건성 평가

`run_celebdf_arcface.py --input-condition`은 깨끗한 영상, JPEG 압축, 흐림, 어두움, 저해상도, 복합 열화 중 하나를 얼굴 검출 전에 적용한다. 변환은 매번 같은 결과가 나오는 결정론적 처리이며 프레임이나 얼굴 crop을 저장하지 않는다.

`audit_celebdf_robustness.py`는 깨끗한 조건의 등록 임베딩과 모든 조건에서 공통으로 성공한 query만 사용한다. 아래처럼 여섯 조건의 NPZ와 실행 보고서를 전달한다. NPZ·reject ID는 신뢰된 runtime에만 두고 출력 디렉터리에는 집계값만 생성한다.

```bash
python3 scripts/audit_celebdf_robustness.py \
  --embedding-run clean=/trusted/clean/video_embeddings.npz \
  --embedding-run jpeg_q30=/trusted/jpeg_q30/video_embeddings.npz \
  --embedding-run gaussian_blur_sigma2=/trusted/gaussian_blur_sigma2/video_embeddings.npz \
  --embedding-run low_light_gamma2=/trusted/low_light_gamma2/video_embeddings.npz \
  --embedding-run downscale_0_25=/trusted/downscale_0_25/video_embeddings.npz \
  --embedding-run combined_mobile_stress=/trusted/combined_mobile_stress/video_embeddings.npz \
  --run-report clean=/trusted/clean/run.json \
  --run-report jpeg_q30=/trusted/jpeg_q30/run.json \
  --run-report gaussian_blur_sigma2=/trusted/gaussian_blur_sigma2/run.json \
  --run-report low_light_gamma2=/trusted/low_light_gamma2/run.json \
  --run-report downscale_0_25=/trusted/downscale_0_25/run.json \
  --run-report combined_mobile_stress=/trusted/combined_mobile_stress/run.json \
  --output-dir outputs/celebdf_robustness/sanitized
```
