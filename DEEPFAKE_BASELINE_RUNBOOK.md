# Celeb-DF 딥페이크 판별 기준선 실행 안내

이 실험은 기존 ArcFace 얼굴 동일인 비교와 다르다. 영상 속 얼굴이 **실제인지, 딥페이크로 조작됐는지**를 EfficientNet-B4로 판별한다.

## 무엇을 학습하나요?

```text
Celeb-DF-v2 전체 6,529개 영상
        ↓ 영상 단위로 먼저 분할
공식 Test 518개 잠금 + 나머지 Train/Validation
        ↓
영상마다 최대 32개 프레임 선택
        ↓
얼굴 탐지·정렬 후 얼굴 crop 생성
        ↓
EfficientNet-B4가 프레임별 딥페이크 점수 학습
        ↓
Validation에서 8/16/32프레임과 평균/중앙값/상위평균 비교
        ↓
선택을 끝낸 뒤 공식 Test와 촬영 열화 조건 평가
```

내부 라벨은 `실제=0`, `딥페이크=1`이다. 공식 테스트 파일의 표기는 `1=실제`, `0=가짜`로 반대이므로 코드가 이를 변환하고 경로의 종류와 대조한다.

## 왜 이렇게 나누나요?

- 공식 테스트 518개를 학습이나 기준값 결정에 사용하면 시험 문제를 미리 본 것과 같아진다.
- 프레임을 먼저 뽑고 나누면 같은 영상 장면이 Train과 Validation 양쪽에 섞일 수 있다.
- 따라서 **영상을 먼저 나눈 뒤** 프레임을 추출한다.
- Celeb-real과 Celeb-synthesis는 첫 번째 ID인 원본 대상 인물·영상 맥락을 그룹으로 묶어 내부 Train/Validation에 섞이지 않게 한다.
- 두 번째 기증 인물 ID는 여러 대상 조합에 반복돼 완전 분리할 수 없으므로 교집합 수를 한계로 공개한다.
- 공식 테스트는 데이터셋 제작자가 고정한 목록이라 학습 후보와 인물 ID가 일부 겹칠 수 있다. 이 교집합은 결과에 관측값으로 공개한다.

## 권장 실행 방법

전체 6,529개 실험은 **Kaggle 무료 GPU 2단계 실행**을 권장한다. 1단계 전처리 Output을 비공개로 저장한 뒤 2단계 학습에 연결하므로 세션 실패 시 반복 범위가 작다. 클릭 순서는 [`DEEPFAKE_KAGGLE_RUNBOOK.md`](DEEPFAKE_KAGGLE_RUNBOOK.md)에 있다.

- 1단계: [`notebooks/celebdf_deepfake_preprocess_kaggle.ipynb`](notebooks/celebdf_deepfake_preprocess_kaggle.ipynb)
- 2단계: [`notebooks/celebdf_deepfake_train_kaggle.ipynb`](notebooks/celebdf_deepfake_train_kaggle.ipynb)
- Colab 대안: [`notebooks/celebdf_efficientnet_b4_colab.ipynb`](notebooks/celebdf_efficientnet_b4_colab.ipynb)

Kaggle에서는 전체 ZIP을 Private Dataset으로 연결한다. Colab을 사용할 때만 Google Drive의 다음 경로를 쓴다.

준비물은 Google Drive의 다음 파일 하나다.

```text
/content/drive/MyDrive/Celeb-DF-v2.zip
정확한 크기: 9,952,957,051 bytes
```

노트북 설정에서 다음 두 확인값은 이용 조건을 직접 확인한 경우에만 켠다.

```text
I_CONFIRM_CELEBDF_CLOUD_PROCESSING_IS_ALLOWED = True
I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = True
```

InsightFace의 제공 사전학습 검출 가중치는 비상업 연구 조건이다. 해커톤 연구 검증에는 사용할 수 있지만 상용 서비스에 그대로 넣는 모델로 승인하지 않는다.

## Colab 세션 종료에 대비한 저장 방식

- 원본 ZIP은 Drive에 그대로 둔다.
- 압축을 푼 영상과 작업용 얼굴 crop은 Colab `/content`에서 처리한다.
- 얼굴 crop 전처리가 끝나면 하나의 TAR 캐시로 Drive에 복사한다.
- 학습 checkpoint, ONNX, 개인별 frame score는 Drive의 비공개 실험 폴더에 저장한다.
- GitHub에는 실제 영상, 얼굴 crop, 파일명/인물별 점수, checkpoint, ONNX를 올리지 않는다.

세션이 끝난 뒤에도 TAR 캐시와 checkpoint가 있으면 반복 작업을 줄일 수 있다. 다만 전처리 도중 세션이 종료되면 해당 전처리는 다시 실행해야 한다.

## 모델 선택 규칙

공식 Test를 보기 전에 Validation에서 다음을 정한다.

1. 영상당 프레임 수: `8`, `16`, `32`
2. 프레임 점수 통합: `평균`, `중앙값`, `딥페이크 점수가 높은 상위 25% 평균`
3. 판정 기준값: 실제 영상을 가짜로 잘못 경보하는 비율이 Validation에서 최대 1%가 되도록 선택

동률이면 계산량이 적은 프레임 수와 단순한 통합 방식을 선택한다. 선택이 끝난 후에만 공식 Test를 평가한다.

## 결과를 읽는 법

| 지표 | 쉬운 뜻 | 초기 연구 기준 |
|---|---|---:|
| Video ROC-AUC | 실제와 가짜를 전반적으로 얼마나 잘 줄 세우는지 | 0.90 이상 |
| FPR | 실제 영상을 가짜라고 잘못 경보한 비율 | 1% 이하 |
| FNR | 딥페이크를 실제라고 놓친 비율 | 낮을수록 좋음 |
| Recall | 딥페이크를 찾아낸 비율 | 높을수록 좋음 |
| AP | 가짜 후보 우선순위의 품질 | 높을수록 좋음 |
| p95 | 느린 영상 5% 경계의 처리시간 | 보고 후 API 설계 |

두 Gate를 모두 통과해도 Celeb-DF 내부 연구 기준선을 통과한 것이다. 한국인, 최신 생성 방식, 웹 재압축 영상에서 바로 운영 승인됐다는 뜻은 아니다.

## 로컬 명령 구조

노트북은 아래 명령을 순서대로 호출한다.

```bash
# 1. ZIP 목록 검사, 공식 Test 잠금, Train/Validation 분할
python scripts/celebdf_deepfake.py inventory /path/Celeb-DF-v2.zip \
  --manifest /private/celebdf_manifest.csv \
  --summary /private/inventory.json

# 2. 영상 추출
python scripts/celebdf_deepfake.py extract /path/Celeb-DF-v2.zip \
  --manifest /private/celebdf_manifest.csv \
  --output /private/videos

# 3. 얼굴 탐지·정렬 crop 생성
python scripts/run_celebdf_deepfake.py preprocess \
  --manifest /private/celebdf_manifest.csv \
  --video-root /private/videos \
  --crop-root /private/crops \
  --crop-manifest /private/crops.csv \
  --rejects /private/rejects.csv \
  --run-report /private/preprocess.json \
  --accept-noncommercial-detector-license

# 4. EfficientNet-B4 학습
python scripts/run_celebdf_deepfake.py train \
  --crop-manifest /private/crops.csv \
  --crop-root /private/crops \
  --checkpoint /private/efficientnet_b4.pt \
  --train-report /private/train.json \
  --require-cuda

# 5. Validation 선택 후 공식 Test·열화 평가
python scripts/run_celebdf_deepfake.py evaluate \
  --crop-manifest /private/crops.csv \
  --crop-root /private/crops \
  --checkpoint /private/efficientnet_b4.pt \
  --private-scores /private/frame_scores.csv \
  --metrics /private/aggregate_metrics.json

# 6. API 연결용 ONNX 내보내기와 CPU 스모크
python scripts/run_celebdf_deepfake.py export-onnx \
  --checkpoint /private/efficientnet_b4.pt \
  --output /private/efficientnet_b4.onnx \
  --report /private/onnx_export.json

python scripts/run_celebdf_deepfake.py smoke-onnx \
  --model /private/efficientnet_b4.onnx \
  --export-report /private/onnx_export.json \
  --crop-manifest /private/crops.csv \
  --crop-root /private/crops \
  --report /private/onnx_cpu_smoke.json
```

## 아직 완료로 말하면 안 되는 것

코드와 노트북이 준비된 것만으로 정확도는 나오지 않는다. Colab에서 전체 학습·공식 Test를 마치고 `aggregate_metrics.json`의 숫자를 검토한 뒤 Issue #15와 PR에 집계 결과를 추가해야 실험 완료다.
