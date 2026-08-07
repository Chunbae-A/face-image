# Kaggle 무료 GPU로 딥페이크 모델 실행하기

## 결론

이번 Celeb-DF 전체 실험은 Colab보다 **Kaggle 비공개 Dataset + 무료 GPU**를 권장한다.

- 10GB 원본을 비공개 Dataset으로 한 번 등록하면 세션마다 브라우저로 다시 올리지 않아도 된다.
- 긴 실행은 `Save & Run All`로 백그라운드에서 돌릴 수 있어 브라우저를 계속 켜둘 필요가 없다.
- Kaggle은 공식 안내에서 무료 P100 GPU와 주당 약 30시간의 quota를 설명한다. 실제 quota는 수요에 따라 달라질 수 있으므로 실행 전 화면에서 남은 시간을 확인한다.
- 전체 작업을 전처리와 학습으로 나눠, 두 번째 단계가 실패해도 6,529개 얼굴 전처리를 다시 하지 않게 한다.

공식 참고: [Kaggle GPU 사용 안내](https://www.kaggle.com/docs/efficient-gpu-usage), [Kaggle CLI Notebook 안내](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [Kaggle CLI Dataset 생성 안내](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md)

## 전체 흐름

```text
Private Dataset: Celeb-DF-v2.zip
        ↓
1단계 Notebook: 얼굴 탐지·정렬
        ↓ Private Notebook Output
얼굴 crop TAR + 비공개 manifest
        ↓
2단계 Notebook: EfficientNet-B4 학습
        ↓
Validation에서 8/16/32프레임·통합 방식·기준값 선택
        ↓
공식 Test 518개 + 열화 조건 평가
        ↓
비식별 결과 ZIP + 비공개 ONNX 모델 ZIP
        ↓
3단계 Notebook: 점수 보정
        ↓
비식별 calibration JSON + reliability diagram
```

## 0. 반드시 지킬 것

- Dataset과 세 Notebook을 모두 `Private`로 유지한다.
- 원본 영상, 정렬 얼굴 crop, 영상별 점수, checkpoint와 ONNX를 공개하지 않는다.
- 비식별 집계 결과 ZIP만 GitHub Issue와 PR에 올린다.
- Celeb-DF의 Kaggle 비공개 처리 허용 여부와 InsightFace 제공 검출 가중치의 비상업 연구 조건을 직접 확인한다.
- Notebook을 공개로 전환하지 않는다. 1단계 Output에는 얼굴 crop이 들어 있다.

## 1. 전체 Celeb-DF를 Private Dataset으로 등록

Kaggle의 `Datasets` → `New Dataset`에서 승인받은 전체 파일을 올린다.

```text
파일명: Celeb-DF-v2.zip
정확한 크기: 9,952,957,051 bytes
공개 범위: Private
```

예전에 사용한 약 929MB `Celeb-real` 부분집합이 아니라 **9.95GB 전체 ZIP**이어야 한다. Kaggle이 ZIP을 자동으로 풀어도 괜찮다. 노트북은 다음 두 형태를 모두 검사한다.

- ZIP 그대로: 정확한 바이트 크기 확인
- 자동 압축 해제: 세 폴더의 영상 6,529개와 총 바이트 확인

## 2. 1단계 — 얼굴 전처리

사용할 파일: [`notebooks/celebdf_deepfake_preprocess_kaggle.ipynb`](notebooks/celebdf_deepfake_preprocess_kaggle.ipynb)

1. Kaggle `Code` → `New Notebook`을 연다.
2. 위 IPYNB를 Import한다.
3. `Add Input`에서 방금 만든 전체 Celeb-DF Private Dataset을 연결한다.
4. `Settings` → `Accelerator`에서 사용 가능한 GPU를 선택한다.
5. `Internet`을 켠다. InsightFace 검출 모델 최초 다운로드에만 필요하다.
6. 1번 셀에서 다음을 `True`로 바꾼다.

   ```python
   I_CONFIRM_CELEBDF_KAGGLE_PRIVATE_PROCESSING_IS_ALLOWED = True
   I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = True
   ```

7. 먼저 일반 `Run All`로 Smoke가 통과하는지 확인한다.
8. 문제가 없으면 `Save Version` → `Save & Run All`로 전체 실행을 남긴다.

완료 Output:

```text
celebdf_deepfake_preprocess_private.tar
celebdf_deepfake_preflight_sanitized.zip
```

첫 번째 TAR에는 얼굴 crop이 들어 있으므로 private 상태를 유지한다. 두 번째 ZIP만 비식별 사전검사 결과다.

## 3. 2단계 — 모델 학습과 공식 평가

사용할 파일: [`notebooks/celebdf_deepfake_train_kaggle.ipynb`](notebooks/celebdf_deepfake_train_kaggle.ipynb)

1. 새 Private Kaggle Notebook을 만든다.
2. 위 IPYNB를 Import한다.
3. `Add Input`에서 **1단계 Notebook의 저장된 Output**을 연결한다.
4. `Settings`에서 **GPU T4 x2**와 Internet을 켠다. Internet은 ImageNet 사전학습 가중치에 필요하다. 2026-08-07 Kaggle PyTorch 환경에서는 P100의 `sm_60` 커널이 포함되지 않아 학습이 실패했으므로 P100을 선택하지 않는다.
5. 1번 셀의 다음 값을 `True`로 바꾼다.

   ```python
   I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED = True
   ```

6. `Run All`로 경로와 GPU를 먼저 확인한다.
7. 전체 실행은 `Save Version` → `Save & Run All`로 남긴다.

모델은 Train과 Validation만으로 학습하고 다음을 Validation에서 선택한다.

- 프레임 수: 8, 16, 32
- 통합 방식: 평균, 중앙값, 위험도 상위 25% 평균
- 판정 기준값: Validation 실제 영상 FPR 목표 1%

선택이 끝난 뒤 공식 Test 518개를 평가한다.

## 4. 받아야 할 결과

```text
celebdf_deepfake_sanitized_results.zip
celebdf_deepfake_private_model.zip
```

`sanitized_results.zip`에는 다음만 들어 있다.

- 전체 집계 데이터 수와 누수 검사
- 학습 설정과 epoch별 Validation 집계
- 공식 Test와 열화 조건 집계 지표
- ROC, PR, 혼동행렬, 열화 비교 그래프
- ONNX CPU 스모크 결과
- 연구용 모델 카드

`private_model.zip`에는 PyTorch checkpoint와 API 연결용 ONNX가 들어 있다. GitHub에는 올리지 않고 로컬의 접근 제한된 모델 보관소로 내려받는다.

## 5. 나에게 알려줄 값

2단계 7번 셀 출력 전체를 복사해 보내면 된다. 핵심 필드는 다음이다.

```text
selected_frames
selected_aggregation
threshold
official_test_auc
official_test_fpr
official_test_recall
coverage
research_gate_pass
```

그리고 마지막 셀의 두 SHA-256도 함께 보낸다. 그러면 비식별 결과를 저장소 보고서와 GitHub Issue #15, Draft PR에 연결한다.

## 6. 주의할 점

- `Save & Run All`을 여러 번 누르면 동시 실행이 생겨 GPU quota가 중복 소비될 수 있다.
- 실행 중인 세션은 Kaggle `Active Events`에서 확인한다.
- 1단계가 성공하기 전에 2단계를 시작하지 않는다.
- Gate 미통과 결과도 숨기지 않는다. AUC 0.90 미만 또는 실제 영상 FPR 1% 초과면 즉시경보 모델로 승인하지 않는다.
- Gate 통과도 Celeb-DF 내부 연구 결과다. 실제 웹 영상과 한국인 대상 운영 성능은 별도 검증이 필요하다.

## 7. 3단계 — 화면용 점수 보정

사용할 파일: [`notebooks/celebdf_score_calibration_kaggle.ipynb`](notebooks/celebdf_score_calibration_kaggle.ipynb)

이 단계는 학습을 다시 하지 않는다. 새 Private Notebook에 파일을 Import하고 Input으로 다음 두 저장 결과를 연결한다.

1. `deepsogak-celebdf-preprocess`의 저장된 Output
2. `deepsogak-celebdf-train`의 저장된 Output

GPU를 켜고 `Run All`을 실행한다. 노트북은 clean 16프레임 점수만 다시 계산한 뒤 Temperature, Platt, Isotonic 보정을 비교한다. 개별 점수는 `/kaggle/temp`에서 삭제하고 다음 비식별 파일만 Output에 남긴다.

```text
deepfake_video_calibration.json
deepfake_score_calibration.png
README_score_calibration.md
deepfake_score_calibration_sanitized.zip
```

자세한 결과 해석과 API 설치 방법은 [`SCORE_CALIBRATION_RUNBOOK.md`](SCORE_CALIBRATION_RUNBOOK.md)를 따른다.
