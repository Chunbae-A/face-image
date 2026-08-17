# EfficientNet-B4와 Xception Kaggle 비교 실행 안내

## 최종 결과

두 모델을 같은 조건에서 실제 비교했고 기본 모델은 **EfficientNet-B4를 유지**했다. Xception은 JPEG q30에서만 더 좋았지만 전체 열화 성능과 속도는 EfficientNet-B4가 우세했다.

| Validation 지표 | EfficientNet-B4 | Xception |
|---|---:|---:|
| clean ROC-AUC | **0.9999873** | 0.99996184 |
| 열화 5조건 macro ROC-AUC | **0.99862955** | 0.99839426 |
| p95 추론시간 | **69.35ms** | 94.22ms |

고정한 EfficientNet-B4의 공식 Test ROC-AUC는 `0.99955387`이었지만 실제 영상 FPR은 `1.6854%`로 내부 목표 `1%`를 넘었다. 따라서 모델 교체나 운영 자동 차단은 승인하지 않았다. Xception의 JPEG 장점만 조건부로 결합하는 후속 검증은 [Issue #35](https://github.com/Chunbae-A/face-image/issues/35)에서 진행한다.

## 재현 파일

- 실행 노트북: [`notebooks/celebdf_effb4_xception_compare_kaggle.ipynb`](../../notebooks/celebdf_effb4_xception_compare_kaggle.ipynb)
- 고정 설정: [`configs/deepfake/effb4_xception_comparison.json`](../../configs/deepfake/effb4_xception_comparison.json)
- 결과 선택 코드: [`scripts/compare_deepfake_model_candidates.py`](../../scripts/compare_deepfake_model_candidates.py)

## 쉽게 말하면 무엇을 하나요?

```text
같은 얼굴 crop과 같은 Train/Validation 분할
        ↓
EfficientNet-B4와 Xception을 각각 학습
        ↓
공식 Test를 열지 않고 Validation 5조건 비교
        ↓
Recall 95% 이상에서 실제영상 오경고가 적은 모델 고정
        ↓
고정 모델 한 개만 공식 Test 실행
        ↓
ONNX 변환과 CPU API 연결 시험
```

이번 비교에서 두 모델은 모두 다음 값을 사용한다.

| 항목 | 고정값 |
|---|---:|
| 입력 | 얼굴 RGB `256×256` |
| 정규화 | mean/std `0.5` |
| seed | `20260808` |
| 학습 프레임 | 영상당 16개 |
| 평가 프레임 | 영상당 16개 |
| 점수 통합 | 평균 |
| epoch 예산 | 최대 8 |
| effective batch | 16 |
| optimizer | AdamW |
| 판정 기준 선택 | Validation FPR 목표 1% |

입력 크기와 정규화를 같게 정한 이유는 모델 이외의 차이를 줄이기 위해서다. DeepfakeBench 공식 설정도 Xception과 EfficientNet-B4를 같은 `256×256`, mean/std `0.5` 조건으로 비교한다. [DeepfakeBench Xception 설정](https://github.com/SCLBD/DeepfakeBench/blob/main/training/config/detector/xception.yaml), [EfficientNet-B4 설정](https://github.com/SCLBD/DeepfakeBench/blob/main/training/config/detector/efficientnetb4.yaml)

Xception은 `timm==1.0.28`의 `legacy_xception.tf_in1k`를 사용한다. `num_classes=1`로 바꿔 실제/딥페이크 점수 하나를 출력한다. [timm 모델 생성 공식 문서](https://huggingface.co/docs/timm/main/en/reference/models), [Xception 구현](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/xception.py)

## Kaggle에서 실행하는 순서

### 1. 준비물 확인

이전에 만든 다음 비공개 전처리 Output이 필요하다.

```text
celebdf_deepfake_preprocess_private.tar
```

이 TAR 안에는 얼굴 crop과 비공개 manifest가 있으므로 공개 Dataset으로 전환하거나 GitHub에 올리면 안 된다.

### 2. 노트북 가져오기

1. Kaggle에서 새 Notebook을 만든다.
2. 저장소의 `notebooks/celebdf_effb4_xception_compare_kaggle.ipynb`를 Import한다.
3. Notebook을 `Private`으로 유지한다.
4. `Add Input`에서 위 전처리 Output을 연결한다.

### 3. 실행 환경 설정

Kaggle 오른쪽 `Settings`에서 다음을 설정한다.

- Accelerator: `GPU T4 x2` 또는 사용 가능한 GPU
- Internet: `On`
- Notebook visibility: `Private`

Internet은 `timm`과 ImageNet 사전학습 가중치를 받는 데 필요하다. 아래 내용은 결과를 다시 검증할 때 사용하는 재현 절차다.

### 4. 확인값 한 개 변경

첫 번째 설정 셀에서 다음 값만 바꾼다.

```python
I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED = True
```

나머지는 첫 실행에서 변경하지 않는다.

### 5. 전체 실행

`Run All` 또는 `Save Version → Run All`을 실행한다. 첫 epoch가 끝난 뒤 출력되는 시간을 보고 남은 시간을 추정한다. GPU·Kaggle 부하에 따라 달라지므로 실행 전 완료시간을 확정하지 않는다.

노트북은 다음 안전장치를 자동 검사한다.

- 두 후보의 crop manifest SHA-256이 같음
- seed, 입력 크기, 정규화와 프레임 수가 같음
- 두 후보 비교 전에 공식 Test 추론 0건
- Validation으로 후보를 고정한 fingerprint 생성
- 공식 Test 대상 후보는 정확히 1개

## 완료 후 받을 파일

Kaggle Output에는 두 ZIP이 생긴다.

| 파일 | 용도 | 공개 가능 여부 |
|---|---|---|
| `effb4_xception_sanitized_results.zip` | 집계 지표·그래프·모델 카드 | 검토 후 GitHub 보고 가능 |
| `effb4_xception_private_models.zip` | checkpoint 2개와 선택 ONNX | **비공개, GitHub 업로드 금지** |

공유할 때는 `sanitized_results.zip`만 전달한다. 비공개 ZIP에는 얼굴 원본은 없지만 모델 가중치와 연구 산출물이 들어 있으므로 공개하지 않는다.

## 결과를 판단하는 기준

1. 먼저 Validation의 `FPR at Recall ≥ 95%`가 낮은 후보를 선택한다.
2. 동률이면 흐림·저조도·축소 등 열화 조건 평균 AUC가 높은 후보를 선택한다.
3. 다시 동률이면 p95 처리시간이 짧은 후보를 선택한다.
4. 공식 Test는 이렇게 고정된 한 후보에만 실행한다.

최종 공식 Test에서 아래 내부 Gate를 확인한다.

- 실제 영상 FPR `≤ 1%`
- 딥페이크 Recall `≥ 95%`
- 처리 coverage `≥ 99%`

통과하더라도 외부 데이터와 실제 웹 영상 검증이 남아 있으므로 바로 운영 API 모델을 교체하지 않는다.
