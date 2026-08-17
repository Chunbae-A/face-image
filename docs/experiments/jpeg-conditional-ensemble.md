# JPEG 조건부 두 모델 결합 실험 실행 안내

## 이번 실험을 한 문장으로 설명하면

평소에는 빠른 EfficientNet-B4 하나만 사용하고, **JPEG 압축이 심한 영상에서만 Xception의 의견을 추가했을 때 오경고가 줄어드는지** 확인한다.

두 모델을 처음부터 다시 학습하지 않는다. 기존 두 checkpoint로 같은 Validation 프레임을 다시 채점하고, 점수 결합 비율만 비교한다. 따라서 이전 두 모델 학습보다 훨씬 가볍다.

## 실제 실행 결론

Kaggle Version 4에서 실행을 완료했고 `conditional_primary_weight_0_25`가 Validation 후보로 선택됐다.

- JPEG ROC-AUC: `0.999122 → 0.999606`
- clean 기준값 고정 Recall: `74.31% → 78.85%`
- JPEG p95 추론시간: `41.30ms → 92.24ms`
- 공식 Test 사용: 0건
- 운영 승인: `false`

따라서 “연구 후보 선택”이지 “API 배포 완료”가 아니다. 실제 웹 JPEG 품질 Gate와 외부 영상 검증 전에는 EfficientNet-B4 단독 API를 유지한다. 전체 결과는 [`reports/celebdf_jpeg_conditional_ensemble/2026-08-17`](../../reports/celebdf_jpeg_conditional_ensemble/2026-08-17)에 있다.

## 준비된 파일

- 실행 노트북: [`notebooks/celebdf_jpeg_conditional_ensemble_kaggle.ipynb`](../../notebooks/celebdf_jpeg_conditional_ensemble_kaggle.ipynb)
- 고정 설정: [`configs/deepfake/jpeg_conditional_ensemble.json`](../../configs/deepfake/jpeg_conditional_ensemble.json)
- 점수 결합 코드: [`scripts/optimize_deepfake_score_ensemble.py`](../../scripts/optimize_deepfake_score_ensemble.py)
- GitHub 작업: [Issue #35](https://github.com/Chunbae-A/face-image/issues/35)

## 사용할 비공개 Output 두 개

1. 전처리 Output의 `celebdf_deepfake_preprocess_private.tar`
2. 이전 비교 Output의 `effb4_xception_private_models.zip`

두 파일에는 얼굴 crop 또는 모델 checkpoint가 있으므로 Kaggle Notebook과 Input을 모두 `Private`으로 유지한다. GitHub에 올리지 않는다.

노트북은 두 파일이 `Add Input`으로 이미 연결돼 있으면 그대로 사용한다. CLI의 비대화형 실행에서는 `kernel-metadata.json`의 `kernel_sources`에 두 노트북을 미리 연결해야 한다. 실행 중 새 Input을 붙이는 방식은 Kaggle이 차단하므로 자동 다운로드는 대화형 편집 실행의 보조 경로로만 사용한다.

## 실행 순서

1. Kaggle에서 새 Private Notebook을 만든다.
2. `celebdf_jpeg_conditional_ensemble_kaggle.ipynb`를 Import한다.
3. `Settings`에서 GPU를 고른다.
4. Internet을 켠다. `timm` 설치와 두 비공개 Output의 자동 연결에 사용한다.
5. 첫 번째 설정 셀에서 아래 두 값을 `True`로 바꾼다.

```python
I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED = True
I_CONFIRM_PRIVATE_MODEL_OUTPUT_MAY_BE_USED = True
```

6. `Save Version` → `Save & Run All`을 한 번만 누른다.

노트북은 2026-08 Kaggle 기본 PyTorch와 Tesla T4의 호환 실패를 피하기 위해 T4 커널이 포함된 `torch 2.6.0 + torchvision 0.21.0 + CUDA 12.4` 공식 wheel을 설치한다. 본 추론 전에 작은 CUDA 연산을 실행하므로 GPU 이름만 보이고 실제 연산이 실패하는 상태를 즉시 발견한다.

CLI 자동 실행은 사용자가 비공개 Output 사용을 승인한 경우에만 생성기의 `--confirm-private-inputs` 옵션을 사용한다. GitHub에 저장하는 기본 노트북은 두 확인값을 계속 `False`로 유지한다.

학습은 하지 않지만 Validation 5조건을 두 모델로 다시 추론한다. Kaggle GPU 종류와 부하에 따라 달라지며 대략 30~70분 범위를 예상한다.

## 완료 후 결과

Kaggle Output에서 `jpeg_conditional_ensemble_sanitized_results.zip`을 받는다. ZIP에는 조건별 집계 JSON, 비교 그래프, 한국어 요약만 들어 있다. 영상 ID, 프레임 점수, 얼굴 crop, checkpoint, ONNX는 들어 있지 않다. 실행이 끝나면 노트북이 `/kaggle/temp`의 비공개 임시 파일을 삭제한다.

## 결과 해석

| 결과 | 뜻 | 다음 결정 |
|---|---|---|
| `ensemble_selected=true` | JPEG에서 보조 모델의 추가 효과가 Validation 기준을 통과 | 외부 영상 검증 후 API 조건부 라우팅 검토 |
| `ensemble_selected=false` | 섞어도 이득이 없거나 다른 조건이 나빠짐 | EfficientNet-B4 단독 유지 |

이 결과만으로 운영 배포하지 않는다. 결합이 채택되면 실제 웹 JPEG 품질 판별 규칙을 연결하고 여러 seed와 외부 데이터로 다시 확인한다. 속도를 유지하려면 이후 Xception의 장점을 EfficientNet-B4에 옮기는 지식 증류 파인튜닝을 검토한다.
