# JPEG 조건부 두 모델 결합 실험 실행 안내

## 이번 실험을 한 문장으로 설명하면

평소에는 빠른 EfficientNet-B4 하나만 사용하고, **JPEG 압축이 심한 영상에서만 Xception의 의견을 추가했을 때 오경고가 줄어드는지** 확인한다.

두 모델을 처음부터 다시 학습하지 않는다. 기존 두 checkpoint로 같은 Validation 프레임을 다시 채점하고, 점수 결합 비율만 비교한다. 따라서 이전 두 모델 학습보다 훨씬 가볍다.

## 준비된 파일

- 실행 노트북: [`notebooks/celebdf_jpeg_conditional_ensemble_kaggle.ipynb`](notebooks/celebdf_jpeg_conditional_ensemble_kaggle.ipynb)
- 고정 설정: [`configs/deepfake/jpeg_conditional_ensemble.json`](configs/deepfake/jpeg_conditional_ensemble.json)
- 점수 결합 코드: [`scripts/optimize_deepfake_score_ensemble.py`](scripts/optimize_deepfake_score_ensemble.py)
- GitHub 작업: [Issue #35](https://github.com/Chunbae-A/face-image/issues/35)

## Kaggle에 연결할 비공개 Input 두 개

1. 전처리 Output의 `celebdf_deepfake_preprocess_private.tar`
2. 이전 비교 Output의 `effb4_xception_private_models.zip`

두 파일에는 얼굴 crop 또는 모델 checkpoint가 있으므로 Kaggle Notebook과 Input을 모두 `Private`으로 유지한다. GitHub에 올리지 않는다.

## 실행 순서

1. Kaggle에서 새 Private Notebook을 만든다.
2. `celebdf_jpeg_conditional_ensemble_kaggle.ipynb`를 Import한다.
3. `Add Input`에서 위 두 비공개 Output을 연결한다.
4. `Settings`에서 `GPU T4 x2` 또는 사용 가능한 GPU를 고른다.
5. Internet을 켠다. `timm` 설치에 사용한다.
6. 첫 번째 설정 셀에서 아래 두 값을 `True`로 바꾼다.

```python
I_CONFIRM_PRIVATE_PREPROCESS_OUTPUT_MAY_BE_USED = True
I_CONFIRM_PRIVATE_MODEL_OUTPUT_MAY_BE_USED = True
```

7. `Save Version` → `Save & Run All`을 한 번만 누른다.

학습은 하지 않지만 Validation 5조건을 두 모델로 다시 추론한다. Kaggle GPU 종류와 부하에 따라 달라지며 **대략 30~70분 범위**를 예상한다. 첫 모델의 완료 시간을 보고 남은 시간을 판단하는 편이 정확하다.

## 완료 후 결과

Kaggle Output에서 다음 파일을 받는다.

```text
jpeg_conditional_ensemble_sanitized_results.zip
```

이 ZIP에는 다음 세 파일만 들어 있다.

- 조건별 집계 JSON
- 비교 그래프 PNG
- 쉬운 한국어 요약 README

영상 ID, 프레임 점수, 얼굴 crop, checkpoint, ONNX는 들어 있지 않다. 실행이 끝나면 노트북이 `/kaggle/temp`의 비공개 임시 파일을 삭제한다.

## 결과 해석

| 결과 | 뜻 | 다음 결정 |
|---|---|---|
| `ensemble_selected=true` | JPEG에서 보조 모델의 추가 효과가 Validation 기준을 통과 | 외부 영상 검증 후 API 조건부 라우팅 검토 |
| `ensemble_selected=false` | 섞어도 이득이 없거나 다른 조건이 나빠짐 | EfficientNet-B4 단독 유지 |

이 결과만으로 운영 배포하지 않는다. 결합이 채택되면 다음 단계에서 실제 웹 JPEG 품질 판별 규칙을 연결하고, 여러 seed와 외부 데이터로 다시 확인한다. 속도를 유지하려면 이후 Xception의 장점을 EfficientNet-B4에 옮기는 지식 증류 파인튜닝을 검토한다.
