# JPEG 압축 조건부 EfficientNet-B4·Xception 결합 결과

## 한 줄 결론

Celeb-DF Validation의 강한 JPEG 압축 조건에서는 EfficientNet-B4 25%와 Xception 75%의 logit을 결합하는 후보가 선택됐다. JPEG ROC-AUC와 고정 기준 Recall은 좋아졌지만 추론시간이 두 배 이상 늘고 외부 웹 영상 검증이 없으므로 **연구 후보로만 기록하고 현재 API 기본 모델은 바꾸지 않는다.**

## 왜 실험했나요?

앞선 공정 비교에서 EfficientNet-B4는 전체 열화 성능과 속도가 Xception보다 좋았지만, JPEG q30에서만 Xception의 ROC-AUC가 더 높았다. 모든 요청에 느린 두 모델을 실행하지 않고 JPEG 압축이 강한 입력에만 Xception을 보조로 쓰면 약점을 줄일 수 있는지 확인했다.

## 무엇을 비교했나요?

- 데이터 선택: 공식 Test를 열지 않은 Celeb-DF Validation 836개 영상
- 같은 입력으로 짝지은 점수: 66,880개 프레임
- 영상 점수: 영상당 16프레임 평균
- 조건: clean, JPEG q30, 흐림, 저조도, 25% 축소
- 후보: EfficientNet-B4 단독, 모든 조건 고정 결합, JPEG에서만 조건부 결합
- 결합 비율: EfficientNet-B4 가중치 0%, 25%, 50%, 75%, 100%
- 선택 원칙: clean FPR을 악화시키지 않고 JPEG AUC 또는 Recall 95% 지점 FPR을 개선하며 다른 열화 조건 AUC를 악화시키지 않는 후보

공식 Test는 결합 비율이나 기준값 선택에 사용하지 않았고 실제 추론도 하지 않았다.

## 선택 결과

선택 정책은 `conditional_primary_weight_0_25`다. 평소에는 EfficientNet-B4만 실행하고, 강한 JPEG 압축으로 판정된 입력에서만 EfficientNet-B4 25%와 Xception 75%를 logit 공간에서 결합하는 연구 후보라는 뜻이다.

| JPEG q30 Validation | EfficientNet-B4 단독 | 조건부 결합 | 변화 |
|---|---:|---:|---:|
| ROC-AUC | 0.999122 | **0.999606** | **+0.000483** |
| clean 기준값 고정 Recall | 74.31% | **78.85%** | **+4.53%p** |
| clean 기준값 고정 FPR | 0.00% | 0.00% | 동일 |
| Recall 95% 지점 FPR | 0.00% | 0.00% | 동일 |
| p95 추론시간 | **41.30ms** | 92.24ms | +50.94ms, 약 2.23배 |

조건부 정책은 JPEG 외 조건에서 EfficientNet-B4 단독 경로를 그대로 사용하므로 clean·흐림·저조도·축소 AUC 회귀는 0이었다. Validation 5조건 중 보조 모델 경로 비중 20%는 실험 격자 비율이며 실제 웹 트래픽 비율이 아니다.

## 서비스에는 어떻게 적용하나요?

지금 API에는 적용하지 않는다. 다음 세 조건을 먼저 충족해야 한다.

1. 실제 웹 이미지·영상에서 JPEG q30 수준을 판별하는 품질 Gate를 별도 검증한다.
2. Celeb-DF 밖의 실제 영상과 여러 seed에서 개선이 반복되는지 확인한다.
3. 두 모델 실행으로 늘어난 p95 지연과 비용을 데모·운영 허용 범위와 비교한다.

외부 검증까지 통과하면 기본 경로는 EfficientNet-B4, 강한 JPEG 입력만 Xception 보조 결합, Xception 실패 시 EfficientNet-B4 결과와 사람 검토 경고를 반환하는 방식으로 API를 확장한다. 자동 차단·자동 신고는 여전히 허용하지 않는다.

## 재현성과 개인정보

- Kaggle Private Notebook: [deepsogak-jpeg-conditional-ensemble](https://www.kaggle.com/code/hywznn/deepsogak-jpeg-conditional-ensemble)
- 설정 SHA-256: `79cea43d527eedd7814cb327b2ee1abff13652bddb1b7cf297ec2abb37418fc7`
- 입력 짝 fingerprint: `c311cca71977bab75b7b57c21fd3af40b4f4f21cf612768cfb461a2d613d4216`
- 선택 fingerprint: `102537392e7f0b82f156336476d206aeabc008dcace89b9810158e6fedf848fc`
- 결과 JSON: [`jpeg_conditional_ensemble_validation.json`](jpeg_conditional_ensemble_validation.json)
- 비교 그래프: [`jpeg_conditional_ensemble_validation.png`](jpeg_conditional_ensemble_validation.png)

공개 결과에는 영상 ID, 프레임별 점수, 얼굴 crop, checkpoint, ONNX가 포함되지 않았다. 비공개 입력과 중간 점수는 Kaggle `/kaggle/temp`에서만 사용하고 실행 종료 전에 삭제했다.

