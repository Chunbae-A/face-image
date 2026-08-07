# 딥페이크 점수 보정 실행 안내

## 왜 하나요?

모델이 낸 `0.84`는 **84% 확률이 아니다.** 보정은 비슷한 점수를 받은 영상 중 실제로 가짜였던 비율을 검증해 화면 숫자의 뜻을 맞추는 과정이다.

## 무엇을 비교하나요?

- Temperature Scaling
- Platt Scaling
- Isotonic Calibration
- 보정 전·후 ECE: 예측 숫자와 실제 비율이 얼마나 다른지
- 보정 전·후 Brier Score: 확률 오차의 평균
- 실제 영상을 가짜로 경고한 FPR

방법 선택과 위험 구간 선택에는 Validation만 사용한다. 공식 Test는 선택이 끝난 뒤 마지막 확인에만 사용한다.

## Kaggle에서 실행

1. 새 **Private** Kaggle Notebook을 만든다.
2. [`notebooks/celebdf_score_calibration_kaggle.ipynb`](notebooks/celebdf_score_calibration_kaggle.ipynb)를 Import한다.
3. Input에 기존 `deepsogak-celebdf-preprocess` Output을 연결한다.
4. Input에 기존 `deepsogak-celebdf-train` Output을 연결한다.
5. GPU를 선택하고 `Run All`을 누른다.

모델을 재학습하지 않으며, Validation 836개와 공식 Test 518개의 clean 16프레임만 다시 추론한다. 비공개 영상별·프레임별 점수는 `/kaggle/temp`에만 존재하고 결과 저장 전에 삭제된다.

## 결과 판단

2026-08-08 실제 실행 결과는 [`reports/deepfake_score_calibration/2026-08-08`](reports/deepfake_score_calibration/2026-08-08)에 있다. Isotonic이 선택됐고 ECE Gate는 통과했지만 공식 Test 실제영상 FPR이 `0.016854`로 목표 `0.01`을 넘어 `display_approved=false`가 됐다.

`deepfake_video_calibration.json`의 다음 값을 확인한다.

```json
{
  "calibration_status": "research_only_unapproved",
  "display_approved": false,
  "selected_method": "...",
  "gate": {
    "overall_pass": false
  }
}
```

- `display_approved=true`: Celeb-DF 내부 연구 Gate를 통과해 API가 보정 확률을 반환할 수 있다.
- `display_approved=false`: 실험은 완료했지만 화면 퍼센트 표시는 금지한다.
- Gate는 ECE 0.05 이하와 공식 Test 실제영상 FPR 1% 이하를 모두 요구한다.

## API에 설치

Kaggle에서 받은 JSON만 모델 디렉터리에 둔다. 원본 점수 CSV는 내려받거나 GitHub에 올리지 않는다.

```bash
cp /다운로드/경로/deepfake_video_calibration.json \
  .models/deepfake/deepfake_video_calibration.json
docker compose up --build --detach
curl http://127.0.0.1:8000/health
```

`/health`의 `deepfake_video_calibration_status`와 `deepfake_video_calibration_version`을 확인한다.

## 화면 규칙

| API 값 | 화면 표시 |
|---|---|
| `calibrated_probability`가 숫자 | 검증 범위를 함께 적고 퍼센트 표시 가능 |
| `calibrated_probability=null` | 퍼센트 숨김, `원점수`와 `검토 필요` 표시 |
| `risk_level=low` | 낮은 위험 후보, 자동 삭제 금지 |
| `risk_level=review` | 사람이 원본·출처 확인 |
| `risk_level=high` | 우선 검토 후보, 피해 확정 표현 금지 |

얼굴 유사도는 별도 문제다. ArcFace의 개별 동일인·타인 검증 점수가 준비되기 전까지 얼굴 유사도 역시 확률로 표시하지 않는다.
