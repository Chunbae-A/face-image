# 딥소각 얼굴가드 문서 안내

처음 방문했다면 저장소 루트의 [프로젝트 소개](../README.md)를 먼저 읽고, 목적에 맞는 문서를 아래에서 선택한다. 문서 속 명령은 별도 안내가 없으면 저장소 루트에서 실행한다.

## 가장 빠른 시작

1. API를 직접 실행하려면 [API 빠른 실행](api/quickstart.md)
2. 발표 흐름을 준비하려면 [데모 파이프라인](demo/pipeline.md)
3. 모델 결과를 확인하려면 [딥페이크 기준선 결과](../reports/celebdf_deepfake_baseline/2026-08-07/README.md)
4. 실험을 다시 실행하려면 [딥페이크 Kaggle 실행 가이드](experiments/deepfake-kaggle.md)

## API와 서비스 연결

| 문서 | 언제 읽나요? |
|---|---|
| [API 빠른 실행](api/quickstart.md) | 저장소만 받은 사람이 API를 처음 실행할 때 |
| [API 운영 가이드](api/operations.md) | 모든 엔드포인트·환경변수·오류 코드를 확인할 때 |
| [비동기 노출 스캔](api/async-exposure-scan.md) | `scan_id` 기반 검색·얼굴 선별·딥페이크 분석을 시연할 때 |
| [SearXNG 검색](api/searxng.md) | 무료 키워드 검색 어댑터를 함께 실행할 때 |

## 데모

| 문서 | 내용 |
|---|---|
| [얼굴가드 데모 파이프라인](demo/pipeline.md) | 화면 순서, API 연결, 안전 문구와 데모 통과 기준 |

## 모델링과 실험 재현

아래 순서는 계획에서 실제 모델 비교와 점수 보정으로 이어지는 흐름이다.

| 순서 | 문서 | 내용 |
|---:|---|---|
| 1 | [얼굴가드 실험 계획](experiments/faceguard-plan.md) | 얼굴 동일인 비교 가설·데이터 분할·평가 기준 |
| 2 | [Colab 실행 가이드](experiments/colab.md) | 얼굴인식 기준선과 강건성 실험 실행 |
| 3 | [Kaggle 얼굴인식 가이드](experiments/kaggle-face-verification.md) | 얼굴인식 열화 실험용 무료 GPU 실행 |
| 4 | [딥페이크 기준선](experiments/deepfake-baseline.md) | Celeb-DF-v2 전처리·학습·평가·ONNX 구조 |
| 5 | [딥페이크 Kaggle 실행](experiments/deepfake-kaggle.md) | 전체 데이터 전처리와 학습 실행 순서 |
| 6 | [모델 고도화 계획](experiments/model-improvement.md) | Xception·SBI·Hard Negative·FTCN 비교 순서 |
| 7 | [EfficientNet-B4·Xception 비교](experiments/xception-comparison.md) | 동일 조건 비교 방법과 실제 결과 |
| 8 | [점수 보정](experiments/score-calibration.md) | 원점수를 화면용 확률로 바꾸기 위한 검증 |

JPEG 조건부 두 모델 결합은 현재 [Issue #35](https://github.com/Chunbae-A/face-image/issues/35)에서 검증 중이며, 결과가 확정되기 전에는 운영 모델로 취급하지 않는다.

## 실제 결과

실행 방법과 실제 결과를 섞지 않기 위해 날짜가 있는 결과물은 [`reports/`](../reports)에만 둔다.

- [ArcFace 기준선 재현성·누수 감사](../reports/celebdf_baseline_audit/2026-08-06/README.md)
- [촬영 열화 6조건 평가](../reports/celebdf_robustness/2026-08-06/README.md)
- [딥페이크 기준선 최종 결과](../reports/celebdf_deepfake_baseline/2026-08-07/README.md)
- [딥페이크 점수 보정 결과](../reports/deepfake_score_calibration/2026-08-08/README.md)
- [얼굴가드 데모 연결 시험](../reports/faceguard_demo_smoke/2026-08-06/README.md)
- [영상 딥페이크 API 연결 시험](../reports/video_deepfake_api_smoke/2026-08-08/README.md)

## 개발자용 위치

- [기여 규칙](../CONTRIBUTING.md): 이슈·브랜치·커밋·PR 규칙
- [`scripts/`](../scripts): 데이터 점검·학습·평가 명령
- [`notebooks/`](../notebooks): Colab·Kaggle 재현 노트북
- [`configs/`](../configs): 고정된 실험 설정

문서를 추가할 때는 목적에 맞는 하위 디렉터리에 넣고 이 안내 문서에서 연결한다. 실제 실행 결과는 `reports/<실험>/<날짜>/`에 둔다.
