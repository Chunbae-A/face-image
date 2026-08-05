# face-image

딥소각 **얼굴가드**의 일반 얼굴 동일인 검증 baseline 저장소다. Celeb-DF-v2의 `Celeb-real` 590개 영상에서 얼굴을 탐지·정렬하고 ArcFace 임베딩을 만든 뒤, 등록 영상 3개/5개 프로토콜을 비교한다.

> 이 실험은 **얼굴 동일인 검증**이며 딥페이크 탐지 정확도가 아니다.

## Celeb-real baseline 결과

2026-08-06 Google Colab Tesla T4에서 영상당 10프레임을 사용해 실행한 결과다.

- 전체 590개 영상 중 589개 성공: **99.83%**
- 평가 가능 인물: 56명
- subject-disjoint validation/test: 17명/39명
- 동일 영상의 프레임이 등록과 query에 동시에 들어가지 않음
- threshold는 validation에서 선택하고 test에 고정 적용

| 지표 | 등록 3개 | 등록 5개 |
|---|---:|---:|
| Test ROC-AUC | 1.0000 | 1.0000 |
| Test EER | 0.0000 | 0.0000 |
| FAR 0.001 기준 threshold | 0.277838 | 0.285558 |
| FAR 0.001 기준 Test TAR | 1.0000 | 1.0000 |
| FAR 0.001 기준 Test FAR | 0.000387 | 0.000258 |

이 결과는 Celeb-real 내부 baseline이다. 한국인 얼굴, 실제 셀카, 모바일 카메라 및 운영 환경의 일반화 성능은 별도의 데이터로 다시 평가해야 한다.

## Colab에서 재실행

1. [`notebooks/celebdf_arcface_full_colab.ipynb`](notebooks/celebdf_arcface_full_colab.ipynb)를 Google Colab에 업로드한다.
2. GPU runtime을 선택한다. Tesla T4에서 검증했다.
3. 정식 승인받은 `Celeb-DF-v2.zip`을 약관이 허용하는 경로에 둔다. 기본 경로는 `/content/drive/MyDrive/Celeb-DF-v2.zip`이다.
4. 클라우드 처리 확인과 InsightFace 비상업 연구용 가중치 확인값을 `True`로 변경한다.
5. 설치 후 runtime을 재시작했다면 2번 설치 셀은 다시 실행하지 않고 1, 3~13번 셀을 실행한다.

노트북은 Colab의 CUDA 사용자 라이브러리와 호환되도록 `onnxruntime-gpu==1.23.2`를 사용한다. 기본 `CODE_SOURCE="embedded"`는 GitHub clone 권한 없이도 필요한 스크립트를 노트북 내부에서 복원한다.

## 구조

- [`FACEGUARD_EXPERIMENT_PLAN.md`](FACEGUARD_EXPERIMENT_PLAN.md): 한국인 안면 데이터 승인 후 Debug/Pilot/Full 실험 계획
- [`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md): Colab, 보안 게이트, checkpoint 운영 가이드
- [`configs/faceguard`](configs/faceguard): 일반 얼굴가드와 Celeb-DF 고정 프로토콜
- [`scripts`](scripts): ZIP inventory/안전 추출, ArcFace 추론, 평가 도구
- [`tests`](tests): 데이터 누수·압축 경로·프로토콜·지표 테스트

## 로컬 검증

```bash
python3 -m unittest discover -s tests
```

## 데이터와 라이선스

실제 얼굴 영상·이미지, 임베딩, 제공기관 압축파일은 Git에 커밋하지 않는다. InsightFace 코드와 제공 사전학습 가중치의 라이선스는 다르며, `buffalo_l` 가중치는 비상업 연구 범위에서만 사용한다.
