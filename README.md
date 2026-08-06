# face-image

딥소각 **얼굴가드**의 일반 얼굴 동일인 검증 baseline 저장소다. Celeb-DF-v2의 `Celeb-real` 590개 영상에서 얼굴을 탐지·정렬하고 ArcFace 임베딩을 만든 뒤, 등록 영상 3개/5개 프로토콜을 비교한다.

> 이 실험은 **얼굴 동일인 검증**이며 딥페이크 탐지 정확도가 아니다.

## 초기 Celeb-real baseline 결과

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

## 재현성·누수 감사 결과

2026-08-06 Tesla T4에서 `frames={1,5,10} × reference={1,3,5} × seed=5개`의 45개 조건을 감사했다. 모든 seed에서 validation/test identity 교집합, registration/query video 교집합, 전역 중복 video ID가 0건이었다.

| 프레임/영상 | 등록 영상 | Test ROC-AUC 평균 | Test EER 평균 | Test TAR 평균 @ validation FAR 0.001 | 관측 Test FAR 평균 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.999998 | 0.000194 | 0.999005 | 0.002197 |
| 1 | 3 | 1.000000 | 0.000013 | 1.000000 | 0.000895 |
| 1 | 5 | 1.000000 | 0.000000 | 1.000000 | 0.000868 |
| 5 | 1 | 1.000000 | 0.000000 | 1.000000 | 0.001459 |
| **5** | **3** | **1.000000** | **0.000000** | **1.000000** | **0.001380** |
| 5 | 5 | 1.000000 | 0.000000 | 1.000000 | 0.002142 |
| 10 | 1 | 1.000000 | 0.000000 | 1.000000 | 0.001459 |
| 10 | 3 | 1.000000 | 0.000000 | 1.000000 | 0.001252 |
| 10 | 5 | 1.000000 | 0.000000 | 1.000000 | 0.001555 |

고정 decision gate에 따라 **영상당 5프레임·등록 영상 3개**를 연구 baseline으로 채택한다. 10프레임 대비 TAR 손실과 등록 5개 대비 TAR 이득이 모두 0이었다. 다만 선택 조건의 관측 Test FAR 평균 `0.001380`은 목표 `0.001`보다 높으므로, 운영 threshold로 승인된 결과가 아니며 외부 데이터에서 재보정해야 한다.

집계 결과와 그래프는 [`reports/celebdf_baseline_audit/2026-08-06`](reports/celebdf_baseline_audit/2026-08-06)에 있다. 영상, 얼굴 crop, 개별 score와 임베딩은 포함하지 않는다.

## Colab에서 재실행

1. [`notebooks/celebdf_arcface_full_colab.ipynb`](notebooks/celebdf_arcface_full_colab.ipynb)를 Google Colab에 업로드한다.
2. GPU runtime을 선택한다. Tesla T4에서 검증했다.
3. 정식 승인받은 `Celeb-DF-v2.zip`을 약관이 허용하는 경로에 둔다. 기본 경로는 `/content/drive/MyDrive/Celeb-DF-v2.zip`이다.
4. 클라우드 처리 확인과 InsightFace 비상업 연구용 가중치 확인값을 `True`로 변경한다.
5. 설치 후 runtime을 재시작했다면 2번 설치 셀은 다시 실행하지 않고 1, 3~13번 셀을 실행한다.

노트북은 Colab의 CUDA 사용자 라이브러리와 호환되도록 `onnxruntime-gpu==1.23.2`를 사용한다. 기본 `CODE_SOURCE="embedded"`는 GitHub clone 권한 없이도 필요한 스크립트를 노트북 내부에서 복원한다.

### Baseline 감사

[`notebooks/celebdf_arcface_audit_colab.ipynb`](notebooks/celebdf_arcface_audit_colab.ipynb)는 Issue [#4](https://github.com/Chunbae-A/face-image/issues/4)의 `frames={1,5,10} × reference={1,3,5} × seed=5개` 감사를 실행한다. 원본과 임베딩은 runtime에만 두고, 누수 검사·집계 metric·hash·그래프만 결과 ZIP에 포함한다.

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
