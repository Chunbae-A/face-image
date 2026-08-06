# Celeb-DF ArcFace baseline 재현성·누수 감사

## 실행 범위

- 실행일: 2026-08-06
- 환경: Google Colab, Tesla T4 15GB
- 데이터: 정식 승인받은 Celeb-DF-v2의 `Celeb-real` 590개 영상, 59명
- 평가 가능 인물: 영상 8개 이상인 56명
- 모델: InsightFace `buffalo_l`, 비상업 연구용 가중치
- 런타임: `onnxruntime-gpu==1.23.2`, CUDAExecutionProvider
- 프로토콜: 영상당 1/5/10프레임, 등록 영상 1/3/5개, seed 5개
- threshold: validation identity에서만 선택한 뒤 test에 고정 적용
- bootstrap: subject bootstrap 500회

이 실험은 일반 얼굴 동일인 검증 baseline이며 딥페이크 탐지 실험이 아니다.

## 결정

고정 decision gate는 영상당 **5프레임**, 등록 영상 **3개**를 권장했다.

- 5프레임의 10프레임 대비 TAR 손실: `0.000000`
- 등록 5개의 등록 3개 대비 TAR 이득: `0.000000`
- 선택 조건 Test ROC-AUC 평균: `1.000000`
- 선택 조건 Test EER 평균: `0.000000`
- 선택 조건 Test TAR 평균 @ validation FAR 0.001: `1.000000`
- 선택 조건 관측 Test FAR 평균: `0.001380`
- 선택 조건 threshold seed 표준편차: `0.029050`

관측 Test FAR가 목표 `0.001`보다 높으므로 이 threshold를 운영값으로 승인하지 않는다. 더 어려운 외부 데이터와 실제 셀카 데이터에서 calibration을 다시 수행해야 한다.

## 입력 품질과 처리 시간

| 프레임/영상 | 성공 영상 | 성공률 | 유효 프레임 평균 | 영상당 decode 평균 | 영상당 ArcFace 평균 |
|---:|---:|---:|---:|---:|---:|
| 1 | 590/590 | 1.000000 | 1.000000 | 0.016001s | 0.024153s |
| 5 | 589/590 | 0.998305 | 5.000000 | 0.065200s | 0.099976s |
| 10 | 589/590 | 0.998305 | 9.998302 | 0.126798s | 0.200428s |

5프레임과 10프레임에서 같은 영상 1개가 `insufficient_valid_faces`로 제외됐다. 공유 결과에는 해당 video ID가 없다.

## 누수 검사

3개 frame run × 5개 seed의 모든 검사에서 다음 값이 0이었다.

- 전역 중복 video ID
- validation/test identity 교집합
- registration/query video 교집합

등록 수와 관계없이 query는 각 인물의 6번째 영상부터 사용했다. 결과 파일에는 identity와 video 목록 대신 fingerprint만 저장했다.

## 산출물

- [`celebdf_baseline_audit.json`](celebdf_baseline_audit.json): 결정, 입력 품질, 모델·manifest hash, 누수 검사
- [`celebdf_baseline_audit_metrics.csv`](celebdf_baseline_audit_metrics.csv): 45개 조건의 seed별 metric과 CI
- [`celebdf_baseline_audit_summary.csv`](celebdf_baseline_audit_summary.csv): frame/reference별 집계
- [`celebdf_baseline_audit.png`](celebdf_baseline_audit.png): TAR 및 관측 FAR 비교
- [`audit_runtime_config.json`](audit_runtime_config.json): 공개 가능한 실행 설정

원본 결과 ZIP SHA-256은 `88f1a8d128dd80bc8b1482f30aa6f44626b74ae516d24038899b7e9bf0602ba7`이다. 이 디렉터리에는 얼굴 이미지, 영상, 프레임, 개별 score, 임베딩, Drive 링크가 없다.

## 해석 제한

- Celeb-real 내부 데이터는 실제 셀카보다 쉬울 수 있다.
- 한국인 얼굴, 모바일 카메라, 조명·자세·압축 변화에 대한 일반화는 아직 검증하지 않았다.
- 높은 ROC-AUC를 딥페이크 탐지 정확도로 해석하지 않는다.
- AI-Hub 승인 후 별도 Issue에서 동일 프로토콜을 한국인 안면 데이터에 적용한다.
