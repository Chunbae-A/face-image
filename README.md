# 딥소각 얼굴가드

사용자가 등록한 얼굴을 기준으로 **공개 웹 후보에서 같은 사람을 골라내고, 후보 영상의 딥페이크 위험을 수치와 근거로 보여주기 위한 모델링·API 프로젝트**다.

> 현재 바로 실행할 수 있는 범위는 ArcFace 얼굴 동일인 비교, SearXNG 공개 이미지 후보 수집, 후보 이미지 ArcFace 선별, EfficientNet-B4 ONNX 이미지 분석, **짧은 영상 16프레임 분석 API**, **`scan_id`로 진행률을 조회하는 비동기 이미지 후보 API**다. 얼굴 사진 자체로 웹을 찾는 역이미지 검색과 검색 후보 영상의 자동 다운로드는 개발 중이며, 현재 모델은 운영·본인인증·자동 차단 용도로 승인되지 않았다.

## 30초 요약

딥소각이 최종적으로 만들려는 흐름은 다음과 같다.

```text
본인 얼굴 3장 등록
        ↓
공개 웹에서 이미지·영상 후보 수집
        ↓
ArcFace: 정말 본인 얼굴과 비슷한 후보인지 선별
        ↓
EfficientNet-B4: 후보 영상이 딥페이크인지 분석
        ↓
얼굴 유사도 + 딥페이크 위험도 + 발견 위치를 화면에 표시
        ↓
사용자가 증거를 확인하고 신고·제외 여부 결정
```

모델의 수치는 판단을 돕는 근거다. 얼굴 유사도만으로 딥페이크라고 결론 내리거나, 모델 결과만으로 게시물을 자동 삭제하지 않는다.

## 왜 만들었고, 무엇을 해결했나요?

| 질문 | 답 |
|---|---|
| 왜 했나? | 웹 검색 결과에는 본인이 아닌 얼굴과 실제·조작 여부를 모르는 이미지가 함께 섞이므로, 검색 결과를 그대로 피해 후보로 보여줄 수 없기 때문이다. |
| 어떻게 했나? | 검색 도구는 공개 후보 URL만 모으고, ArcFace가 등록 얼굴과 같은 사람 후보를 선별한 뒤, EfficientNet-B4 ONNX가 이미지·영상의 딥페이크 의심 원점수를 계산한다. |
| 무엇을 해결했나? | 검색·동일인 선별·딥페이크 분석을 `scan_id` 작업 하나로 연결하고, 클라이언트가 확률로 오해하지 않는 화면용 API 응답을 만들었다. |
| 어떻게 활용하나? | 딥소각 서버가 모델 API를 내부 호출하고, 앱에는 출처·얼굴 일치 단계·딥페이크 의심 신호·권장 행동만 전달한다. 사용자가 원문을 검토하고 신고자료를 만드는 근거로 쓴다. |

아직 해결하지 않은 범위도 분명히 구분한다. 얼굴 사진 자체를 이용한 웹
역검색, 검색된 영상 URL 자동 다운로드, 실제 웹·모바일 외부 데이터의 운영
Gate는 남아 있다. 따라서 현재 응답은 자동 차단·자동 신고 명령이 아니다.

## 지금 어디까지 됐나요?

| 기능 | 하는 일 | 현재 상태 |
|---|---|---|
| 공개 후보 검색 | 공개 웹에서 이미지·영상 URL과 출처 수집 | **무료 URL 제보 + SearXNG 키워드 검색 구현, 얼굴 역검색 미연결** ([#13](https://github.com/Chunbae-A/face-image/issues/13)) |
| 얼굴 후보 선별 | 등록 얼굴과 후보 얼굴의 동일인 가능성 비교 | **검색 이미지 다운로드·ArcFace 배치 연결 완료, 다중 얼굴·영상 트랙 미연결** ([#14](https://github.com/Chunbae-A/face-image/issues/14)) |
| 딥페이크 판별 | 후보 얼굴 프레임이 실제인지 조작인지 분석 | **ONNX 이미지·영상 16프레임 평균 API 구현, 운영 Gate 미통과** ([#25](https://github.com/Chunbae-A/face-image/issues/25)) |
| 딥페이크 모델 고도화 | 실제 영상 오경고와 웹 촬영 열화 약점 개선 | **EfficientNet-B4/Xception 공정 비교 코드·Kaggle 노트북 준비, GPU 결과 대기; SBI·Hard Negative·FTCN 실행 전** ([#29](https://github.com/Chunbae-A/face-image/issues/29)) |
| 화면용 신뢰도 | 얼굴 유사도와 딥페이크 점수를 사용자용 수치로 보정 | **K-FACE 400명 얼굴 검증·딥페이크 보정 실험 완료, 두 Gate 모두 미통과로 확률 표시 보류** ([#16](https://github.com/Chunbae-A/face-image/issues/16)) |
| 통합 비동기 API | 검색 → 얼굴 선별 → 딥페이크 판별을 하나의 작업으로 연결 | **임시 등록·`scan_id`·진행 조회·이미지 후보 결과 완료, 영상 URL·영구 큐는 미연결** ([#17](https://github.com/Chunbae-A/face-image/issues/17)) |

외부 서버 배포는 현재 필수 범위가 아니다. 데모는 로컬 Docker API와 Swagger 화면으로 실행한다.

## 사용한 AI 모델

이 프로젝트에는 목적이 다른 두 모델이 있다.

| 모델 | 질문 | 사용 방법 |
|---|---|---|
| ArcFace `buffalo_l` | “두 얼굴이 같은 사람인가?” | 얼굴을 512개 숫자 특징으로 바꾸고 코사인 유사도를 계산 |
| EfficientNet-B4 | “이 얼굴 프레임이 실제인가, 딥페이크인가?” | Celeb-DF-v2 얼굴 프레임으로 실제·가짜 이진 분류 학습 |

얼굴을 찾고 정렬하는 전처리에는 InsightFace의 SCRFD 계열 검출기를 사용한다. ArcFace는 처음부터 다시 학습하지 않았고, 사전학습 모델에 딥소각의 등록 방식·데이터 분리·판정 기준·API를 설계해 검증했다. EfficientNet-B4는 ImageNet 사전학습 가중치에서 시작해 Celeb-DF-v2로 학습했다.

## 1. 얼굴 동일인 비교 결과

승인받은 Celeb-DF-v2의 실제 얼굴 영상 `Celeb-real`로 등록 사진 수와 판정 기준을 검증했다.

- 전체 영상: 590개, 59명
- 정상 처리: 589개, **99.83%**
- 최종 평가 가능 인물: 56명
- Validation/Test 인물: 17명/39명
- 채택 설정: 영상당 5프레임, 등록 영상 3개 평균
- 반복 검증: 프레임 수 3종 × 등록 수 3종 × seed 5개 = 45조건
- 누수 검사: Validation/Test 인물 교집합 0건, 등록/확인 영상 교집합 0건

| Test 지표 | 결과 | 쉬운 뜻 |
|---|---:|---|
| ROC-AUC | 1.000000 | 같은 사람과 다른 사람의 점수 순서를 잘 구분 |
| EER | 0.000000 | 이 데이터 내부에서 두 오류율이 만나는 지점 |
| TAR | 1.000000 | 같은 사람을 통과시킨 비율 |
| 관측 FAR | 0.001380 | 다른 사람을 같은 사람으로 잘못 통과시킨 비율 |

연구 목표 FAR은 `0.001`, 즉 다른 사람 10,000번 비교 중 최대 10번의 오통과였다. 관측값은 약 13.8번 수준이므로 **Celeb-real 내부 결과가 좋아도 운영 본인인증 기준은 통과하지 못했다.** 연구 판정 기준값은 `0.2823836207389832`이며 API 응답에도 `research_only_unapproved`로 표시한다.

전체 45조건과 신뢰구간은 [`reports/celebdf_baseline_audit/2026-08-06`](reports/celebdf_baseline_audit/2026-08-06), 촬영 열화 6조건 결과는 [`reports/celebdf_robustness/2026-08-06`](reports/celebdf_robustness/2026-08-06)에 있다.

### K-FACE 400명 한국인 저·중화질 검증

AI-Hub 승인 K-FACE 400명에서 인물당 저화질 15장·중화질
15장, 총 12,000장을 처리했다. 중화질 등록 3장·5장을 비교하고,
400명을 validation 200명과 test 200명으로 완전히 나눠 기준값 누수를
막았다.

| Test 조건 | TAR | FAR | 판정 |
|---|---:|---:|---|
| 3장 등록·저화질 질의 | 84.44% | 0.1109% | Gate 미통과 |
| 3장 등록·중화질 질의 | 92.15% | 0.1090% | Gate 미통과 |
| 5장 등록·저화질 질의 | **87.74%** | 0.1201% | Gate 미통과 |
| 5장 등록·중화질 질의 | **95.58%** | 0.1132% | Gate 미통과 |

5장 등록은 3장보다 본인 통과율을 높였으므로 시연에서 권장한다.
그러나 연구 Gate인 `TAR ≥ 90%`, `FAR ≤ 0.1%`를 모든 화질에서 동시에
만족하지 못했다. K-FACE 기준값은 API 운영값으로 교체하지 않고
`research_only_unapproved`를 유지한다. 전체 프로토콜·점수 분포·한계는
[`reports/kface_v3_400_evaluation/2026-08-15`](reports/kface_v3_400_evaluation/2026-08-15)에 있다.

#### K-FACE v3.1 적응형 검출·품질 Gate

Mac에서 기존 실패 이미지만 SCRFD 입력 크기 `960 → 1280`으로 다시
처리해 검출 성공률을 저화질 `78.87% → 86.65%`, 중화질
`88.45% → 92.08%`로 높였다. 복구 입력을 모두 자동 판정에 섞지 않고,
검출점수 `0.60` 이상과 중화질 등록 프레임 9장, validation FAR
`0.09%` 안전 여유를 적용했다.

| v3.1 Test 조건 | TAR | FAR | 연구 Gate |
|---|---:|---:|---|
| 9장 등록·저화질 질의 | **91.95%** | **0.0828%** | 통과 |
| 9장 등록·중화질 질의 | **92.91%** | **0.0975%** | 통과 |

이는 subject-disjoint test에서 `TAR ≥ 90%`, `FAR ≤ 0.1%`를 처음
동시에 만족한 **연구 후보**다. 사용자에게 파일 9장을 요구하기보다 짧은
촬영에서 좋은 프레임 9장을 자동 선택하는 후속 UX가 필요하다. 실제 웹·SNS
외부 검증 전에는 현재 API의 최대 등록 수와 기준값을 바꾸지 않는다. 전체
재현 방법·비교·한계는
[`reports/kface_adaptive_quality_gate/2026-08-16`](reports/kface_adaptive_quality_gate/2026-08-16)에 있다.

#### K-FACE v3.2 전체 864만 장 반복 검증

12,000장 표본에서 찾은 v3.1 설정을 K-FACE 저·중화질 864만 장 전체와
5개의 인물 분할에 다시 적용했다. 등록 수를 3장·5장·9장으로 늘릴수록
본인 통과율은 좋아졌지만 전체 저화질 조건에서는 연구 Gate를 통과하지
못했다.

| 전체 반복 Test | 저화질 최악 TAR | 중화질 최악 TAR | 최악 FAR | 판정 |
|---|---:|---:|---:|---|
| 3장 등록 | 69.65% | 88.88% | 0.1021% | 미통과 |
| 5장 등록 | 76.35% | 93.35% | 0.1073% | 미통과 |
| 9장 등록 | **80.36%** | **95.22%** | 0.1207% | 미통과 |

표본 Gate 통과는 방향 탐색 결과이고, 전체·반복 검증은 더 보수적인 운영
판단 근거다. 9장 연구 후보 기준값 `0.3862`는 현재 API에 적용하지 않으며
`threshold_status=research_only_unapproved`를 유지한다. 전체 방법·결과·API
판단은
[`reports/kface_full_verification/2026-08-17`](reports/kface_full_verification/2026-08-17)에
있다.

#### K-FACE v3.3 실제 얼굴 픽셀 크기·밝기 Gate 탐색

전체 864만 장 특징값에서 검출점수, 실제 얼굴 픽셀 크기와
밝기를 조합한 11개 품질 Gate를 5개 인물 분할로 비교했다. 가장
균형이 좋은 `검출 0.70 + 얼굴 42px + 밝기 35` 조건은 최악
TAR `94.17%`를 달성했지만 FAR이 `0.1074%`로 목표를 넘었고,
자동 처리 coverage도 최소 `30.62%`에 그쳤다.

따라서 품질 거절만 강제하는 API 변경은 하지 않는다. 다음에는
품질 가중 등록·다중 등록 중심·더 엄격한 validation 안전 여유를
비교한다. 전체 결과는
[`reports/kface_quality_gate_analysis/2026-08-17`](reports/kface_quality_gate_analysis/2026-08-17)에
있다.

#### K-FACE v3.4 등록 5장 품질 가중·다중 중심 비교

확인 사진을 추가로 거절하지 않고 coverage 100% 조건에서 단순
평균, 품질 가중 평균, 등록 중심 2개를 비교했다. 품질 가중
평균은 validation FAR 0.09% 기준 저화질 TAR을 `76.35% → 76.62%`로
약 `+0.28%p` 높였지만 목표 개선폭에 부족했다.

validation FAR 안전 여유를 0.08%로 높이면 test FAR은 `0.0963%`로
목표를 통과했지만 저화질 TAR은 `75.80%`였다. 등록 중심 2개는
저화질 TAR을 약 72%로 낮춰 후보에서 제외했다. 등록 전략만으로는
저해상도 정보 손실을 해결하지 못했으므로 API는 변경하지 않는다.
전체 결과는
[`reports/kface_enrollment_strategy_benchmark/2026-08-17`](reports/kface_enrollment_strategy_benchmark/2026-08-17)에
있다.

#### K-FACE v3.5 저화질 임베딩 보정 모델

400명을 Train 240명·Validation 80명·잠긴 Test 80명으로 인물 단위
분리하고, 저화질 ArcFace 특징을 중화질 특징에 가깝게 바꾸는
`512→128→512` residual MLP를 학습했다. Validation에서 선택된
대조 학습 후보를 잠긴 Test에서 한 번 평가했지만 저화질 TAR은
`79.01% → 78.05%`, 중화질 TAR은 `94.61% → 92.34%`로 낮아졌고
최악 FAR은 `0.1065% → 0.1250%`로 높아졌다.

다른 사람 평균 유사도가 같은 사람보다 더 크게 올라 인물 구분력이
나빠진 것이 원인이다. 사전 개선 Gate를 통과하지 못해 ONNX를 만들지
않았고 API도 변경하지 않았다. 전체 결과는
[`reports/kface_lowres_embedding_adapter/2026-08-17`](reports/kface_lowres_embedding_adapter/2026-08-17)에
있다.

## 2. 딥페이크 영상 판별 결과

전체 Celeb-DF-v2를 사용했다.

| 데이터 | 영상 수 |
|---|---:|
| Celeb-real | 590 |
| YouTube-real | 300 |
| Celeb-synthesis | 5,639 |
| 전체 | **6,529** |
| 제작자가 고정한 공식 Test | 518 |

프레임을 뽑기 전에 영상을 Train/Validation/Test로 나눴다. 같은 영상 장면이 학습과 평가 양쪽에 섞이는 것을 막기 위해서다. 전체 전처리에서는 6,529개 영상 중 6,528개를 처리해 얼굴 crop 208,893개를 만들었고, 1개 영상은 유효 얼굴 프레임 부족으로 제외했다.

Kaggle 4차 실행에서 학습·공식 Test·ONNX 변환·CPU 추론 시험을 모두 완료했다. 아래 값은 공식 비식별 결과 파일에서 확인한 **최종 연구 기준선**이다.

| 공식 Test 지표 | 결과 | 판단 |
|---|---:|---|
| 선택 프레임 수 | 16 | 영상마다 최대 16개 얼굴 점수 사용 |
| 점수 통합 | 평균 | 프레임별 점수를 평균해 영상 점수 생성 |
| 판정 기준값 | 0.751988 | 이 값 이상을 딥페이크 후보로 판단 |
| Video ROC-AUC | 0.999802 | 실제와 가짜의 전반적인 순위 구분은 높음 |
| Recall | 1.000000 | 공식 Test 딥페이크를 모두 탐지 |
| FPR | **0.016854** | 실제 영상을 가짜로 잘못 경고한 비율 약 1.69% |
| 얼굴 처리 coverage | 1.000000 | 공식 Test 평가 대상 처리 완료 |

초기 연구 Gate는 `ROC-AUC ≥ 0.90`, `FPR ≤ 0.01`이다. AUC는 통과했지만 FPR이 목표 1%보다 높아 **즉시 경보·자동 차단용 모델로 승인하지 않는다.** ONNX는 정상 생성됐고 CPU 추론 smoke도 약 `171ms`로 통과했지만, 이는 연결 시험일 뿐 API 응답시간 SLA가 아니다.

열화 평가에서는 해상도 25% 축소와 흐림에서 FPR이 각각 약 `12.92%`, `11.80%`로 증가했고 저조도에서는 Recall이 약 `78.24%`로 감소했다. 따라서 실제 웹에서 재압축·축소된 영상에 바로 적용하면 안 된다. 상세 수치·무결성 hash·그래프는 [최종 결과 보고서](reports/celebdf_deepfake_baseline/2026-08-07)에 있다.

재현 방법은 [`docs/experiments/deepfake-kaggle.md`](docs/experiments/deepfake-kaggle.md), 데이터 분할과 사전검사·현재 결과는 [`reports/celebdf_deepfake_baseline/2026-08-07`](reports/celebdf_deepfake_baseline/2026-08-07)에 있다.

## 3. 화면용 딥페이크 점수 보정 결과

Kaggle GPU에서 Validation 836개와 공식 Test 518개 영상의 clean 16프레임 평균 점수를 다시 계산하고 Temperature, Platt, Isotonic 보정을 비교했다. 방법 선택에는 Validation만 사용했으며 공식 Test로 보정값을 고른 경우는 0건이다.

| 공식 Test 지표 | 보정 전 | 보정 후 | 판단 |
|---|---:|---:|---|
| ECE | 0.029784 | 0.003861 | 0.05 이하 통과 |
| Brier Score | 0.013203 | 0.003861 | 개선 |
| 실제영상 FPR | 0.016854 | 판정 기준 동일 | 0.01 이하 Gate 실패 |

Isotonic 보정의 숫자 정합성은 좋아졌지만 실제 영상 178개 중 3개를 가짜로 잘못 경고했다. 따라서 API는 `calibration_status=research_only_unapproved`, `calibrated_probability=null`을 반환하며 화면에서 `84% 확률`처럼 표시하지 않는다. 상세 결과와 reliability diagram은 [`reports/deepfake_score_calibration/2026-08-08`](reports/deepfake_score_calibration/2026-08-08)에 있다.

## 얼굴가드 API 실행

현재 HTTP API는 **얼굴 동일인 후보 선별**, **공개 URL 정규화**, **SearXNG 후보 수집**, **단일 얼굴 딥페이크 ONNX 분석**, **짧은 영상 16프레임 평균 분석**, **검색 이미지 → ArcFace → ONNX 통합 경로**, **`scan_id` 기반 비동기 이미지 후보 처리**를 제공한다. 얼굴 역이미지 검색과 영상 URL을 직접 받는 비동기 경로는 아직 구현 전이다.

### Docker 권장 실행

```bash
git clone https://github.com/Chunbae-A/face-image.git
cd face-image
cp .env.example .env
```

InsightFace 제공 가중치의 비상업 연구 조건을 직접 확인한 경우에만 `.env`에서 다음 값을 변경한다.

```text
FACEGUARD_ACCEPT_NONCOMMERCIAL_MODEL_LICENSE=true
```

딥페이크 API까지 사용하려면 권한이 있는 팀원이 비공개 모델 ZIP의 `efficientnet_b4.onnx`를 `.models/deepfake/efficientnet_b4.onnx`에 둔다. GitHub에는 모델을 올리지 않으며, 실행 시 SHA-256 `c32a8532...a6f1` 전체 값이 자동 검증된다. 모델이 없더라도 얼굴 비교·검색 API는 실행되지만 딥페이크 분석은 `MODEL_UNAVAILABLE`을 반환한다.

점수 보정 실험을 마친 뒤에는 비식별 `deepfake_video_calibration.json`도 같은 디렉터리에 둔다. 파일이 없거나 연구 Gate를 통과하지 못하면 API의 `calibrated_probability`는 자동으로 `null`이 된다.

서버를 실행한다.

```bash
docker compose up --build --detach
curl http://127.0.0.1:8000/health
```

무료 키워드 검색까지 함께 켜려면 기본 명령 대신 다음 결합 구성을 실행한다.

```bash
docker compose -f docker-compose.yml -f docker-compose.searxng.yml up --build --detach
```

브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)를 열고 `POST /v1/faceguard/verify`, `POST /v1/deepfake/analyze` 또는 `POST /v1/deepfake/analyze-video`를 시험한다.

- `reference_images`: 본인 등록 얼굴 1~5장, **3장 권장**
- `query_image`: 확인할 후보 얼굴 1장
- 지원 형식: JPEG, PNG, WEBP

영상 분석은 MP4·MOV, 최대 50MB·120초를 지원한다. 영상 전체에서 최대 16개 대표 프레임을 뽑고 프레임별 얼굴 점수의 평균과 대략적인 의심 시간 구간을 반환한다. 등록 사진을 함께 보내면 여러 얼굴 중 등록 얼굴과 가장 비슷한 얼굴을 우선 분석한다. 원본 영상과 얼굴 crop은 영구 저장하지 않는다.

응답의 핵심값은 다음과 같다.

| 필드 | 의미 |
|---|---|
| `is_same_person` | 연구 기준값을 넘었는지 여부 |
| `similarity` | 두 얼굴의 코사인 유사도 |
| `raw_score` | 모델이 직접 낸 원점수. 확률이나 퍼센트가 아님 |
| `calibrated_probability` | 보정과 Gate를 모두 통과할 때만 제공하는 확률, 그 외 `null` |
| `calibration_status` | `not_available`, `research_only_unapproved`, `validated` 등 보정 상태 |
| `risk_level` | 영상 보정 파일이 있을 때 `low`, `review`, `high` 중 검토 구간 |
| `threshold` | 현재 연구 판정 기준값 |
| `reference_quality`, `query_quality` | 얼굴 크기·선명도·밝기 등 입력 품질 |
| `processing_ms` | 서버 내부 처리시간 |

`is_same_person=true`는 “같은 사람 후보”라는 뜻이지 딥페이크라는 뜻이 아니다. 자세한 실행과 오류 코드는 [API 빠른 실행](docs/api/quickstart.md)과 [API 운영 가이드](docs/api/operations.md)에 있다.

공개 후보 경로는 같은 Swagger 화면의 `POST /v1/search/candidates`에서 시험한다.

- `privacy_strict`: 사용자가 직접 넣은 공개 URL만 정리한다. 로컬·내부망 주소를 차단하고 추적 파라미터를 제거한 뒤 중복을 합친다.
- `web_monitoring`: 명시적 동의 후 검색어를 로컬 SearXNG에 보내 공개 이미지·영상 후보를 찾는다. 얼굴 사진은 보내지 않는다.

SearXNG은 **검색어 기반 메타검색**이다. 등록 얼굴 사진과 닮은 웹 사진을 자동으로 찾는 얼굴 역검색은 아니며, 찾은 후보가 본인인지와 딥페이크인지는 다음 ArcFace·ONNX 단계에서 별도로 검사해야 한다. 자세한 실행법은 [SearXNG 실행 가이드](docs/api/searxng.md)에 있다.

검색부터 딥페이크 이미지 분석까지 한 번에 시험할 때는 `POST /v1/pipeline/search-and-filter`를 사용한다. 등록 사진은 로컬 ArcFace에만 입력되고, SearXNG에는 검색어만 전달된다. 넓은 얼굴 후보 기준을 통과한 이미지만 ONNX로 분석하며 후보별 `similarity_raw`, `deepfake.deepfake_score`, 판정 여부, 품질과 실패 코드를 반환한다.

### 비동기 노출 스캔 데모

긴 검색 작업 동안 화면이 멈추지 않게 하려면 다음 API를 순서대로 사용한다.

1. `POST /v1/faceguard/enrollments`: 본인 사진 3장을 임시 등록하고 `enrollment_id`를 받는다.
2. `POST /v1/exposure-scans`: 공개 후보 URL 또는 동의한 검색어로 작업을 만들고 즉시 `scan_id`를 받는다.
3. `GET /v1/exposure-scans/{scan_id}`: `searching → identity_filtering → deepfake_analyzing → completed` 진행 단계와 개수를 확인한다.
4. `GET /v1/exposure-scans/{scan_id}/client-candidates`: 딥소각 화면용 후보와 검토 행동값을 확인한다.
5. `GET /v1/exposure-scans/{scan_id}/candidates`: 모델 개발자가 후보별 상세 판정 근거를 확인한다.

등록 요청에서 원본 사진을 처리한 뒤에는 풀링한 임베딩과 품질 정보만 기본 30분 동안 메모리에 남는다. 스캔 결과는 기본 60분 동안 남으며 서버를 재시작하면 모두 사라진다. 재시도 시 중복 작업을 막으려면 `Idempotency-Key` 헤더를 같게 보낸다. 초보자용 Swagger 실행 순서와 JSON 예시는 [비동기 노출 스캔 안내](docs/api/async-exposure-scan.md)에 있다. 프론트·백엔드 연결 계약은 [클라이언트용 공개 노출 모니터링 API](docs/api/client-monitoring.md)에 정리했다.

## 데모에서 보여줄 내용

현재 안전하게 시연할 수 있는 데모는 다음과 같다.

1. 동의받은 사람의 등록 사진 3장을 넣는다.
2. 같은 사람의 다른 사진을 넣어 유사도와 품질 수치를 확인한다.
3. 다른 사람의 사진을 넣어 불일치 결과를 확인한다.
4. Celeb-DF 전체 딥페이크 실험 결과에서 AUC와 FPR을 함께 보여준다.
5. “AUC는 높지만 FPR Gate를 통과하지 못해 운영 승인하지 않았다”고 설명한다.

검색 → 얼굴 선별 → 단일 이미지 딥페이크 판별은 `scan_id`를 사용해 비동기로 시연할 수 있고, 짧은 영상 16프레임 분석은 별도 API로 시연할 수 있다. 다만 `deepfake_score`, `video_score`, `raw_score`는 확률이나 확정 신뢰도가 아니다. 화면은 `calibrated_probability`가 `null`이면 퍼센트를 숨기고 `원점수·검토 필요`로 표시해야 한다. 검색 후보 영상을 자동으로 내려받는 부분과 프로세스 재시작에도 복구되는 영구 큐는 [Issue #17](https://github.com/Chunbae-A/face-image/issues/17)의 남은 범위다. 화면 흐름과 오류 처리는 [데모 파이프라인](docs/demo/pipeline.md)에 있다.

## 저장소 안내

| 경로 | 내용 |
|---|---|
| [`faceguard_api`](faceguard_api) | FastAPI 기반 얼굴 동일인 비교 API |
| [`notebooks`](notebooks) | Colab·Kaggle 재현 노트북 |
| [`scripts`](scripts) | 데이터 점검·전처리·학습·평가 도구 |
| [`configs/faceguard`](configs/faceguard) | 분할·평가 프로토콜 설정 |
| [`configs/deepfake`](configs/deepfake) | 딥페이크 기준선·후보 비교의 고정 실험 설정 |
| [`reports`](reports) | 개인정보를 제외한 집계 결과와 그래프 |
| [`tests`](tests) | 누수·지표·API·노트북 재현 테스트 |
| [`docs`](docs) | 목적별 사용 가이드·데모·실험 문서와 읽는 순서 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 브랜치·커밋·PR·금지 산출물 규칙 |

## 로컬 테스트

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api-test.txt
python -m unittest discover -s tests
python scripts/check_repository_hygiene.py
```

## 데이터·개인정보·라이선스

- 얼굴 영상·이미지, 얼굴 crop, 임베딩, 영상별 점수, checkpoint와 ONNX는 GitHub에 올리지 않는다.
- 공개 웹 검색은 접근이 허용된 공개 영역만 대상으로 하고 비공개 계정이나 접근 제한 영역을 자동 탐색하지 않는다.
- 원본 사진과 얼굴 특징은 API 요청 처리 후 영구 저장하지 않는다.
- InsightFace 코드와 제공 사전학습 가중치의 사용 조건은 다르다.
- 현재 `buffalo_l`와 얼굴 검출 가중치는 비상업 연구 범위로만 사용한다.
- 현재 얼굴 기준값과 딥페이크 기준값은 모두 연구용이다. K-FACE로 한국인 저·중화질 통제 조건은 검증했지만, 실제 웹 재압축·최신 생성 방식의 운영 성능을 보장하지 않는다.

## 다음 작업 순서

모델링은 아래 순서로 진행한다. 새 모델의 성능 수치는 실제 실행
후에만 기록하고, 각 단계는 같은 분할과 같은 지표로 비교한다.

1. K-FACE 저화질→중화질 임베딩 보정 어댑터 비교 완료·미채택 ([#44](https://github.com/Chunbae-A/face-image/issues/44))
2. 저해상도 얼굴 crop 기반 인식 미세 조정과 새로운 외부 잠긴 Test 설계 ([#47](https://github.com/Chunbae-A/face-image/issues/47))
3. EfficientNet-B4를 재현하고 Xception을 같은 조건에서 비교 ([#29](https://github.com/Chunbae-A/face-image/issues/29))
4. SBI로 보지 못한 조작 방식의 일반화 성능 검증 ([#30](https://github.com/Chunbae-A/face-image/issues/30))
5. 실제 영상 hard negative와 외부 검증 subset 구축 ([#31](https://github.com/Chunbae-A/face-image/issues/31))
6. 결승 후보에 FTCN 시간 정보와 앙상블 추가 효과 검증 ([#32](https://github.com/Chunbae-A/face-image/issues/32))

제품 파이프라인은 모델링과 병행해 다음 순서로 이어간다.

1. 검색 이미지 다중 얼굴 처리와 영상 얼굴 트랙 비교 구현 ([#14](https://github.com/Chunbae-A/face-image/issues/14))
2. 비동기 스캔에 후보 영상 URL 다운로드와 재시작 복구용 영구 큐 연결 ([#17](https://github.com/Chunbae-A/face-image/issues/17))
3. 검색·선별·이미지·영상 ONNX 연결 결과를 실제 동의 데이터로 수치 검증 ([#18](https://github.com/Chunbae-A/face-image/issues/18))

## 발표용 한 문장

> 딥소각 얼굴가드는 등록 얼굴과 공개 후보의 동일인 가능성을 ArcFace로 선별하고, Celeb-DF-v2 전체로 학습한 별도 모델이 후보 영상의 딥페이크 위험을 분석하도록 설계했습니다. 현재 얼굴 비교 API와 딥페이크 학습·평가는 완료했지만 오경고율이 목표를 넘었기 때문에 연구 결과로 공개하고 운영 적용은 보류했습니다.
