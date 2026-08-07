# 딥소각 얼굴가드

사용자가 등록한 얼굴을 기준으로 **공개 웹 후보에서 같은 사람을 골라내고, 후보 영상의 딥페이크 위험을 수치와 근거로 보여주기 위한 모델링·API 프로젝트**다.

> 현재 바로 실행할 수 있는 범위는 ArcFace 얼굴 동일인 비교, SearXNG 검색어 기반 공개 이미지 후보 수집, 후보 이미지 다운로드·동일인 가능성 선별 API다. 얼굴 사진 자체로 웹을 찾는 역이미지 검색, 영상 얼굴 트랙, 딥페이크 판별 통합은 개발 중이며 현재 모델은 운영·본인인증·자동 차단 용도로 승인되지 않았다.

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

## 지금 어디까지 됐나요?

| 기능 | 하는 일 | 현재 상태 |
|---|---|---|
| 공개 후보 검색 | 공개 웹에서 이미지·영상 URL과 출처 수집 | **무료 URL 제보 + SearXNG 키워드 검색 구현, 얼굴 역검색 미연결** ([#13](https://github.com/Chunbae-A/face-image/issues/13)) |
| 얼굴 후보 선별 | 등록 얼굴과 후보 얼굴의 동일인 가능성 비교 | **검색 이미지 다운로드·ArcFace 배치 연결 완료, 다중 얼굴·영상 트랙 미연결** ([#14](https://github.com/Chunbae-A/face-image/issues/14)) |
| 딥페이크 판별 | 후보 영상의 얼굴 프레임이 실제인지 조작인지 분석 | **연구 기준선·ONNX 완료, 운영 Gate 미통과** ([#15](https://github.com/Chunbae-A/face-image/issues/15)) |
| 화면용 신뢰도 | 얼굴 유사도와 딥페이크 점수를 사용자용 수치로 보정 | 구현 예정 ([#16](https://github.com/Chunbae-A/face-image/issues/16)) |
| 통합 비동기 API | 검색 → 얼굴 선별 → 딥페이크 판별을 하나의 작업으로 연결 | 구현 예정 ([#17](https://github.com/Chunbae-A/face-image/issues/17)) |

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

재현 방법은 [`DEEPFAKE_KAGGLE_RUNBOOK.md`](DEEPFAKE_KAGGLE_RUNBOOK.md), 데이터 분할과 사전검사·현재 결과는 [`reports/celebdf_deepfake_baseline/2026-08-07`](reports/celebdf_deepfake_baseline/2026-08-07)에 있다.

## 얼굴가드 API 실행

현재 HTTP API는 **얼굴 동일인 후보 선별**, **공개 URL 정규화**, **SearXNG 후보 수집**, **검색 이미지 → ArcFace 선별 통합 경로**를 제공한다. 얼굴 역이미지 검색, 영상 트랙, 딥페이크 ONNX 통합은 아직 구현 전이다.

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

서버를 실행한다.

```bash
docker compose up --build --detach
curl http://127.0.0.1:8000/health
```

무료 키워드 검색까지 함께 켜려면 기본 명령 대신 다음 결합 구성을 실행한다.

```bash
docker compose -f docker-compose.yml -f docker-compose.searxng.yml up --build --detach
```

브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)를 열고 `POST /v1/faceguard/verify`를 시험한다.

- `reference_images`: 본인 등록 얼굴 1~5장, **3장 권장**
- `query_image`: 확인할 후보 얼굴 1장
- 지원 형식: JPEG, PNG, WEBP

응답의 핵심값은 다음과 같다.

| 필드 | 의미 |
|---|---|
| `is_same_person` | 연구 기준값을 넘었는지 여부 |
| `similarity` | 두 얼굴의 코사인 유사도 |
| `threshold` | 현재 연구 판정 기준값 |
| `reference_quality`, `query_quality` | 얼굴 크기·선명도·밝기 등 입력 품질 |
| `processing_ms` | 서버 내부 처리시간 |

`is_same_person=true`는 “같은 사람 후보”라는 뜻이지 딥페이크라는 뜻이 아니다. 자세한 실행과 오류 코드는 [`API_QUICKSTART.md`](API_QUICKSTART.md)와 [`API_RUNBOOK.md`](API_RUNBOOK.md)에 있다.

공개 후보 경로는 같은 Swagger 화면의 `POST /v1/search/candidates`에서 시험한다.

- `privacy_strict`: 사용자가 직접 넣은 공개 URL만 정리한다. 로컬·내부망 주소를 차단하고 추적 파라미터를 제거한 뒤 중복을 합친다.
- `web_monitoring`: 명시적 동의 후 검색어를 로컬 SearXNG에 보내 공개 이미지·영상 후보를 찾는다. 얼굴 사진은 보내지 않는다.

SearXNG은 **검색어 기반 메타검색**이다. 등록 얼굴 사진과 닮은 웹 사진을 자동으로 찾는 얼굴 역검색은 아니며, 찾은 후보가 본인인지와 딥페이크인지는 다음 ArcFace·ONNX 단계에서 별도로 검사해야 한다. 자세한 실행법은 [`SEARXNG_RUNBOOK.md`](SEARXNG_RUNBOOK.md)에 있다.

검색과 얼굴 선별을 한 번에 시험할 때는 `POST /v1/pipeline/search-and-filter`를 사용한다. 등록 사진은 로컬 ArcFace에만 입력되고, SearXNG에는 검색어만 전달된다. 결과에는 후보별 `similarity_raw`, 넓은 후보 기준 통과 여부, 최종 동일인 기준 통과 여부, 이미지 품질과 실패 코드가 포함된다.

## 데모에서 보여줄 내용

현재 안전하게 시연할 수 있는 데모는 다음과 같다.

1. 동의받은 사람의 등록 사진 3장을 넣는다.
2. 같은 사람의 다른 사진을 넣어 유사도와 품질 수치를 확인한다.
3. 다른 사람의 사진을 넣어 불일치 결과를 확인한다.
4. Celeb-DF 전체 딥페이크 실험 결과에서 AUC와 FPR을 함께 보여준다.
5. “AUC는 높지만 FPR Gate를 통과하지 못해 운영 승인하지 않았다”고 설명한다.

검색 → 얼굴 선별 → 딥페이크 판별의 완전한 데모는 [Issue #17](https://github.com/Chunbae-A/face-image/issues/17)의 통합 API가 완료된 뒤 제공한다. 화면 흐름과 오류 처리는 [`DEMO_PIPELINE.md`](DEMO_PIPELINE.md)에 있다.

## 저장소 안내

| 경로 | 내용 |
|---|---|
| [`faceguard_api`](faceguard_api) | FastAPI 기반 얼굴 동일인 비교 API |
| [`notebooks`](notebooks) | Colab·Kaggle 재현 노트북 |
| [`scripts`](scripts) | 데이터 점검·전처리·학습·평가 도구 |
| [`configs/faceguard`](configs/faceguard) | 분할·평가 프로토콜 설정 |
| [`reports`](reports) | 개인정보를 제외한 집계 결과와 그래프 |
| [`tests`](tests) | 누수·지표·API·노트북 재현 테스트 |
| [`DEEPFAKE_BASELINE_RUNBOOK.md`](DEEPFAKE_BASELINE_RUNBOOK.md) | 딥페이크 모델 명령 구조 설명 |
| [`DEEPFAKE_KAGGLE_RUNBOOK.md`](DEEPFAKE_KAGGLE_RUNBOOK.md) | Kaggle 무료 GPU 실행 순서 |
| [`SEARXNG_RUNBOOK.md`](SEARXNG_RUNBOOK.md) | 무료 키워드 검색 실행·시험·한계 |
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
- 현재 얼굴 기준값과 딥페이크 기준값은 모두 연구용이며 한국인·최신 생성 방식·실제 웹 재압축 영상의 운영 성능을 보장하지 않는다.

## 다음 작업 순서

1. 검색 이미지의 다중 얼굴 처리와 영상 얼굴 트랙 비교 구현 ([#14](https://github.com/Chunbae-A/face-image/issues/14))
2. 넓은 후보 기준값을 공개 웹 validation 데이터로 보정 ([#14](https://github.com/Chunbae-A/face-image/issues/14))
3. 얼굴 역이미지 검색 제공자 후보와 Recall·비용을 별도 검증 ([#13](https://github.com/Chunbae-A/face-image/issues/13))
4. 얼굴·딥페이크 기준값과 품질 Gate 보정 ([#16](https://github.com/Chunbae-A/face-image/issues/16))
5. 연구용 딥페이크 ONNX 추론을 통합 API에 연결 ([#17](https://github.com/Chunbae-A/face-image/issues/17))
6. 검색·선별·판별 전체 데모 수치 검증 ([#18](https://github.com/Chunbae-A/face-image/issues/18))

## 발표용 한 문장

> 딥소각 얼굴가드는 등록 얼굴과 공개 후보의 동일인 가능성을 ArcFace로 선별하고, Celeb-DF-v2 전체로 학습한 별도 모델이 후보 영상의 딥페이크 위험을 분석하도록 설계했습니다. 현재 얼굴 비교 API와 딥페이크 학습·평가는 완료했지만 오경고율이 목표를 넘었기 때문에 연구 결과로 공개하고 운영 적용은 보류했습니다.
