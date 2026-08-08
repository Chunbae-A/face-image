# 얼굴가드 API 빠른 실행

얼굴 비교·검색 API는 GitHub 저장소만 있으면 실행할 수 있다. 딥페이크 분석에는 GitHub에 올리지 않은 **팀 비공개 ONNX 파일**이 추가로 필요하며, Celeb-DF 원본 데이터셋은 필요하지 않다.

## 준비물

- Docker Desktop
- Git
- 인터넷 연결
- 동의받은 테스트용 얼굴 사진

## 1. 저장소 받기

```bash
git clone https://github.com/Chunbae-A/face-image.git
cd face-image
```

이미 저장소를 받았다면 `cd face-image`만 실행한다.

## 2. 환경설정

Mac/Linux:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

`.env` 파일에서 InsightFace 제공 가중치의 비상업 연구 조건을 직접 확인한 뒤 다음 값을 변경한다.

```text
FACEGUARD_ACCEPT_NONCOMMERCIAL_MODEL_LICENSE=true
```

딥페이크 분석까지 실행할 팀원은 권한이 있는 비공개 모델 ZIP에서 ONNX만 꺼낸다.

```bash
mkdir -p .models/deepfake
unzip -j /권한있는/경로/celebdf_deepfake_private_model.zip \
  efficientnet_b4.onnx -d .models/deepfake
shasum -a 256 .models/deepfake/efficientnet_b4.onnx
```

해시는 `c32a8532e2e1bd275b833b16460946eb307207098e0c07e2247851b71c23a6f1`이어야 한다. ONNX는 `.gitignore` 대상이므로 커밋하지 않는다.

## 3. API 실행

Docker Desktop을 켠 상태에서 실행한다.

```bash
docker compose up --build --detach
```

처음에는 Docker 이미지와 얼굴인식 모델을 준비하므로 몇 분 걸릴 수 있다.

## 4. 상태 확인

```bash
curl http://127.0.0.1:8000/health
```

응답에 `"status":"ok"`가 나오면 서버는 정상이다. 두 모델은 첫 요청 때 지연 로딩하므로 초기 `model_loaded`와 `deepfake_model_loaded`가 `false`여도 정상이다.

## 5. 얼굴 비교 시험

브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)를 연다.

1. `POST /v1/faceguard/verify` 선택
2. `Try it out` 선택
3. `reference_images`에 등록 얼굴 1~5장 선택 — 3장 권장
4. `query_image`에 확인 얼굴 1장 선택
5. `Execute` 선택

`is_same_person`은 동일인 **후보 판정**이며 운영 본인인증 확정값이 아니다. `raw_score`는 확률이 아니고, 현재 얼굴 보정 데이터가 없으므로 `calibrated_probability=null`, `calibration_status=not_available`로 반환한다. `decision_threshold`는 판정에 사용한 연구 기준값이다.

## 6. 딥페이크 단일 이미지 시험

같은 Swagger 화면에서 `POST /v1/deepfake/analyze`를 선택한다.

1. `image`에 얼굴 한 명이 선명하게 나온 이미지를 넣는다.
2. `Execute`를 누른다.
3. `raw_score`, `calibrated_probability`, `is_suspected_deepfake`, `threshold_status`를 확인한다.

`deepfake_score`와 `raw_score`는 같은 0~1 모델 원점수지만 보정된 확률이나 정확도 신뢰도가 아니다. 현재 `0.751988...` 기준은 영상 16프레임 평균에서 선택한 값이므로 단일 이미지 응답은 항상 `calibrated_probability=null`, `calibration_status=not_applicable_single_image`로 표시한다.

## 7. 딥페이크 영상 시험

같은 Swagger 화면에서 `POST /v1/deepfake/analyze-video`를 선택한다.

1. `video`에 최대 50MB·120초의 MP4 또는 MOV 영상을 넣는다.
2. 영상에 여러 사람이 나오면 `reference_images`에 분석할 사람의 등록 사진을 넣는다. 3장을 권장한다.
3. `Execute`를 누른다.
4. `raw_score`, `calibrated_probability`, `calibration_status`, `risk_level`, `analyzed_frame_count`, `suspicious_segments`를 확인한다.

영상 전체에서 최대 16개 대표 프레임을 고르고, 분석 가능한 얼굴 프레임의 점수를 평균한다. `suspicious_segments`는 정밀한 원본 프레임 탐지가 아니라 표본 프레임을 기준으로 만든 **검토 권장 시간대**다. 등록 사진이 없으면 첫 대표 프레임에서 가장 큰 얼굴을 기준으로 다음 프레임을 추적한다.

`video_score`와 `raw_score`도 보정된 확률이 아니다. 보정 파일이 없으면 `calibration_status=not_available`이고, 파일은 있지만 Gate를 통과하지 못하면 `research_only_unapproved`다. 두 경우 모두 `calibrated_probability=null`이므로 화면에 `85% 확률`처럼 표시하지 않는다.

## 8. 무료 공개 URL 후보 시험

같은 Swagger 화면에서 `POST /v1/search/candidates`를 선택하고 다음 JSON을 넣는다.

```json
{
  "privacy_mode": "privacy_strict",
  "web_monitoring_consent": false,
  "candidates": [
    {
      "page_url": "https://example.com/public-post",
      "media_url": "https://cdn.example.com/public-video.mp4"
    }
  ]
}
```

이 기능은 공개 URL을 안전하게 정리하고 중복을 제거한다. 인터넷에서 새 후보를 자동으로 찾는 역이미지 검색 기능은 아직 연결되지 않았다.

## 9. 무료 키워드 검색 시험

실행 중인 기본 API를 먼저 종료하고 SearXNG 결합 구성을 켠다.

```bash
docker compose down
docker compose -f docker-compose.yml -f docker-compose.searxng.yml up --build --detach
```

`POST /v1/search/candidates`에 다음 JSON을 넣는다. 검색어에는 공개 검색에 동의한 표현만 사용한다.

```json
{
  "privacy_mode": "web_monitoring",
  "web_monitoring_consent": true,
  "query_text": "동의받은 검색어",
  "categories": ["images"],
  "language": "ko-KR",
  "safe_search": 2,
  "maximum_results": 10,
  "candidates": []
}
```

응답의 `provider`가 `searxng`이면 정상이다. 이 기능은 검색어로 후보 URL을 모으는 단계이며 얼굴 사진 역검색이나 동일인 판정은 하지 않는다.

## 10. 검색부터 얼굴 선별·딥페이크 분석까지 한 번에 시험

같은 Swagger 화면에서 `POST /v1/pipeline/search-and-filter`를 선택한다.

1. `reference_images`에 동의받은 등록 얼굴 3장을 넣는다.
2. `query_text`에 공개 검색에 동의한 검색어를 넣는다.
3. `web_monitoring_consent`를 `true`로 설정한다.
4. `maximum_results`는 처음에는 `3`으로 설정한다.
5. `Execute`를 누른다.

결과에서 확인할 값은 다음과 같다.

- `similarity_raw`: 등록 얼굴과 후보 얼굴의 코사인 유사도 원값
- `retrieval_match`: 넓은 후보수집 기준을 통과했는지
- `identity_match`: 더 엄격한 연구용 동일인 기준을 통과했는지
- `quality_summary`: 후보 얼굴 크기·선명도·밝기
- `deepfake.status`: `analyzed`, `not_analyzed`, `failed`, `unavailable` 중 분석 상태
- `deepfake.deepfake_score`: 넓은 얼굴 후보 기준을 통과한 이미지의 ONNX 점수
- `deepfake.is_suspected_deepfake`: 연구 기준값을 넘었는지
- `error_code`: 다운로드·얼굴 검출에 실패한 이유

현재 `retrieval_threshold=0.20`은 기능 연결용 임시값이며 정확도 검증을 마친 운영값이 아니다. ArcFace가 후보가 아니라고 판단한 이미지는 불필요한 ONNX 경고를 피하려고 `deepfake.status=not_analyzed`로 남긴다. 여러 얼굴이 있는 이미지와 영상 후보는 아직 이 경로에서 처리하지 않는다.

## 11. `scan_id` 비동기 데모

화면이 검색 처리를 기다리며 멈추지 않게 하려면 `POST /v1/faceguard/enrollments`로 얼굴을 임시 등록한 뒤 `POST /v1/exposure-scans`로 작업을 시작한다. 서버가 즉시 반환한 `scan_id`를 `GET /v1/exposure-scans/{scan_id}`에 넣어 진행 상태를 확인한다.

초보자용 네 단계 Swagger 순서와 복사할 JSON은 [비동기 노출 스캔 안내](async-exposure-scan.md)에 있다. 현재 비동기 경로는 공개 이미지 후보만 지원하고, 서버를 재시작하면 메모리의 등록·작업·결과가 사라진다.

## 12. 종료

```bash
docker compose -f docker-compose.yml -f docker-compose.searxng.yml down
```

## 주의사항

- 현재 API와 `buffalo_l` 가중치는 연구·해커톤 데모용이다.
- `threshold_status`가 `research_only_unapproved`이므로 계정 복구·삭제 같은 작업을 자동 승인하지 않는다.
- 동의받지 않은 얼굴 사진을 사용하거나 저장소에 업로드하지 않는다.
- 로그인·TLS·요청 제한이 없으므로 API를 인터넷에 직접 공개하지 않는다.

상세 설정과 오류 코드는 [API 운영 가이드](operations.md)를 참고한다.
