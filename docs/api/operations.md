# 딥소각 얼굴가드 API 사용 가이드

이 API는 **등록 얼굴 사진과 공개 후보의 동일인 가능성을 비교**하고, 이미지 또는 짧은 영상의 얼굴을 Celeb-DF EfficientNet-B4 ONNX로 분석한다. 실제 얼굴 비교에서는 등록 사진 3장을 권장한다.

현재 버전은 연구·해커톤 검증용이다. Celeb-real 기준선의 관측 오인식률이 목표보다 높았으므로 응답의 `threshold_status`는 항상 `research_only_unapproved`로 표시한다. 한국인 얼굴·실제 휴대전화 데이터 검증과 모델 가중치의 상용 사용권 해결 전에는 운영 본인인증 수단으로 사용하지 않는다.

## API가 하는 일

```text
등록 사진 3장 ─┐
               ├─ 얼굴 하나 확인 → ArcFace 특징 평균 ─┐
확인 사진 1장 ─┘                                      ├─ 코사인 유사도 → 동일인 후보 판정
                                                      ┘
```

- 사진마다 얼굴이 정확히 한 명인지 확인한다.
- 얼굴이 없거나 여러 명이면 임의로 선택하지 않고 거절한다.
- 얼굴이 너무 작거나 탐지 신뢰도가 낮으면 재촬영 오류를 반환한다.
- 원본 사진, 얼굴 crop, 512차원 임베딩을 애플리케이션의 영구 파일·DB·응답·로그에 저장하지 않는다.
- 업로드 본문은 요청 처리 중 메모리 또는 프레임워크의 임시 버퍼에만 존재하며, 처리 후 닫는다.
- 딥페이크 모델 점수는 보정된 확률이 아니며 단일 이미지 분석을 영상 판정 정확도로 표현하지 않는다.

## 가장 빠른 로컬 실행

Python 3.11 환경을 권장한다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
```

InsightFace `buffalo_l` 가중치의 비상업 연구 조건을 직접 확인한 경우에만 다음 값을 설정한다.

```bash
export FACEGUARD_ACCEPT_NONCOMMERCIAL_MODEL_LICENSE=true
export FACEGUARD_DEVICE=auto
export FACEGUARD_DEEPFAKE_MODEL_PATH=.models/deepfake/efficientnet_b4.onnx
python -m uvicorn faceguard_api.app:app --host 127.0.0.1 --port 8000
```

딥페이크 경로에는 권한이 있는 비공개 모델 ZIP의 `efficientnet_b4.onnx`가 있어야 한다. API는 실행 전에 SHA-256 `c32a8532e2e1bd275b833b16460946eb307207098e0c07e2247851b71c23a6f1`을 검증하며 ONNX를 GitHub에 커밋하지 않는다.

영상 점수 보정 실험 결과가 있으면 `deepfake_video_calibration.json`도 같은 디렉터리에 둔다. 이 파일이 없거나 검증 Gate를 통과하지 못한 경우 API는 보정 확률 없이 원점수와 `calibration_status`, `calibration_version`, `risk_level`, `warning`을 반환하고, 화면용 확률인 `calibrated_probability`는 `null`로 유지한다.

첫 추론에서는 모델 파일을 내려받고 준비하므로 이후 요청보다 오래 걸릴 수 있다. 서버가 켜지면 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)에서 요청을 직접 시험할 수 있다.

## Docker 실행

```bash
cp .env.example .env
```

`.env`에서 이용 조건 확인값을 `true`로 바꾼 뒤 실행한다.

딥페이크 API까지 사용할 때는 호스트의 `.models/deepfake/efficientnet_b4.onnx`를 준비한다. Compose는 이 디렉터리를 컨테이너 `/models/deepfake`에 읽기 전용으로 마운트한다.

```bash
docker compose up --build
```

무료 SearXNG 키워드 검색도 함께 사용할 때는 다음 결합 구성을 실행한다.

```bash
docker compose -f docker-compose.yml -f docker-compose.searxng.yml up --build
```

SearXNG는 Docker 내부망에서만 열리고 호스트 포트를 공개하지 않는다. 검색어만 외부 검색엔진으로 전달하며 등록 얼굴 사진은 보내지 않는다.

기본 Docker 이미지는 CPU용이다. Linux CUDA 서버에서는 `requirements-api-gpu.txt`의 `onnxruntime-gpu`를 사용하고 NVIDIA Container Runtime을 별도로 설정한다.

영상 엔드포인트는 ASGI middleware에서 전체 요청을 기본 `91MiB`로 제한한다. 이는 영상 50MB, 등록 사진 5장×8MB와 multipart 여유 1MiB를 합친 값이며 `FACEGUARD_MAX_VIDEO_REQUEST_BYTES`로 조정한다. 인터넷에 배치할 때는 애플리케이션에 도달하기 전에 프록시에서도 같은 제한을 둔다. Nginx 예시는 다음과 같다.

```nginx
location /v1/deepfake/analyze-video {
    client_max_body_size 91m;
    proxy_pass http://faceguard-api:8000;
}
```

`Content-Length`가 있으면 multipart 파싱 전에 거절하고, chunked 요청도 수신 누적량이 제한을 넘는 즉시 중단한다. 파일별 50MB 영상·8MB 등록 사진 검사는 보조 방어로 그대로 적용한다.

## 1. 서버 상태 확인

```bash
curl http://127.0.0.1:8000/health
```

예시:

```json
{
  "status": "ok",
  "api_version": "0.8.0",
  "model_name": "buffalo_l",
  "model_loaded": false,
  "execution_provider": null,
  "model_fingerprint": null,
  "license_accepted": true,
  "threshold_status": "research_only_unapproved",
  "search_providers": ["user_url", "searxng"],
  "web_search_enabled": true,
  "deepfake_model_name": "efficientnet_b4_celebdf_v2",
  "deepfake_model_loaded": false,
  "deepfake_execution_provider": null,
  "deepfake_model_fingerprint": null,
  "deepfake_threshold_status": "research_only_single_image_unvalidated",
  "deepfake_video_threshold_status": "research_only_unapproved",
  "deepfake_video_calibration_status": "not_available",
  "deepfake_video_calibration_version": null
}
```

`model_loaded`와 `deepfake_model_loaded`가 `false`인 것은 아직 첫 추론을 하지 않아 모델을 지연 로딩하는 상태다. 기본 Compose에서는 `search_providers`가 `["user_url"]`, SearXNG 결합 구성에서는 `["user_url", "searxng"]`이다.

## 2. 동일인 확인 요청

같은 `reference_images` 이름을 반복해 등록 사진을 보낸다.

```bash
curl -X POST http://127.0.0.1:8000/v1/faceguard/verify \
  -F "reference_images=@./samples/register-1.jpg" \
  -F "reference_images=@./samples/register-2.jpg" \
  -F "reference_images=@./samples/register-3.jpg" \
  -F "query_image=@./samples/query.jpg"
```

응답 예시:

```json
{
  "request_id": "58a8762f-5d64-49f4-bde7-cd12d89f9236",
  "is_same_person": true,
  "similarity": 0.71,
  "raw_score": 0.71,
  "calibrated_probability": null,
  "calibration_status": "not_available",
  "calibration_version": null,
  "threshold": 0.2823836207389832,
  "decision_threshold": 0.2823836207389832,
  "threshold_status": "research_only_unapproved",
  "threshold_source": "Celeb-real 기준선 5프레임·등록 3개, FAR 0.001 목표의 seed별 기준값 최댓값",
  "warning": "현재 판정 기준값은 Celeb-real 연구 기준선이며 운영 확정값이 아닙니다.",
  "reference_count": 3,
  "recommended_reference_count": 3,
  "reference_quality": [
    {
      "detection_score": 0.99,
      "face_area_ratio": 0.24,
      "blur_score": 183.2,
      "brightness_mean": 126.7,
      "image_width": 1280,
      "image_height": 960
    }
  ],
  "query_quality": {
    "detection_score": 0.98,
    "face_area_ratio": 0.21,
    "blur_score": 170.4,
    "brightness_mean": 121.5,
    "image_width": 1280,
    "image_height": 960
  },
  "processing_ms": 182.4,
  "model_name": "buffalo_l",
  "execution_provider": "CPUExecutionProvider",
  "model_fingerprint": "실행한 탐지·인식 ONNX 파일을 묶어 계산한 SHA-256"
}
```

실제 응답의 `reference_quality`에는 등록 사진 수만큼 품질 정보가 들어간다. 위 예시는 읽기 쉽게 한 장만 표시했다.

## 3. 단일 얼굴 이미지 딥페이크 분석

```bash
curl -X POST http://127.0.0.1:8000/v1/deepfake/analyze \
  -F "image=@./samples/candidate.jpg"
```

핵심 응답은 다음과 같다.

```json
{
  "status": "completed",
  "is_suspected_deepfake": true,
  "deepfake_score": 0.83,
  "raw_score": 0.83,
  "calibrated_probability": null,
  "calibration_status": "not_applicable_single_image",
  "calibration_version": null,
  "risk_level": null,
  "raw_logit": 1.59,
  "threshold": 0.7519882693886758,
  "decision_threshold": 0.7519882693886758,
  "threshold_status": "research_only_single_image_unvalidated",
  "inference_ms": 171.0,
  "model_name": "efficientnet_b4_celebdf_v2",
  "execution_provider": "CPUExecutionProvider",
  "model_fingerprint": "c32a8532e2e1bd275b833b16460946eb307207098e0c07e2247851b71c23a6f1",
  "config_version": "deepfake-single-image-v1"
}
```

- `deepfake_score`: sigmoid를 적용한 0~1 모델 점수다. 보정된 확률이나 UI 신뢰도가 아니다.
- `is_suspected_deepfake`: 연구 기준값을 넘었는지만 표시한다.
- `raw_logit`: sigmoid 적용 전 모델 출력으로 재현성과 디버깅에 사용한다.
- `threshold_status`: 영상 16프레임 평균 기준을 단일 이미지에 임시 재사용했음을 명시한다.
- `quality_summary`: 얼굴 검출 신뢰도, 얼굴 면적, 선명도와 밝기다.

처리 순서는 `SCRFD 얼굴 검출 → 5점 랜드마크 정렬(224×224) → 380×380 Resize → ImageNet 정규화 → EfficientNet-B4 ONNX → sigmoid`다. 이미지·정렬 crop·모델 입력 텐서는 저장하지 않는다.

## 4. 짧은 영상 16프레임 딥페이크 분석

등록 사진 없이 주인공 얼굴을 추적하려면 다음처럼 호출한다.

```bash
curl -X POST http://127.0.0.1:8000/v1/deepfake/analyze-video \
  -F "video=@./samples/candidate.mp4"
```

영상에 여러 사람이 나오고 특정 사람만 분석하려면 등록 사진을 함께 보낸다.

```bash
curl -X POST http://127.0.0.1:8000/v1/deepfake/analyze-video \
  -F "video=@./samples/candidate.mp4" \
  -F "reference_images=@./samples/register-1.jpg" \
  -F "reference_images=@./samples/register-2.jpg" \
  -F "reference_images=@./samples/register-3.jpg"
```

핵심 응답은 다음과 같다.

```json
{
  "status": "completed",
  "is_suspected_deepfake": true,
  "video_score": 0.84,
  "raw_score": 0.84,
  "calibrated_probability": null,
  "calibration_status": "research_only_unapproved",
  "calibration_version": "celebdf-video-calibration-v1",
  "risk_level": "high",
  "threshold": 0.7519882693886758,
  "decision_threshold": 0.7519882693886758,
  "threshold_status": "research_only_unapproved",
  "aggregation": "mean",
  "requested_frame_count": 16,
  "analyzed_frame_count": 15,
  "skipped_frame_count": 1,
  "reference_count": 3,
  "suspicious_segments": [
    {
      "start_seconds": 4.2,
      "end_seconds": 7.8,
      "peak_score": 0.94,
      "analyzed_frame_count": 2
    }
  ],
  "config_version": "deepfake-video-16-frame-mean-v1"
}
```

- MP4·MOV, 최대 50MB·120초만 허용한다.
- 학습 전처리와 같이 영상 시작·끝을 조금 피하고 최대 16개 프레임을 균등 추출한다.
- 등록 사진이 있으면 프레임의 여러 얼굴 중 등록 대표 얼굴과 가장 비슷한 얼굴을 고른다.
- 등록 사진이 없으면 첫 프레임의 가장 큰 얼굴을 기준으로 다음 프레임을 추적한다.
- 유효 얼굴이 4프레임 미만이면 점수를 만들지 않고 오류를 반환한다.
- `video_score`는 유효 프레임 점수의 평균이며 보정된 확률이 아니다.
- `raw_score`는 `video_score`와 같은 모델 원점수다. 기존 클라이언트 호환을 위해 두 이름을 함께 제공한다.
- `calibrated_probability`는 별도 validation·공식 test Gate를 통과한 경우에만 숫자를 반환한다. `null`이면 퍼센트로 표시하지 않는다.
- `risk_level`은 validation에서 정한 검토 우선순위이지 확률이 아니다. 보정 파일의 `review_band_empty=true`이면 두 목표 경계가 만나 별도 `review` 구간이 없으므로 `low` 또는 `high`만 반환한다.
- `suspicious_segments`는 표본 프레임 주변을 묶은 **대략적인 검토 시간대**이지 모든 원본 프레임을 정밀 판독한 구간이 아니다.
- 요청 영상은 임시 디렉터리에서 디코딩하고 요청 종료 전에 삭제한다. 얼굴 crop과 임베딩은 영구 저장하지 않는다.

## 5. 무료 공개 URL 후보 정규화

외부 검색 API 키 없이 사용자가 알고 있는 공개 페이지·미디어 URL을 공통 후보 형식으로 바꿀 수 있다. 이 경로는 얼굴 모델 가중치를 실행하지 않으므로 InsightFace 이용 조건 확인값과 무관하게 사용할 수 있다.

```bash
curl -X POST http://127.0.0.1:8000/v1/search/candidates \
  -H 'Content-Type: application/json' \
  -d '{
    "privacy_mode": "privacy_strict",
    "web_monitoring_consent": false,
    "candidates": [
      {
        "page_url": "https://example.com/public-post?utm_source=demo",
        "media_url": "https://cdn.example.com/public-video.mp4"
      },
      {
        "page_url": "https://example.com/reposted-content",
        "media_url": "https://cdn.example.com/public-video.mp4"
      }
    ]
  }'
```

응답 예시:

```json
{
  "request_id": "요청마다 생성되는 UUID",
  "status": "completed",
  "privacy_mode": "privacy_strict",
  "candidates": [
    {
      "page_url": "https://example.com/public-post",
      "media_url": "https://cdn.example.com/public-video.mp4",
      "thumbnail_url": null,
      "provider": "user_url",
      "providers": ["user_url"],
      "rank": 1,
      "retrieved_at": "2026-08-07T12:00:00Z"
    }
  ],
  "providers": [
    {
      "provider": "user_url",
      "status": "completed",
      "candidate_count": 2,
      "processing_ms": 0.2,
      "error_code": null
    }
  ],
  "raw_candidate_count": 2,
  "candidate_count": 1,
  "duplicate_count": 1,
  "truncated_count": 0,
  "processing_ms": 0.3,
  "warning": "현재 무료 모드는 인터넷 자동 검색이나 후보 발견을 완료했다는 뜻이 아닙니다."
}
```

안전 규칙은 다음과 같다.

- `http`, `https` 공개 주소만 허용한다.
- `localhost`, 사설·예약 IP, 내부 도메인, 기본값이 아닌 포트를 차단한다.
- URL에 아이디·비밀번호 또는 `token`, `api_key`, `signature` 같은 비밀 쿼리가 있으면 거절한다.
- `utm_*`, `fbclid`, `gclid` 같은 추적 쿼리와 URL fragment를 제거한다.
- 같은 미디어 URL을 SHA-256으로 비교하고, 제공된 콘텐츠 SHA-256 또는 64-bit pHash가 같으면 한 후보로 합친다.
- 콘텐츠 hash는 중복 제거에만 쓰며 공개 API 응답에 다시 내보내지 않는다.

`privacy_strict`에서는 외부 검색을 호출하지 않는다. `web_monitoring`은 사용자의 명시적 동의와 별도 외부 검색 제공자 설정이 모두 있을 때만 사용할 수 있다. 기본 Compose에는 외부 검색 제공자가 없으므로 `web_monitoring` 요청은 `SEARCH_PROVIDER_UNAVAILABLE`로 거절한다.

## 6. SearXNG 무료 키워드 검색

SearXNG 결합 Compose를 켠 뒤 다음처럼 호출한다.

```bash
curl -X POST http://127.0.0.1:8000/v1/search/candidates \
  -H 'Content-Type: application/json' \
  -d '{
    "privacy_mode": "web_monitoring",
    "web_monitoring_consent": true,
    "query_text": "동의받은 검색어",
    "categories": ["images", "videos"],
    "language": "ko-KR",
    "safe_search": 2,
    "maximum_results": 20,
    "candidates": []
  }'
```

- `query_text`: 외부 공개 검색에 동의한 검색어, 최대 200자
- `categories`: `images`, `videos` 중 하나 이상
- `safe_search`: 보통 `2` 권장
- `maximum_results`: 요청할 후보 수, 최대 50개
- `source_engine`: SearXNG가 결과를 받은 실제 검색 엔진 이름

API는 SearXNG의 결과 URL도 내부망·로컬 주소인지 다시 검사하고, 제공자 오류가 일부 발생하면 `partial_failed` 상태로 정상 후보만 반환한다. 입력 검색어는 API 응답에 되돌려 주지 않는다.

중요한 한계가 있다. SearXNG는 **검색어 기반 후보 수집기**이므로 얼굴 사진 자체로 같은 얼굴을 찾아주는 역이미지 검색이 아니다. 이미지 후보의 다운로드·ArcFace 선별은 아래 통합 API로 연결됐지만 다중 얼굴 이미지와 영상 트랙은 Issue #14, 얼굴 역검색 제공자 비교는 Issue #13의 후속 범위다.

## 7. 검색 이미지 → ArcFace → 딥페이크 ONNX 통합 API

`POST /v1/pipeline/search-and-filter`는 한 요청에서 다음 순서로 처리한다.

```text
등록 얼굴 먼저 로컬 검증
        ↓
검색어만 SearXNG로 전송
        ↓
후보 URL의 DNS·리다이렉트·형식·크기 검사
        ↓
후보 이미지를 메모리에서 ArcFace 비교
        ↓
retrieval_match=true 후보만 단일 이미지 ONNX 분석
        ↓
후보별 유사도·딥페이크 점수·품질·실패 코드 반환
```

Swagger에서 등록 얼굴 사진과 폼 값을 입력하는 방법은 [API 빠른 실행](quickstart.md)에 있다. 명령줄 예시는 다음과 같다.

```bash
curl -X POST http://127.0.0.1:8000/v1/pipeline/search-and-filter \
  -F "reference_images=@./samples/register-1.jpg" \
  -F "reference_images=@./samples/register-2.jpg" \
  -F "reference_images=@./samples/register-3.jpg" \
  -F "query_text=동의받은 검색어" \
  -F "web_monitoring_consent=true" \
  -F "maximum_results=3"
```

판정값은 얼굴 두 단계와 딥페이크 한 단계다.

- `retrieval_threshold=0.20`: 실제 본인 후보를 넓게 남기기 위한 **미보정 임시값**
- `identity_threshold=0.2823836207389832`: Celeb-real 연구 기준값이며 운영 미승인
- `deepfake_threshold=0.7519882693886758`: Celeb-DF **영상 평균** 연구 기준값을 단일 이미지에 임시 재사용

각 후보의 `deepfake.status`는 `analyzed`, `not_analyzed`, `failed`, `unavailable` 중 하나다. ArcFace 넓은 기준을 통과하지 않은 후보는 불필요한 오경고와 연산을 피하기 위해 `not_analyzed`로 남긴다. 모델이 없거나 실패하면 얼굴 유사도 결과는 유지하고 딥페이크 점수를 만들지 않으며 최상위 상태는 `partial_failed`가 된다. `identity_match=true`나 `is_suspected_deepfake=true`도 피해 사실 확정이 아니다. 등록 사진·후보 이미지·정렬 crop·임베딩은 응답이나 GitHub에 저장하지 않는다.

## 딥소각 백엔드 연결 규칙

- 브라우저 앱에서 모델 API를 직접 공개하지 말고 딥소각 백엔드가 서버 간 호출한다.
- 사용자 이름이나 계정 ID를 모델 API에 전달하지 않는다.
- HTTP 요청 로그에 multipart 본문과 파일명을 남기지 않는다.
- 첫 모델 로딩을 제외한 요청 제한시간은 실제 GPU/CPU 측정 후 정한다.
- `is_same_person=true` 하나만으로 삭제·복구 같은 민감 작업을 자동 승인하지 않는다.
- `threshold_status != approved`이면 연구 화면에서만 사용한다.
- 운영에서는 TLS, 서비스 인증, 요청 횟수 제한, 감사 로그를 API Gateway 또는 딥소각 백엔드에서 적용한다.

현재 API는 무상태이므로 매 요청에 등록 사진도 같이 보낸다. 사용자별 등록 임베딩을 장기 저장하려면 암호화, 키 관리, 보관 기간, 삭제권, 접근 기록을 먼저 설계하고 별도 저장 API로 구현한다.

## 현재 구현 검증 상태

2026-08-06 승인받은 Celeb-real 전용 ZIP에서 원본을 저장하지 않고 CPU 스모크 테스트를 수행했다.

- InsightFace `buffalo_l` 탐지·인식 모듈 로딩 성공
- ONNX Runtime `CPUExecutionProvider` 사용
- 512차원 임베딩과 L2 norm `1.0` 확인
- 등록 영상 3개 평균과 같은 사람 영상 1개: 유사도 `0.718976`, 동일인 후보 통과
- 같은 등록과 다른 사람 영상 1개: 유사도 `-0.047100`, 동일인 후보 거절
- 원본 프레임과 임베딩 저장 없음
- 비공개 EfficientNet-B4 ONNX SHA-256 일치와 CPU 유한 출력 smoke 통과
- 단일 이미지 API는 테스트용 가짜 모델 없이 실제 ONNX 로딩 경로 검증
- 4초 임시 MP4에서 대표 프레임 16/16개 실제 ONNX 분석과 평균 집계 통과
- 영상 연결 시험의 전체 처리 약 `5,293.5ms`, ONNX 추론 합계 약 `1,235.4ms`

이 값은 파이프라인이 끝까지 동작하는지 확인한 **1건의 스모크 테스트**다. 정확도, 오인식률, 한국인 일반화 또는 운영 안전성을 증명하는 결과가 아니다.

영상 연결 시험은 **2026-08-08 Asia/Seoul(KST)**에 사용 동의를 받은 실제 얼굴 이미지로 런타임에서만 4초 MP4를 만들었고 종료 즉시 삭제했다. 비식별 결과와 한계는 [`reports/video_deepfake_api_smoke/2026-08-08`](../../reports/video_deepfake_api_smoke/2026-08-08)에 기록했다.

추가로 Docker HTTP API에 승인받은 Celeb-real 전체 프레임을 직접 전송한 로컬 데모 스모크에서는 같은 사람 `0.694200`, 다른 사람 `-0.051950`이 관측됐다. 예열 후 CPU 처리시간은 두 요청에서 `962.706ms`, `1,188.254ms`였다. 개인정보를 제외한 실행 환경과 한계는 [`reports/faceguard_demo_smoke/2026-08-06`](../../reports/faceguard_demo_smoke/2026-08-06)에 기록했다.

## 주요 오류

| 오류 코드 | 뜻 | 사용자 안내 |
|---|---|---|
| `MODEL_LICENSE_NOT_ACCEPTED` | 모델 가중치 이용 조건 미확인 | 서버 설정 확인 |
| `UNSUPPORTED_CONTENT_TYPE` | 지원하지 않는 파일 형식 | JPEG·PNG·WEBP로 변환 |
| `UNSUPPORTED_VIDEO_CONTENT_TYPE` | 지원하지 않는 영상 형식 | MP4·MOV로 변환 |
| `IMAGE_TOO_LARGE` | 한 장이 8MB 초과 | 이미지 크기 축소 |
| `VIDEO_TOO_LARGE` | 영상이 50MB 초과 | 영상 길이·해상도 축소 |
| `VIDEO_TOO_LONG` | 영상이 120초 초과 | 짧은 구간으로 나눠 분석 |
| `REQUEST_BODY_TOO_LARGE` | 영상·등록 사진을 합친 요청이 91MiB 초과 | 파일 크기를 줄여 재시도 |
| `INVALID_VIDEO` | 손상되거나 디코딩 불가 | MP4·MOV로 다시 인코딩 |
| `INSUFFICIENT_VALID_VIDEO_FRAMES` | 유효 얼굴이 4프레임 미만 | 얼굴이 크고 밝은 영상 사용 |
| `TOO_MANY_PIXELS` | 해상도가 2천만 픽셀 초과 | 해상도 축소 |
| `NO_FACE` | 얼굴을 찾지 못함 | 정면에서 밝게 재촬영 |
| `MULTIPLE_FACES` | 얼굴이 둘 이상 있음 | 혼자 나온 사진 사용 |
| `INVALID_FACE_LANDMARKS` | 정렬용 눈·코·입 위치 부족 | 정면에 가까운 선명한 사진 사용 |
| `FACE_ALIGNMENT_FAILED` | ONNX 입력용 얼굴 정렬 실패 | 다른 얼굴 사진으로 재시도 |
| `FACE_TOO_SMALL` | 얼굴 면적이 너무 작음 | 카메라에 더 가까이 촬영 |
| `LOW_DETECTION_SCORE` | 얼굴이 불분명함 | 흔들림·가림을 없애고 재촬영 |
| `MODEL_UNAVAILABLE` | 모델 또는 실행 환경 문제 | 서버 로그와 모델 설치 확인 |
| `PRIVATE_NETWORK_URL_BLOCKED` | 내부망·로컬 URL 입력 | 접근 가능한 공개 URL 사용 |
| `URL_SECRET_PARAMETER_NOT_ALLOWED` | URL에 토큰·키로 보이는 쿼리 포함 | 비밀 쿼리를 제거한 공개 URL 사용 |
| `WEB_MONITORING_CONSENT_REQUIRED` | 외부 이미지 검색 동의 없음 | 개인정보 엄격 모드 사용 또는 별도 동의 |
| `SEARCH_PROVIDER_UNAVAILABLE` | 외부 검색 제공자 미설정 | 무료 URL 제보 모드 사용 |
| `ENROLLMENT_NOT_FOUND` | 임시 얼굴 등록 ID 없음 | 사진을 다시 등록 |
| `ENROLLMENT_EXPIRED` | 기본 30분의 등록 TTL 만료 | 사진을 다시 등록 |
| `ENROLLMENT_CAPACITY_EXCEEDED` | 활성 등록 저장소 포화 | 잠시 후 재시도 |
| `IDEMPOTENCY_KEY_REUSED` | 같은 멱등성 키를 다른 요청에 사용 | 새 요청 키 발급 |
| `SCAN_NOT_FOUND` | 스캔 ID 없음 | `scan_id` 다시 확인 |
| `SCAN_EXPIRED` | 기본 60분의 결과 TTL 만료 | 새 스캔 시작 |
| `SCAN_CAPACITY_EXCEEDED` | 비동기 작업 저장소 포화 | 잠시 후 재시도 |
| `CANDIDATE_DNS_FAILED` | 후보 이미지 도메인 확인 실패 | 다른 공개 후보 사용 |
| `CANDIDATE_DOWNLOAD_TIMEOUT` | 후보 이미지 다운로드 제한시간 초과 | 나중에 재시도 |
| `CANDIDATE_IMAGE_TOO_LARGE` | 후보 이미지가 8MB 초과 | 더 작은 공개 이미지 사용 |
| `UNSUPPORTED_CANDIDATE_CONTENT_TYPE` | 후보가 JPEG·PNG·WEBP가 아님 | 이미지 직접 URL 확인 |

후보 다운로드 코드는 최상위 HTTP 오류가 아니라 `status="partial_failed"` 응답의 `candidates[].error_code`에 들어간다. 후보별 딥페이크 실패는 `candidates[].deepfake.error_code`에 들어간다. 따라서 한 후보나 ONNX가 실패해도 다른 정상 후보와 ArcFace 유사도 결과는 함께 반환되며, 실패한 모델 점수는 임의로 생성하지 않는다.

오류 응답은 항상 같은 형태다.

```json
{
  "error": {
    "code": "NO_FACE",
    "message": "얼굴을 찾지 못했습니다. 정면에서 더 밝고 가깝게 다시 촬영하세요."
  }
}
```

## 테스트

`scan_id` 기반 비동기 이미지 후보 데모는 [비동기 노출 스캔 안내](async-exposure-scan.md)의 Swagger 순서를 따른다. 등록 임베딩은 기본 30분, 스캔 결과는 60분 동안 프로세스 메모리에만 보관되며 재시작 복구는 아직 지원하지 않는다.

API 경계 테스트는 실제 얼굴이나 모델 파일 없이 실행된다.

```bash
python -m pip install -r requirements-api-test.txt
python -m unittest discover -s tests
```
