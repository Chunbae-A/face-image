# 딥소각 얼굴가드 API 사용 가이드

이 API는 **등록 얼굴 사진 1~5장과 확인 사진 1장을 받아 같은 사람 후보인지 비교**한다. 실제 사용에서는 등록 사진 3장을 권장한다.

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
python -m uvicorn faceguard_api.app:app --host 127.0.0.1 --port 8000
```

첫 추론에서는 모델 파일을 내려받고 준비하므로 이후 요청보다 오래 걸릴 수 있다. 서버가 켜지면 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)에서 요청을 직접 시험할 수 있다.

## Docker 실행

```bash
cp .env.example .env
```

`.env`에서 이용 조건 확인값을 `true`로 바꾼 뒤 실행한다.

```bash
docker compose up --build
```

기본 Docker 이미지는 CPU용이다. Linux CUDA 서버에서는 `requirements-api-gpu.txt`의 `onnxruntime-gpu`를 사용하고 NVIDIA Container Runtime을 별도로 설정한다.

## 1. 서버 상태 확인

```bash
curl http://127.0.0.1:8000/health
```

예시:

```json
{
  "status": "ok",
  "api_version": "0.1.0",
  "model_name": "buffalo_l",
  "model_loaded": false,
  "execution_provider": null,
  "model_fingerprint": null,
  "license_accepted": true,
  "threshold_status": "research_only_unapproved"
}
```

`model_loaded`가 `false`인 것은 아직 첫 추론을 하지 않아 모델을 지연 로딩하는 상태다.

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
  "threshold": 0.2823836207389832,
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

## 3. 무료 공개 URL 후보 정규화

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

`privacy_strict`에서는 검색용 얼굴 이미지를 외부 제공자에 보내지 않는다. `web_monitoring`은 사용자의 명시적 동의와 별도 외부 검색 제공자 설정이 모두 있을 때만 사용할 수 있다. 현재 기본 설정에는 외부 검색 제공자가 없으므로 `web_monitoring` 요청은 `SEARCH_PROVIDER_UNAVAILABLE`로 거절한다.

즉, 이번 무료 기능은 **제보 URL을 안전한 후보 목록으로 준비하는 단계**다. 새로운 URL을 역이미지 검색으로 자동 발견하는 기능과 실제 URL 다운로드·얼굴 선별은 Issue #13·#14의 후속 구현 범위다.

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

이 값은 파이프라인이 끝까지 동작하는지 확인한 **1건의 스모크 테스트**다. 정확도, 오인식률, 한국인 일반화 또는 운영 안전성을 증명하는 결과가 아니다.

추가로 Docker HTTP API에 승인받은 Celeb-real 전체 프레임을 직접 전송한 로컬 데모 스모크에서는 같은 사람 `0.694200`, 다른 사람 `-0.051950`이 관측됐다. 예열 후 CPU 처리시간은 두 요청에서 `962.706ms`, `1,188.254ms`였다. 개인정보를 제외한 실행 환경과 한계는 [`reports/faceguard_demo_smoke/2026-08-06`](reports/faceguard_demo_smoke/2026-08-06)에 기록했다.

## 주요 오류

| 오류 코드 | 뜻 | 사용자 안내 |
|---|---|---|
| `MODEL_LICENSE_NOT_ACCEPTED` | 모델 가중치 이용 조건 미확인 | 서버 설정 확인 |
| `UNSUPPORTED_CONTENT_TYPE` | 지원하지 않는 파일 형식 | JPEG·PNG·WEBP로 변환 |
| `IMAGE_TOO_LARGE` | 한 장이 8MB 초과 | 이미지 크기 축소 |
| `TOO_MANY_PIXELS` | 해상도가 2천만 픽셀 초과 | 해상도 축소 |
| `NO_FACE` | 얼굴을 찾지 못함 | 정면에서 밝게 재촬영 |
| `MULTIPLE_FACES` | 얼굴이 둘 이상 있음 | 혼자 나온 사진 사용 |
| `FACE_TOO_SMALL` | 얼굴 면적이 너무 작음 | 카메라에 더 가까이 촬영 |
| `LOW_DETECTION_SCORE` | 얼굴이 불분명함 | 흔들림·가림을 없애고 재촬영 |
| `MODEL_UNAVAILABLE` | 모델 또는 실행 환경 문제 | 서버 로그와 모델 설치 확인 |
| `PRIVATE_NETWORK_URL_BLOCKED` | 내부망·로컬 URL 입력 | 접근 가능한 공개 URL 사용 |
| `URL_SECRET_PARAMETER_NOT_ALLOWED` | URL에 토큰·키로 보이는 쿼리 포함 | 비밀 쿼리를 제거한 공개 URL 사용 |
| `WEB_MONITORING_CONSENT_REQUIRED` | 외부 이미지 검색 동의 없음 | 개인정보 엄격 모드 사용 또는 별도 동의 |
| `SEARCH_PROVIDER_UNAVAILABLE` | 외부 검색 제공자 미설정 | 무료 URL 제보 모드 사용 |

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

API 경계 테스트는 실제 얼굴이나 모델 파일 없이 실행된다.

```bash
python -m pip install -r requirements-api-test.txt
python -m unittest discover -s tests
```
