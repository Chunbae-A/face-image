# 비동기 노출 스캔 데모: 처음부터 실행하기

이 기능은 시간이 걸리는 `검색 → 얼굴 비교 → 딥페이크 분석`을 화면이 멈추지 않도록 백그라운드에서 실행한다. 요청을 보내면 서버는 `scan_id`를 먼저 주고, 화면은 이 ID로 진행 상태를 물어보면 된다.

## 준비

1. [README의 Docker 실행 순서](README.md#docker-권장-실행)로 API를 실행한다.
2. 브라우저에서 [Swagger](http://127.0.0.1:8000/docs)를 연다.
3. `비동기 노출 스캔` 그룹을 찾는다.

현재 데모는 **공개 JPEG·PNG·WEBP 이미지 후보**를 처리한다. `media_url`은 웹페이지 주소가 아니라 이미지 파일을 직접 받을 수 있는 공개 주소여야 한다.

## 1. 본인 얼굴 임시 등록

Swagger에서 `POST /v1/faceguard/enrollments`를 열고 `Try it out`을 누른다. `reference_images`에 동의받은 본인 사진 3장을 추가한 뒤 `Execute`를 누른다.

응답에서 다음 값을 복사한다.

```json
{
  "enrollment_id": "여기에-발급된-ID",
  "status": "active",
  "reference_count": 3,
  "storage": "memory_only"
}
```

원본 사진은 응답이 나오면 버린다. 세 사진의 ArcFace 임베딩을 하나로 평균한 값과 품질 정보만 기본 30분 동안 메모리에 남는다.

## 2. 스캔 시작

`POST /v1/exposure-scans`를 열고 `Try it out`을 누른다. 먼저 자동 웹 검색 없이 안전하게 테스트하려면 아래 JSON을 넣는다.

```json
{
  "enrollment_id": "1단계에서 복사한 ID",
  "privacy_mode": "privacy_strict",
  "web_monitoring_consent": false,
  "maximum_results": 5,
  "candidates": [
    {
      "page_url": "https://example.com/public-post",
      "media_url": "https://example.com/public-face.jpg"
    }
  ]
}
```

`Idempotency-Key`는 선택 항목이다. 입력한다면 `demo-20260808-0001`처럼 8자 이상의 고유한 값을 쓴다. 네트워크 오류로 같은 요청을 다시 보낼 때 같은 키를 쓰면 중복 스캔 대신 기존 `scan_id`를 받는다.

성공하면 서버는 분석이 끝나기를 기다리지 않고 HTTP `202`와 함께 다음 형태를 반환한다.

```json
{
  "scan_id": "발급된-스캔-ID",
  "status": "queued",
  "status_url": "/v1/exposure-scans/발급된-스캔-ID",
  "candidates_url": "/v1/exposure-scans/발급된-스캔-ID/candidates"
}
```

## 3. 진행 상태 확인

`GET /v1/exposure-scans/{scan_id}`에 2단계의 `scan_id`를 넣고 조회한다. 화면은 이 API를 1~2초에 한 번씩 호출하면 된다.

| 상태 | 쉬운 뜻 |
|---|---|
| `queued` | 작업 시작 대기 |
| `searching` | 공개 후보 URL 수집·중복 제거 |
| `identity_filtering` | ArcFace로 등록 얼굴과 후보 비교 |
| `deepfake_analyzing` | 넓은 얼굴 기준을 통과한 후보를 ONNX로 분석 |
| `completed` | 전체 정상 완료 |
| `partial_failed` | 일부 URL은 실패했지만 성공 결과는 보존 |
| `failed` | 작업 전체 실패, `error_code` 확인 필요 |

## 4. 후보 결과 확인

상태가 `completed` 또는 `partial_failed`가 되면 `GET /v1/exposure-scans/{scan_id}/candidates`를 호출한다.

- `similarity_raw`: 등록 얼굴과 후보 얼굴의 ArcFace 원점수
- `identity_match`: 현재 연구 기준을 넘은 같은 사람 후보인지
- `deepfake.deepfake_score`: EfficientNet-B4 ONNX 원점수
- `deepfake.is_suspected_deepfake`: 현재 연구 기준을 넘었는지
- `error_code`: 해당 후보만 실패한 이유

`similarity_raw`와 `deepfake_score`는 확률이 아니다. `calibrated_probability` 값이 `null`이면 화면에 `89% 확률`처럼 표시하지 말고 `원점수·사람 검토 필요`로 보여준다.

## 현재 제한

- 이미지 후보만 비동기 통합 처리한다. 공개 영상 URL 자동 다운로드는 다음 개발 범위다.
- 메모리 저장소이므로 API 서버를 재시작하면 등록·작업·결과가 사라진다.
- Docker는 현재 워커 1개로 실행한다. 여러 서버로 확장하려면 Redis 같은 공유 큐·저장소가 필요하다.
- 얼굴·딥페이크 기준값은 모두 연구용이다. 결과만으로 피해를 확정하거나 자동 신고·삭제하지 않는다.
