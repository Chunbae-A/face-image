# 딥소각 클라이언트용 공개 노출 모니터링 API

이 문서는 딥소각 화면을 만드는 개발자가 모델 내부 구조를 몰라도 `공개 후보 검색 → 본인 얼굴 확인 → 딥페이크 신호 확인` 결과를 연결할 수 있도록 필요한 API만 설명한다.

중요한 연결 원칙은 다음과 같다.

- 앱이나 브라우저가 모델 API를 직접 호출하지 않는다.
- 딥소각 서버가 이 API를 호출하고, 화면에는 필요한 결과만 전달한다.
- 얼굴 유사도와 딥페이크 점수는 확률이 아니다. 화면에 임의의 퍼센트로 바꾸지 않는다.
- 결과는 자동 신고·삭제 명령이 아니라 사람이 확인할 후보 목록이다.

## 전체 호출 순서

1. `GET /v1/capabilities`로 기능·모델·연구 기준 상태를 확인한다.
2. `POST /v1/faceguard/enrollments`로 동의받은 본인 사진 1~5장을 임시 등록한다. 3장을 권장한다.
3. 딥소각 서버가 Google Vision으로 후보를 수집한 뒤 `POST /v1/exposure-scans`로 모델 작업을 시작하고 `scan_id`를 받는다.
4. `GET /v1/exposure-scans/{scan_id}`를 1~2초 간격으로 호출해 진행 상태를 확인한다.
5. 작업이 끝나면 `GET /v1/exposure-scans/{scan_id}/client-candidates`로 화면용 후보를 받는다.

## 0. 기능과 모델 상태 확인

`GET /v1/capabilities`는 클라이언트 서버가 어떤 버튼을 노출할지 결정하는
안정적인 계약이다. `load_state=unavailable`인 기능은 비활성화하고,
`decision_status`가 `research_only...`인 결과는 확률이나 확정 판정으로
표시하지 않는다. `automatic_enforcement_allowed`는 현재 항상 `false`다.

## 1. 얼굴 임시 등록

`multipart/form-data`의 `reference_images`에 사진을 넣는다.

```bash
curl -X POST http://127.0.0.1:8000/v1/faceguard/enrollments \
  -F 'reference_images=@reference-1.jpg' \
  -F 'reference_images=@reference-2.jpg' \
  -F 'reference_images=@reference-3.jpg'
```

응답의 `enrollment_id`를 다음 요청에 사용한다. 원본 사진은 처리 직후 버리고, 평균 얼굴 특징값과 품질 정보만 기본 30분 동안 메모리에 남는다.

## 2. Google Vision 후보 분석 시작

딥소각 서버가 Google Vision Web Detection으로 가져온 후보 URL을 모델 API에 전달한다. Google API 키와 검색 동의 처리는 딥소각 서버가 담당하며 모델 API에는 키를 전달하지 않는다.

```json
{
  "enrollment_id": "1단계에서 받은 ID",
  "privacy_mode": "privacy_strict",
  "maximum_results": 5,
  "candidates": [
    {
      "page_url": "https://example.com/discovered-page",
      "media_url": "https://example.com/discovered-image.jpg",
      "thumbnail_url": "https://example.com/thumb.jpg"
    }
  ]
}
```

성공하면 HTTP `202`와 함께 다음 주소를 받는다.

```json
{
  "scan_id": "스캔-ID",
  "status": "queued",
  "status_url": "/v1/exposure-scans/스캔-ID",
  "candidates_url": "/v1/exposure-scans/스캔-ID/candidates",
  "client_candidates_url": "/v1/exposure-scans/스캔-ID/client-candidates"
}
```

`client_candidates_url`은 딥소각 후보 화면용이고, `candidates_url`은 모델 개발자가 상세 원인을 확인할 때 사용한다.

## 3. 진행 상태 확인

`status_url`을 호출해 `completed`, `partial_failed`, `failed` 중 하나가 될 때까지 기다린다. `progress_percent`는 작업 진행률이며 모델 신뢰도가 아니다.

## 4. 화면용 후보 확인

```json
{
  "scan_id": "스캔-ID",
  "status": "completed",
  "candidate_count": 1,
  "identity_match_count": 1,
  "review_candidate_count": 1,
  "candidates": [
    {
      "candidate_id": "후보-ID",
      "source_url": "https://example.com/post",
      "media_url": "https://example.com/image.jpg",
      "thumbnail_url": "https://example.com/thumb.jpg",
      "source_type": "image",
      "source_engine": "google_vision_web_detection",
      "face_similarity": 0.65,
      "face_match_level": "matched",
      "deepfake_score": 0.81,
      "deepfake_signal": "suspected",
      "recommended_action": "review_required",
      "analysis_status": "completed",
      "warning": "연구용 결과 안내"
    }
  ]
}
```

화면에서는 다음 값만 우선 사용하면 된다.

| 필드 | 화면에서의 뜻 |
|---|---|
| `face_match_level` | `matched`: 본인 후보, `review`: 경계 구간, `not_matched`: 다른 사람 권장, `unavailable`: 분석 실패 |
| `deepfake_signal` | `suspected`: 조작 의심 신호, `not_suspected`: 현재 기준에서 미검출, `not_analyzed`: 얼굴 단계에서 제외, `unavailable`: 분석 실패 |
| `recommended_action` | 화면 버튼과 안내 문구를 결정하는 안전한 행동값 |
| `face_similarity` | ArcFace 원점수. 확률이 아님 |
| `deepfake_score` | 딥페이크 모델 원점수. 확률이 아님 |

`recommended_action` 권장 표시는 다음과 같다.

| 값 | 권장 표시 |
|---|---|
| `review_required` | 조작 의심 신호가 있어 원문을 직접 확인하세요 |
| `identity_review_required` | 본인 여부가 경계 구간이므로 먼저 얼굴을 확인하세요 |
| `monitor` | 본인 후보이나 현재 조작 의심 신호는 검출되지 않았습니다 |
| `exclude_recommended` | 다른 사람일 가능성이 높아 제외를 권장합니다 |
| `analysis_unavailable` | 분석하지 못했습니다. 다시 시도하거나 원문을 확인하세요 |

## 현재 한계

- Google Vision은 공개 후보를 찾는 단계다. 동일인 여부는 ArcFace, 딥페이크 신호는 ONNX 모델이 각각 후속 확인한다.
- Google Vision 실제 키와 사용자 검색 동의는 딥소각 서버에서만 관리한다.
- 현재 통합 스캔은 공개 이미지 후보를 처리한다. 공개 영상 자동 수집은 별도 개발이 필요하다.
- 등록과 결과는 메모리 저장 방식이므로 서버 재시작 시 사라진다.
- 모델 기준값은 연구용이다. 자동 신고·삭제 또는 피해 확정에 사용하지 않는다.
