# 얼굴가드 API 빠른 실행

GitHub 저장소만 있으면 실행할 수 있다. **Celeb-DF 데이터셋, Colab, Kaggle 결과 파일은 필요하지 않다.**

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

응답에 `"status":"ok"`와 `"model_loaded":true`가 나오면 정상이다.

## 5. 얼굴 비교 시험

브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)를 연다.

1. `POST /v1/faceguard/verify` 선택
2. `Try it out` 선택
3. `reference_images`에 등록 얼굴 1~5장 선택 — 3장 권장
4. `query_image`에 확인 얼굴 1장 선택
5. `Execute` 선택

`is_same_person`은 동일인 **후보 판정**이며 운영 본인인증 확정값이 아니다.

## 6. 무료 공개 URL 후보 시험

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

## 7. 무료 키워드 검색 시험

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

## 8. 검색부터 얼굴 선별까지 한 번에 시험

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
- `error_code`: 다운로드·얼굴 검출에 실패한 이유

현재 `retrieval_threshold=0.20`은 기능 연결용 임시값이며 정확도 검증을 마친 운영값이 아니다. 여러 얼굴이 있는 이미지와 영상 후보는 아직 이 경로에서 처리하지 않는다.

## 9. 종료

```bash
docker compose -f docker-compose.yml -f docker-compose.searxng.yml down
```

## 주의사항

- 현재 API와 `buffalo_l` 가중치는 연구·해커톤 데모용이다.
- `threshold_status`가 `research_only_unapproved`이므로 계정 복구·삭제 같은 작업을 자동 승인하지 않는다.
- 동의받지 않은 얼굴 사진을 사용하거나 저장소에 업로드하지 않는다.
- 로그인·TLS·요청 제한이 없으므로 API를 인터넷에 직접 공개하지 않는다.

상세 설정과 오류 코드는 [`API_RUNBOOK.md`](API_RUNBOOK.md)를 참고한다.
