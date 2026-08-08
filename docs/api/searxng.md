# SearXNG 무료 후보 검색 실행 가이드

딥소각 API가 검색어를 받아 공개 이미지·영상 후보 URL을 모으도록 로컬 SearXNG를 함께 실행하는 방법이다. 무료 데모용이며 별도 검색 API 키가 필요 없다.

## 무엇을 하고, 무엇을 하지 않나요?

```text
동의받은 검색어
      ↓
로컬 SearXNG가 공개 검색엔진 조회
      ↓
딥소각 API가 공개 URL만 허용하고 중복 제거
      ↓
안전한 후보 이미지만 다운로드
      ↓
로컬 ArcFace가 등록 얼굴과 유사도 비교
```

- 하는 일: 검색어 기반 공개 이미지·영상 후보 URL 수집
- 보내는 정보: `query_text`, 검색 종류, 언어, Safe Search 값
- 보내지 않는 정보: 등록 얼굴 사진, 얼굴 임베딩, 사용자 계정 ID
- 추가 통합 경로: 후보 이미지를 등록 얼굴과 비교해 동일인 가능성 수치 반환
- 하지 않는 일: 얼굴 역이미지 검색, 영상 얼굴 트랙. 딥페이크 판정은 SearXNG가 아니라 후속 로컬 ONNX 단계가 수행한다.

## 1. 실행

Docker Desktop을 켠 뒤 저장소 루트에서 실행한다.

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.searxng.yml up --build --detach
```

공유 환경이라면 `.env`의 `SEARXNG_SECRET`을 임의의 긴 값으로 바꾼다. SearXNG 포트는 Docker 내부에만 열리고, 사용자는 딥소각 API `127.0.0.1:8000`만 호출한다.

## 2. 상태 확인

```bash
curl http://127.0.0.1:8000/health
```

아래 두 값이 보이면 검색 연결이 준비된 것이다.

```json
{
  "search_providers": ["user_url", "searxng"],
  "web_search_enabled": true
}
```

## 3. Swagger에서 검색

1. [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)를 연다.
2. `POST /v1/search/candidates`를 선택한다.
3. `Try it out`을 누르고 아래 JSON을 입력한다.

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

4. `Execute`를 누른다.
5. `candidate_count`가 1 이상이고 후보의 `provider`가 `searxng`이면 정상이다.

검색엔진 상황에 따라 결과가 0개이거나 일부 엔진이 실패할 수 있다. 이것은 얼굴 모델 오류가 아니라 공개 검색엔진의 응답 상태다.

## 4. 검색과 얼굴 선별을 한 번에 실행

Swagger에서 `POST /v1/pipeline/search-and-filter`를 선택한다. 등록 얼굴 3장, 동의받은 검색어, `web_monitoring_consent=true`, `maximum_results=3`을 입력한다.

등록 얼굴은 로컬 ArcFace에만 사용되고 외부 검색엔진에는 전달되지 않는다. 결과의 `retrieval_match`는 넓은 후보 통과, `identity_match`는 더 엄격한 연구 기준 통과를 뜻한다. `retrieval_match=true`인 단일 얼굴 이미지만 로컬 ONNX가 추가 분석하며 `deepfake_score`도 확정값은 아니다.

## 5. 종료

```bash
docker compose -f docker-compose.yml -f docker-compose.searxng.yml down
```

얼굴 모델 캐시 볼륨은 삭제하지 않으므로 다음 실행에서 재사용된다.

## 데모에서 설명할 한 문장

> 무료 SearXNG가 공개 이미지 후보를 모으면 로컬 ArcFace가 본인 후보를 선별하고, 넓은 기준을 통과한 이미지에만 비공개 EfficientNet-B4 ONNX가 딥페이크 점수를 계산합니다. 얼굴 역검색과 영상 분석은 후속 단계입니다.

## 라이선스와 참고

SearXNG는 딥소각 API와 분리된 컨테이너로 실행한다. 배포·수정 시 SearXNG의 AGPL-3.0 조건을 별도로 확인해야 한다.

- [SearXNG Search API](https://docs.searxng.org/dev/search_api.html)
- [SearXNG Docker 설치](https://docs.searxng.org/admin/installation-docker)
- [SearXNG GitHub](https://github.com/searxng/searxng)
