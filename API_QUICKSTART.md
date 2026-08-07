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

## 7. 종료

```bash
docker compose down
```

## 주의사항

- 현재 API와 `buffalo_l` 가중치는 연구·해커톤 데모용이다.
- `threshold_status`가 `research_only_unapproved`이므로 계정 복구·삭제 같은 작업을 자동 승인하지 않는다.
- 동의받지 않은 얼굴 사진을 사용하거나 저장소에 업로드하지 않는다.
- 로그인·TLS·요청 제한이 없으므로 API를 인터넷에 직접 공개하지 않는다.

상세 설정과 오류 코드는 [`API_RUNBOOK.md`](API_RUNBOOK.md)를 참고한다.
