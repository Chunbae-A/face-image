# 개발 워크플로

`face-image`는 얼굴 생체정보와 모델 실험을 다루므로 모든 변경을 Issue → branch → commit → PR → CI 순서로 관리한다.

## 1. Issue 먼저

코드를 수정하기 전에 Issue를 만든다. 실험 Issue에는 최소한 다음을 기록한다.

- 검증 가설과 성공/실패 기준
- 데이터셋, 이용 승인 상태, split 단위
- 모델·가중치·라이선스
- baseline과 변경할 독립변수
- 필수 지표, seed, 실행 환경
- 성능 누수, 개인정보, 재배포 위험

## 2. Branch 이름

형식은 `<type>/<issue-number>-<short-description>`으로 고정한다.

| Type | 용도 | 예시 |
|---|---|---|
| `feat` | 제품/파이프라인 기능 | `feat/21-mobile-embedding-api` |
| `fix` | 재현 가능한 버그 수정 | `fix/22-colab-cuda-provider` |
| `exp` | 모델·데이터·프로토콜 실험 | `exp/23-arcface-seed-audit` |
| `docs` | 문서만 변경 | `docs/24-aihub-protocol` |
| `chore` | CI·툴링·저장소 운영 | `chore/25-repository-workflow` |

작업자나 도구 이름을 branch에 사용하지 않는다. 하나의 branch는 하나의 Issue만 해결한다.

## 3. Commit

Commit 제목은 변경 유형과 목적을 보여준다.

```text
feat: add video embedding pipeline
fix: pin Colab-compatible ONNX Runtime
exp: add multi-seed threshold audit
docs: document AI-Hub approval gate
chore: add repository hygiene CI
```

## 4. Pull Request

- 제목은 변경의 결과를 설명한다.
- 본문에 `Closes #<issue>`를 넣어 Issue를 연결한다.
- 검증 전에는 Draft로 열고, acceptance criteria와 CI가 통과하면 Ready for review로 변경한다.
- 실험 PR은 config, seed, hardware, 데이터/모델 hash, 성공·실패 건수, 주요 지표와 한계를 기록한다.
- 민감 데이터나 비밀값이 포함되지 않았는지 직접 확인한다.

## 5. 필수 검증

```bash
python3 -m unittest discover -s tests
python3 scripts/check_repository_hygiene.py
python3 scripts/build_celebdf_colab_notebook.py
git diff --exit-code -- notebooks/celebdf_arcface_full_colab.ipynb
```

## 6. Git에 넣지 않는 것

- 실제 얼굴 영상·이미지·정렬 crop
- 얼굴 임베딩과 개인별 score 원본
- 데이터셋 archive, 모델 가중치, checkpoint
- API key, token, 인증 파일, 개인 식별자
- 실행 output이 남은 notebook

외부에 공유할 수 있는 것은 집계 metric, 비식별 reject reason, 실행 환경과 hash, 그래프로 제한한다.

## 7. 문서 위치

새 문서를 루트에 추가하지 않는다. 전체 문서 지도는 [`docs/README.md`](docs/README.md)에서 관리한다.

| 위치 | 내용 |
|---|---|
| `docs/api/` | API 실행, 엔드포인트, 검색 어댑터 사용법 |
| `docs/demo/` | 발표와 데모 화면·API 흐름 |
| `docs/experiments/` | 모델 실험 계획과 Colab·Kaggle 재현 절차 |
| `reports/<실험>/<날짜>/` | 개인정보를 제외한 실제 실행 결과 |

문서를 이동하거나 링크를 추가하면 `tests/test_documentation_links.py`를 실행해 상대 링크가 유효한지 확인한다.
