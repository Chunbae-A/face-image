# K-FACE 400명 전체 얼굴가드 반복 검증

## 목적

Mac에서 생성한 K-FACE 저·중화질 전체 ArcFace 특징값으로 딥소각 얼굴가드의
등록 사진 수와 한국인 얼굴 판정 기준값을 다시 검증한다. 새로운 모델을
학습하는 작업이 아니라, 현재 모델을 한국인 데이터에서 평가하고 기준값을
보정하는 모델링 작업이다.

## 입력 현황

| 항목 | 값 |
|---|---:|
| 인물 | 400명 |
| 저·중화질 전체 이미지 | 8,640,000장 |
| 특징 추출 성공 이미지 | 7,749,710장 |
| 제외 이미지 | 890,290장 |
| 비공개 임베딩 chunk | 8,800개 |
| 특징값 크기 | 16,084,514,540 bytes |

- 원본 얼굴 이미지와 실제 인물 식별자는 Kaggle에 올리지 않는다.
- 익명화 임베딩 Dataset과 Notebook을 모두 Private로 유지한다.
- 개별 임베딩과 개별 비교 점수는 Kaggle Output에 저장하지 않는다.

## 검증 조건

1. 등록은 중화질 특징 3장·5장·9장을 각각 평균한다.
2. 인물이 섞이지 않도록 seed마다 validation 200명과 test 200명으로 나눈다.
3. seed `20260815`부터 `20260819`까지 5번 반복한다.
4. 얼굴 검출점수 `0.60` 이상만 자동 판정에 사용한다.
5. validation에서 목표보다 엄격한 FAR `0.09%`로 기준값을 고른다.
6. test에서 TAR `90% 이상`, FAR `0.1% 이하`인지 확인한다.

타인 비교 조합은 수십억 건이므로 개별 점수를 파일로 만들지 않는다. Kaggle
GPU에서 인물별로 계산하고 40,000칸 histogram에 누적한다. 따라서 전체 비교를
반영하면서도 메모리 사용과 민감한 결과 파일 생성을 제한한다. ROC-AUC와 EER은
histogram 근삿값이며 TAR·FAR도 같은 고해상도 구간 경계에서 계산한다.

## 실행 파일

- 업로드 준비: `scripts/prepare_kface_kaggle_dataset.py`
- 전체 평가: `scripts/evaluate_kface_full_embeddings.py`
- 노트북 생성: `scripts/build_kface_full_kaggle_notebook.py`
- Kaggle Notebook: `notebooks/kaggle/kface_full_verification/notebook.ipynb`

## 현재 상태

- [x] 400명 전체 특징 추출 완료
- [x] 전체 처리본 무결성 확인
- [x] 추가 디스크 복사 없는 Kaggle 업로드 폴더 생성
- [x] 반복 평가 코드와 합성 데이터 테스트 완료
- [x] Private GPU Notebook 생성
- [ ] Kaggle Private Dataset 업로드
- [ ] Kaggle GPU 실행
- [ ] 결과 회수 및 보고서 작성
- [ ] API 기준값 적용 여부 결정

Kaggle 결과가 Gate를 통과해도 즉시 운영 기준값으로 교체하지 않는다. 실제 공개
웹 검색 이미지와 동의받은 모바일 촬영 이미지의 외부 검증을 통과한 뒤 API 변경
여부를 결정한다.
