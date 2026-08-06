# 얼굴가드 Kaggle 무료 GPU 실행 가이드

이 문서는 Issue #6의 촬영 열화 실험을 Google Colab 대신 Kaggle 무료 GPU에서 실행하는 방법을 설명한다. 실험량은 `590개 영상 × 5프레임 × 6조건 = 최대 17,700프레임`이다.

## 먼저 확인할 것

- Celeb-DF 제공 조건에서 **Kaggle 비공개 Dataset 처리**가 허용되는지 확인한다.
- Dataset과 Notebook은 반드시 `Private`로 유지한다.
- InsightFace `buffalo_l` 제공 가중치는 비상업 연구 목적으로만 사용한다.
- 원본 영상, 얼굴 crop, 사람별 점수, 임베딩을 GitHub나 Kaggle Output으로 공유하지 않는다.

기존에 확인한 Google Drive·Hosted Colab 허용 여부가 Kaggle까지 자동으로 포함된다고 가정하지 않는다. 허용 여부가 불명확하면 데이터 업로드 단계에서 멈춘다.

## 1. 비공개 데이터셋을 한 번만 만들기

1. Kaggle에 로그인한다.
2. `Datasets` → `New Dataset`을 누른다.
3. 로컬의 929MB `Celeb-DF-v2.zip`을 업로드한다.
4. 공개 범위를 `Private`로 둔다.
5. 업로드 후 파일 크기가 정확히 `928,989,923 bytes`인지 확인한다.

비공개 Dataset으로 등록하면 Notebook 세션이 종료돼도 원본을 다시 업로드할 필요가 없다.

## 2. 노트북 준비

1. `Code` → `New Notebook`을 연다.
2. [`notebooks/celebdf_arcface_robustness_kaggle.ipynb`](notebooks/celebdf_arcface_robustness_kaggle.ipynb)를 가져온다.
3. 오른쪽 `Input` → `Add Input`에서 앞서 만든 비공개 Dataset을 연결한다.
4. 오른쪽 `Settings`에서 `Accelerator`를 `GPU P100`으로 선택한다.
5. `Internet`을 켠다. InsightFace 모델을 처음 내려받을 때 필요하다.

## 3. 실행

1. 1번 셀에서 다음 두 값만 `True`로 바꾼다.

   ```python
   I_CONFIRM_CELEBDF_KAGGLE_PRIVATE_PROCESSING_IS_ALLOWED = True
   I_ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH_LICENSE = True
   ```

2. `Run All`로 위에서부터 실행한다.
3. 4번 셀에서 ZIP 하나와 정확한 바이트가 확인되는지 본다.
4. 6번 셀에서 `CUDAExecutionProvider`와 GPU 이름이 표시되는지 본다.
5. 7번 셀이 여섯 조건을 순서대로 처리할 때 브라우저를 닫지 않는다.

처음 `clean` 조건에서는 2개 영상으로 모델 설치와 얼굴 탐지를 먼저 확인한 뒤 590개 전체 실행으로 이어진다. 조건별로 25개 영상마다 같은 세션의 `/kaggle/temp`에 체크포인트를 저장한다.

## 4. 결과 받기

마지막 셀이 끝나면 다음 두 항목을 확인한다.

```text
private_work_root_deleted: True
raw_or_embedding_files_in_bundle: False
```

그다음 `/kaggle/working/celebdf_robustness_results.zip`만 내려받는다. 이 ZIP에는 집계 JSON·CSV·PNG와 실행 설정만 들어 있다.

## 자주 발생하는 문제

### ZIP을 찾지 못함

오른쪽 `Input`에 비공개 Dataset이 연결돼 있는지 확인한다. 파일명은 `Celeb-DF-v2.zip`, 크기는 `928,989,923 bytes`여야 한다.

### `CUDAExecutionProvider`가 없음

`Settings`에서 GPU를 선택한 다음 세션을 재시작하고 처음부터 실행한다. CPU로 17,700프레임 전체 실험을 진행하지 않는다.

### 모델 다운로드 실패

`Internet`이 켜져 있는지 확인한다. Dataset은 비공개로 유지하되 모델 다운로드를 위한 외부 통신만 허용한다.

### 세션이 종료됨

원본 비공개 Dataset은 남아 있으므로 다시 업로드할 필요는 없다. 다만 생체 임베딩을 영구 저장하지 않는 보안 원칙 때문에 해당 세션의 추론 체크포인트는 다시 생성해야 한다.

Kaggle 무료 GPU의 종류와 주간 사용 한도는 자원 상황에 따라 달라질 수 있다. 최신 내용은 [Kaggle 공식 GPU 사용 안내](https://www.kaggle.com/docs/efficient-gpu-usage)를 확인한다.
