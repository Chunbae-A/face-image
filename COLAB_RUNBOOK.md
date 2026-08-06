# 얼굴가드 Google Colab 실행 가이드

Colab은 딥소각 얼굴가드의 Debug/Pilot 연산 환경으로 사용한다. 얼굴 원본과 임베딩은 생체정보이므로, 호스팅된 Colab·Google Drive에 올릴 수 있는지 신청 약관과 보안 담당자가 먼저 승인해야 한다.

## 실행 모드

| 모드 | 연산 위치 | 실제 얼굴 데이터 | 사용 조건 |
|---|---|---|---|
| A. Hosted Colab GPU | Google 관리 VM | 기본 금지 | 데이터 제공약관의 클라우드·국외 반출 허용과 프로젝트 승인이 모두 있을 때만 |
| B. Colab local runtime | 사용자 Mac | 허용 | Colab UI만 쓰고 코드/데이터는 로컬에서 실행. 현재 MPS 사용 |
| C. Hosted Colab + 비식별 통계 | Google 관리 VM | 금지 | 합성/공개 허용 Debug data, 얼굴이 없는 가짜 manifest, 집계 metric만 사용 |

승인 전 기본은 **B 또는 C**다. A를 사용하려면 `configs/faceguard/experiment.yaml`의 두 cloud 플래그를 승인 기록과 함께 변경한다.

## Colab 제약을 반영한 설계

[Google Colab FAQ](https://research.google.com/colaboratory/faq.html)에 따르면 GPU 종류·사용량 한도·idle timeout·VM 최대 수명은 변동하며 VM은 삭제될 수 있다. 따라서 GPU 명칭과 VRAM을 매 run 실측하고, 영구 디스크처럼 사용하지 않는다.

- 100GB 전체를 runtime에 항상 복사하지 않고 2~10GB shard 단위로 처리한다.
- 하나의 shard를 `/content`로 복사·해제하고, 처리한 뒤 결과/checkpoint/로그를 승인된 영구 저장소에 atomic write한다.
- Drive에서 수많은 작은 파일을 직접 읽지 않고 archive를 runtime에 복사한 뒤 해제한다.
- shard별 `_SUCCESS`, source/config/model hash, 성공/실패 건수를 저장하여 세션 중단 후 이어받는다.
- 노트북 output에 얼굴, 실제 파일경로, subject ID, Drive 링크를 출력하지 않는다. 공유 전 모든 cell output을 삭제한다.
- secrets/API key는 notebook cell, Git, Drive 평문에 저장하지 않고 Colab Secrets를 사용한다.

## 시작

1. GitHub에 작업 브랜치가 있는 경우 [`notebooks/faceguard_colab.ipynb`](notebooks/faceguard_colab.ipynb)를 Colab으로 연다.
2. Runtime → Change runtime type에서 GPU를 선택한다. 할당 GPU는 고정값으로 가정하지 않는다.
3. 보안 게이트 cell에서 hosted cloud 승인 여부를 선택한다. 승인이 없으면 Drive mount/data path cell이 중단된다.
4. 환경 inventory와 저장 여유를 저장한다. 실제 GPU/VRAM/batch size를 run manifest에 기록한다.
5. Debug → Pilot 순서로 실행하고, Pilot 결과 없이 Full 100GB를 Colab으로 복사하지 않는다.

## Hosted Colab이 승인되지 않을 때

Colab의 `Connect to local runtime`으로 사용자 Mac에 연결한다. 이 모드에서 Colab 프론트엔드는 로컬 코드를 실행하므로 노트북의 업로드/마운트 cell을 사용하지 않는다. Google의 [local runtime 안내](https://research.google.com/colaboratory/local-runtimes.html)가 경고하듯, 연결된 notebook은 로컬 파일을 읽고 수정/삭제할 수 있으므로 이 저장소의 노트북만 신뢰하여 연다.

## Celeb-DF-v2 전체 얼굴인식 baseline

AI-Hub 승인과 별개로, 공식 신청·승인으로 받은 Celeb-DF-v2 파일은 [`notebooks/celebdf_arcface_full_colab.ipynb`](notebooks/celebdf_arcface_full_colab.ipynb)에서 처리한다. 단, 데이터 이용약관상 Google Drive·Hosted Colab 처리가 허용되는지 사용자가 확인한 뒤 노트북의 권한 확인 값을 변경해야 한다.

이 노트북의 smoke 실행은 최종 데이터 축소가 아니다. 2개 영상으로 설치·모델 다운로드·얼굴 탐지를 확인한 뒤 동일 checkpoint에서 Celeb-real 590개 영상 전체 실행을 자동으로 이어간다.

1. Colab의 `파일 → 노트북 업로드`에서 노트북 파일을 연다. 기본 `CODE_SOURCE=embedded` 모드는 GitHub 접근 없이 필요한 실행 코드를 노트북에서 복원한다.
2. `Celeb-DF-v2.zip`을 약관상 허용된 경로에 둔다. 기본 Drive 경로는 `/content/drive/MyDrive/Celeb-DF-v2.zip`이다.
3. GPU runtime을 선택하고 권한·InsightFace 비상업 연구 모델 확인 값을 `True`로 바꾼다.
4. ZIP inventory가 590개 영상·59명·평가 가능 56명인지 확인한다.
5. 전체 590개 영상을 추출하고 영상당 10프레임에서 얼굴을 추론한다.
6. 영상별 임베딩을 평균한 뒤 등록 3개/5개 영상과 겹치지 않는 query 영상으로 평가한다.
7. validation/test 인물은 완전히 분리하며, validation에서 정한 FAR threshold를 test에 한 번 적용한다.
8. 얼굴이 없는 결과 묶음만 내려받는다. 얼굴 프레임과 `.npz` 임베딩은 Git 또는 공개 저장소에 올리지 않는다.

결과 묶음에는 inventory, 실행 환경·모델 hash, reject 사유, ROC-AUC/EER/TAR/FAR/FRR 표, ROC/score 분포 그림이 포함된다. Celeb-real 결과는 일반 얼굴 동일인 검증 baseline이며 딥페이크 탐지 정확도나 한국인 특화 성능으로 해석하지 않는다.

## Celeb-DF baseline 재현성 감사

[`notebooks/celebdf_arcface_audit_colab.ipynb`](notebooks/celebdf_arcface_audit_colab.ipynb)는 영상당 1/5/10프레임을 각각 다시 추론한 뒤 등록 1/3/5개, subject split seed 5개를 비교한다. 등록 수와 관계없이 query는 항상 6번째 영상부터 사용해 공정하게 비교한다.

- validation/test identity intersection을 seed별로 검증한다.
- registration/query video intersection을 frame run·seed별로 검증한다.
- NPZ 임베딩은 `/content`에만 두고 Drive/Git에 저장하지 않는다.
- Drive에는 집계 JSON/CSV, hash, 비식별 reject reason count, PNG만 포함한 ZIP을 저장한다.

DriveFS가 `mount failed`로 반복 종료되면 Drive 웹에서 승인 ZIP의 파일 ID를 확인해 감사 노트북의 `DRIVE_SOURCE_FILE_ID`에만 입력한다. 이 fallback은 Google 인증 Drive API로 ZIP을 `/content`에 내려받으며, ID는 Git·결과 ZIP에 기록하지 않는다. 이때 비식별 결과 ZIP은 브라우저로 내려받는다.

감사에는 `Celeb-real`만 필요하다. 이용약관이 허용하면 원본에서 `Celeb-real` 590개 영상만 담은 ZIP을 만들고 Drive의 전용 비공개 폴더에 한 번 업로드하는 방식을 우선한다. 새 Colab runtime에서는 이 파일을 `/content`로 복사하거나, Drive 인증까지 실패하면 1GB 안팎의 단일 ZIP만 세션 저장소에 올린다. `EXPECTED_SOURCE_ZIP_BYTES`에는 로컬에서 확인한 정확한 바이트를 입력한다.

노트북은 `/content/Celeb-DF-v2.zip`이 남아 있어도 예상 바이트와 다르면 불완전한 이전 업로드로 간주해 무시한다. 파일 패널의 표시 단위가 반올림되므로 성공 여부는 UI의 MB/GB가 아니라 정확한 바이트와 ZIP inventory `590 videos / 59 subjects / 56 eligible subjects`로 판단한다.

단일 ZIP 업로드도 불가능할 때만 로컬 ZIP을 2GB 미만의 `Celeb-DF-v2.zip.part-*` 조각으로 나누어 세션 저장소에 모두 올린 뒤 `ASSEMBLE_RUNTIME_UPLOAD_PARTS=True`로 4번 셀을 실행한다. 노트북은 조각 바이트의 합과 결합 ZIP 크기를 비교해 `/content/Celeb-DF-v2.zip`을 생성한다. 이 경로의 원본·조각은 runtime 종료 시 삭제된다.
