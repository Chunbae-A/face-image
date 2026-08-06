# FaceGuard 스크립트

## 실험 사전 검사

```bash
# CPU/RAM/GPU/디스크 inventory
python3 scripts/faceguard_plan.py env --path .

# 다운로드 소요 시간과 저장공간 추정
python3 scripts/faceguard_plan.py download --size-gb 100 --efficiency 0.8
python3 scripts/faceguard_plan.py storage \
  --compressed-gb 100 --unpacked-gb 150 --preprocessed-gb 30 \
  --images 1000000

# subject/source/hash 누수와 Test 증강 금지 검사
python3 scripts/validate_faceguard_manifest.py examples/faceguard_manifest.csv
```

## Celeb-DF-v2 Celeb-real baseline

```bash
# ZIP을 풀지 않고 590개 영상 manifest 생성
python3 scripts/celebdf_faceguard.py inventory /path/to/Celeb-DF-v2.zip \
  --manifest outputs/celebdf_faceguard/celeb_real_manifest.csv \
  --summary outputs/celebdf_faceguard/celeb_real_inventory.json

# Celeb-real 590개만 안전하게 추출
python3 scripts/celebdf_faceguard.py extract /path/to/Celeb-DF-v2.zip \
  --manifest outputs/celebdf_faceguard/celeb_real_manifest.csv \
  --output outputs/celebdf_faceguard/videos --mode full

# 영상별 ArcFace 임베딩 평가
python3 scripts/celebdf_faceguard.py evaluate \
  --embeddings outputs/celebdf_faceguard/results/celeb_real_video_embeddings.npz \
  --output outputs/celebdf_faceguard/results/celeb_real_arcface_metrics.json
```

GPU 추론은 `notebooks/celebdf_arcface_full_colab.ipynb` 또는 `scripts/run_celebdf_arcface.py`를 사용한다. smoke 2개 영상은 환경 확인용이며, 동일 checkpoint에서 590개 전체 실행을 이어간다.

## 노트북 재생성

```bash
python3 scripts/build_celebdf_colab_notebook.py
python3 scripts/build_celebdf_audit_colab_notebook.py
```

노트북은 실행 스크립트를 내장하므로 생성 후 `scripts/celebdf_faceguard.py`, `scripts/run_celebdf_arcface.py`, `scripts/audit_celebdf_baseline.py`의 변경이 정확히 포함됐는지 검증한다.

## Baseline 감사

`audit_celebdf_baseline.py`는 frame 수별 NPZ를 runtime에서 읽고 다중 seed·reference 지표와 누수 검사를 생성한다. 출력에는 subject/video ID 대신 fingerprint와 reject reason 집계만 남긴다.

```bash
python3 scripts/audit_celebdf_baseline.py \
  --embedding-run 1=/trusted/frames_1/video_embeddings.npz \
  --embedding-run 5=/trusted/frames_5/video_embeddings.npz \
  --embedding-run 10=/trusted/frames_10/video_embeddings.npz \
  --run-report 1=/trusted/frames_1/run.json \
  --run-report 5=/trusted/frames_5/run.json \
  --run-report 10=/trusted/frames_10/run.json \
  --output-dir outputs/celebdf_baseline_audit/sanitized
```
