# 영상 딥페이크 API 로컬 연결 시험

## 결론

`POST /v1/deepfake/analyze-video`에 연결되는 영상 16프레임 분석 로직을 실제 비공개 EfficientNet-B4 ONNX와 `CPUExecutionProvider`로 실행해 정상 완료했다.

- 입력: 사용 동의를 받은 실제 얼굴 이미지로 런타임에서만 만든 4초 MP4
- 대표 프레임: 요청 16개, 분석 16개, 실패 0개
- 영상 점수: `0.0001615129`
- 연구 기준값: `0.7519882694`
- 판정: 딥페이크 의심 아님
- 전체 처리시간: 약 `5,293.5ms`
- ONNX 추론시간 합계: 약 `1,235.4ms`
- 모델 SHA-256: `c32a8532e2e1bd275b833b16460946eb307207098e0c07e2247851b71c23a6f1`

Docker 이미지 재빌드 후 `/v1/deepfake/analyze-video`에도 같은 방식의 임시 MP4를 전송했다. 컨테이너 health와 OpenAPI 노출을 확인했고 HTTP `200`, 대표 프레임 `16/16`, CPU 모델 해시 일치와 `config_version=deepfake-video-16-frame-mean-v1`을 확인했다.

## 개인정보 처리

원본 이미지의 경로·파일명·내용, 임시 MP4, 얼굴 crop과 임베딩은 보고서와 GitHub에 포함하지 않았다. 임시 MP4는 별도 임시 디렉터리에서 생성하고 시험 종료와 함께 삭제했다.

## 해석 제한

- 같은 정지 이미지를 반복해 만든 MP4이므로 디코딩·프레임 선택·얼굴 추적·ONNX 평균 연결만 확인한다.
- 이 한 건으로 실제 영상 정확도, 최신 딥페이크 탐지력, 한국인 일반화 또는 응답시간 SLA를 증명하지 않는다.
- 현재 연구 모델은 Celeb-DF 공식 Test의 실제 영상 오경고율 Gate를 통과하지 못했다.
- `video_score`는 보정된 확률이 아니며 사람 검토 없이 신고·삭제에 사용하지 않는다.

비식별 기계 판독 결과는 [`video_deepfake_api_smoke.json`](video_deepfake_api_smoke.json)에 있다.
