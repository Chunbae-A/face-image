"""Celeb-DF EfficientNet-B4 ONNX를 이용한 연구용 단일 얼굴 분석."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from .domain import AlignedEncodedFace, FaceQuality
from .errors import ModelUnavailableError
from .settings import Settings

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class FaceAligningEncoder(Protocol):
    def encode_and_align(
        self, payload: bytes, *, aligned_face_size: int
    ) -> AlignedEncodedFace: ...


@dataclass(frozen=True)
class DeepfakeAnalysis:
    is_suspected_deepfake: bool
    deepfake_score: float
    raw_logit: float
    threshold: float
    quality: FaceQuality
    processing_ms: float
    inference_ms: float
    model_name: str
    execution_provider: str
    model_fingerprint: str


def _sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def preprocess_aligned_face(aligned_bgr: np.ndarray, input_size: int) -> np.ndarray:
    """학습의 Resize→ToTensor→ImageNet Normalize와 같은 입력을 만든다."""

    value = np.asarray(aligned_bgr)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("정렬 얼굴은 uint8 BGR 3채널 이미지여야 합니다.")
    if input_size <= 0:
        raise ValueError("딥페이크 모델 입력 크기는 양수여야 합니다.")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - API 의존성 누락
        raise ModelUnavailableError(
            "Pillow 이미지 라이브러리를 불러오지 못했습니다."
        ) from error

    rgb = np.ascontiguousarray(value[..., ::-1])
    resized = Image.fromarray(rgb).resize(
        (input_size, input_size),
        resample=Image.Resampling.BILINEAR,
    )
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])


class DeepfakeOnnxAnalyzer:
    """비공개 ONNX를 지연 로딩하고 원본이나 얼굴 crop을 저장하지 않는다."""

    def __init__(
        self,
        settings: Settings,
        face_encoder: FaceAligningEncoder,
        *,
        session_factory: Callable[[str, list[str]], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.face_encoder = face_encoder
        self._session_factory = session_factory
        self._session: Any | None = None
        self._input_name: str | None = None
        self._provider: str | None = None
        self._model_fingerprint: str | None = None
        self._initialization_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def provider(self) -> str | None:
        return self._provider

    @property
    def model_fingerprint(self) -> str | None:
        return self._model_fingerprint

    def _providers(self) -> list[str]:
        if self._session_factory is not None:
            return (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.settings.deepfake_device == "cuda"
                else ["CPUExecutionProvider"]
            )
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ModelUnavailableError(
                "ONNX Runtime을 불러오지 못했습니다. API 의존성을 확인하세요."
            ) from error
        available = ort.get_available_providers()
        if (
            self.settings.deepfake_device == "cuda"
            and "CUDAExecutionProvider" not in available
        ):
            raise ModelUnavailableError(
                "딥페이크 모델에 요청한 CUDA 실행 제공자를 찾지 못했습니다."
            )
        use_cuda = (
            self.settings.deepfake_device in {"auto", "cuda"}
            and "CUDAExecutionProvider" in available
        )
        return (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )

    def _initialize(self) -> None:
        if self._session is not None:
            return
        with self._initialization_lock:
            if self._session is not None:
                return
            model_path = self.settings.deepfake_model_path
            if not model_path.is_file():
                raise ModelUnavailableError(
                    "딥페이크 ONNX 파일을 찾지 못했습니다. 비공개 모델 마운트 경로를 확인하세요."
                )
            fingerprint = _sha256(model_path)
            if fingerprint != self.settings.deepfake_model_sha256:
                raise ModelUnavailableError(
                    "딥페이크 ONNX 무결성 해시가 연구 보고서와 일치하지 않습니다."
                )
            providers = self._providers()
            try:
                if self._session_factory is None:
                    import onnxruntime as ort

                    session = ort.InferenceSession(
                        str(model_path),
                        providers=providers,
                    )
                else:
                    session = self._session_factory(str(model_path), providers)
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                if len(inputs) != 1 or len(outputs) != 1:
                    raise ValueError("expected one input and one output")
                input_shape = list(inputs[0].shape)
                expected = [
                    3,
                    self.settings.deepfake_input_size,
                    self.settings.deepfake_input_size,
                ]
                if len(input_shape) != 4 or input_shape[-3:] != expected:
                    raise ValueError(f"unexpected input shape: {input_shape}")
                session_providers = list(session.get_providers())
                if not session_providers:
                    raise ValueError("execution provider is missing")
            except ModelUnavailableError:
                raise
            except Exception as error:
                raise ModelUnavailableError(
                    f"딥페이크 ONNX 준비에 실패했습니다({type(error).__name__})."
                ) from error
            self._session = session
            self._input_name = inputs[0].name
            self._provider = session_providers[0]
            self._model_fingerprint = fingerprint

    def _infer(self, tensor: np.ndarray) -> tuple[float, float]:
        self._initialize()
        assert self._session is not None
        assert self._input_name is not None
        started = time.perf_counter()
        try:
            with self._inference_lock:
                output = self._session.run(None, {self._input_name: tensor})[0]
        except Exception as error:
            raise ModelUnavailableError(
                f"딥페이크 ONNX 추론에 실패했습니다({type(error).__name__})."
            ) from error
        inference_ms = (time.perf_counter() - started) * 1000.0
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size != 1 or not np.all(np.isfinite(values)):
            raise ModelUnavailableError(
                "딥페이크 ONNX가 유효한 단일 점수를 반환하지 않았습니다."
            )
        return float(values[0]), inference_ms

    def smoke(self) -> float:
        """얼굴 데이터 없이 모델 로딩과 유한 출력만 확인한다."""

        tensor = np.zeros(
            (
                1,
                3,
                self.settings.deepfake_input_size,
                self.settings.deepfake_input_size,
            ),
            dtype=np.float32,
        )
        logit, _ = self._infer(tensor)
        return _sigmoid(logit)

    def analyze(self, payload: bytes) -> DeepfakeAnalysis:
        started = time.perf_counter()
        self._initialize()
        aligned = self.face_encoder.encode_and_align(
            payload,
            aligned_face_size=self.settings.deepfake_aligned_face_size,
        )
        result = self.analyze_aligned(aligned)
        return replace(
            result,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    def analyze_aligned(self, aligned: AlignedEncodedFace) -> DeepfakeAnalysis:
        """이미 검출·정렬된 얼굴을 영상 프레임 추론에 재사용한다."""

        started = time.perf_counter()
        self._initialize()
        tensor = preprocess_aligned_face(
            aligned.aligned_bgr,
            self.settings.deepfake_input_size,
        )
        logit, inference_ms = self._infer(tensor)
        score = _sigmoid(logit)
        assert self._provider is not None
        assert self._model_fingerprint is not None
        return DeepfakeAnalysis(
            is_suspected_deepfake=score >= self.settings.deepfake_threshold,
            deepfake_score=score,
            raw_logit=logit,
            threshold=self.settings.deepfake_threshold,
            quality=aligned.face.quality,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            inference_ms=inference_ms,
            model_name=self.settings.deepfake_model_name,
            execution_provider=self._provider,
            model_fingerprint=self._model_fingerprint,
        )
