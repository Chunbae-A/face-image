"""InsightFace를 지연 로딩하여 이미지 한 장을 얼굴 임베딩으로 변환한다."""

from __future__ import annotations

import hashlib
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from .domain import EncodedFace, FaceQuality, l2_normalize
from .errors import FaceGuardError, ModelUnavailableError
from .settings import Settings

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class InsightFaceEncoder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._application: Any | None = None
        self._provider: str | None = None
        self._model_fingerprint: str | None = None
        self._initialization_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._application is not None

    @property
    def provider(self) -> str | None:
        return self._provider

    @property
    def model_fingerprint(self) -> str | None:
        return self._model_fingerprint

    @staticmethod
    def _fingerprint_loaded_models(application: Any) -> str:
        model_paths = sorted(
            {
                Path(model.model_file)
                for model in application.models.values()
                if getattr(model, "model_file", None)
            },
            key=lambda path: path.name,
        )
        if not model_paths or any(not path.is_file() for path in model_paths):
            raise ModelUnavailableError(
                "불러온 얼굴 모델 파일의 해시를 계산하지 못했습니다."
            )
        combined = hashlib.sha256()
        for path in model_paths:
            file_hash = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_hash.update(chunk)
            combined.update(path.name.encode("utf-8"))
            combined.update(b"\0")
            combined.update(file_hash.digest())
        return combined.hexdigest()

    def _initialize(self) -> None:
        if self._application is not None:
            return
        with self._initialization_lock:
            if self._application is not None:
                return
            if not self.settings.accept_noncommercial_model_license:
                raise ModelUnavailableError(
                    "InsightFace 비상업 연구용 가중치 조건을 확인한 뒤 환경변수를 설정하세요."
                )
            try:
                import onnxruntime as ort
                from insightface.app import FaceAnalysis

                available = ort.get_available_providers()
                if (
                    self.settings.device == "cuda"
                    and "CUDAExecutionProvider" not in available
                ):
                    raise ModelUnavailableError(
                        "요청한 CUDA 실행 제공자를 찾지 못했습니다."
                    )
                use_cuda = (
                    self.settings.device in {"auto", "cuda"}
                    and "CUDAExecutionProvider" in available
                )
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if use_cuda
                    else ["CPUExecutionProvider"]
                )
                application = FaceAnalysis(
                    name=self.settings.model_name,
                    root=str(self.settings.model_root),
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                application.prepare(
                    ctx_id=0 if use_cuda else -1,
                    det_size=(
                        self.settings.detection_size,
                        self.settings.detection_size,
                    ),
                )
            except FaceGuardError:
                raise
            except Exception as error:
                raise ModelUnavailableError(
                    f"얼굴 모델 준비에 실패했습니다({type(error).__name__}). 서버 설치와 모델 경로를 확인하세요."
                ) from error
            self._application = application
            self._provider = providers[0]
            self._model_fingerprint = self._fingerprint_loaded_models(application)

    def _decode_image(self, payload: bytes) -> np.ndarray:
        if not payload:
            raise FaceGuardError("EMPTY_IMAGE", "빈 이미지 파일은 사용할 수 없습니다.")
        if len(payload) > self.settings.maximum_image_bytes:
            raise FaceGuardError(
                "IMAGE_TOO_LARGE",
                f"이미지 한 장은 {self.settings.maximum_image_bytes} bytes 이하여야 합니다.",
                413,
            )
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise ModelUnavailableError(
                "Pillow 이미지 라이브러리를 불러오지 못했습니다."
            ) from error
        try:
            with Image.open(BytesIO(payload)) as image:
                image_format = (image.format or "").upper()
                width, height = image.size
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    raise FaceGuardError(
                        "UNSUPPORTED_IMAGE_FORMAT",
                        "JPEG, PNG, WEBP 이미지만 사용할 수 있습니다.",
                        415,
                    )
                if width <= 0 or height <= 0:
                    raise FaceGuardError(
                        "INVALID_IMAGE", "이미지 크기가 올바르지 않습니다."
                    )
                if width * height > self.settings.maximum_image_pixels:
                    raise FaceGuardError(
                        "TOO_MANY_PIXELS",
                        f"이미지는 {self.settings.maximum_image_pixels} 픽셀 이하여야 합니다.",
                        413,
                    )
                image.verify()
        except FaceGuardError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise FaceGuardError(
                "INVALID_IMAGE", "손상되었거나 읽을 수 없는 이미지입니다."
            ) from error

        try:
            import cv2
        except ImportError as error:
            raise ModelUnavailableError(
                "OpenCV 이미지 라이브러리를 불러오지 못했습니다."
            ) from error
        try:
            decoded = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except Exception as error:
            raise FaceGuardError(
                "INVALID_IMAGE", "이미지를 변환하지 못했습니다."
            ) from error
        if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
            raise FaceGuardError("INVALID_IMAGE", "3채널 컬러 이미지가 필요합니다.")
        return decoded

    def encode(self, payload: bytes) -> EncodedFace:
        self._initialize()
        image = self._decode_image(payload)
        assert self._application is not None
        with self._inference_lock:
            faces = self._application.get(image)
        if not faces:
            raise FaceGuardError(
                "NO_FACE",
                "얼굴을 찾지 못했습니다. 정면에서 더 밝고 가깝게 다시 촬영하세요.",
            )
        if len(faces) != 1:
            raise FaceGuardError(
                "MULTIPLE_FACES", "사진에는 한 사람의 얼굴만 있어야 합니다."
            )

        face = faces[0]
        height, width = image.shape[:2]
        bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
        if bbox.size != 4 or not np.all(np.isfinite(bbox)):
            raise FaceGuardError(
                "INVALID_FACE", "얼굴 위치 정보를 확인하지 못했습니다."
            )
        x1 = max(0, min(width, int(np.floor(bbox[0]))))
        y1 = max(0, min(height, int(np.floor(bbox[1]))))
        x2 = max(0, min(width, int(np.ceil(bbox[2]))))
        y2 = max(0, min(height, int(np.ceil(bbox[3]))))
        if x2 <= x1 or y2 <= y1:
            raise FaceGuardError("INVALID_FACE", "얼굴 영역이 올바르지 않습니다.")

        detection_score = float(getattr(face, "det_score", 0.0))
        face_area_ratio = float(((x2 - x1) * (y2 - y1)) / (width * height))
        if detection_score < self.settings.minimum_detection_score:
            raise FaceGuardError(
                "LOW_DETECTION_SCORE",
                "얼굴이 불분명합니다. 더 밝고 선명하게 다시 촬영하세요.",
            )
        if face_area_ratio < self.settings.minimum_face_area_ratio:
            raise FaceGuardError(
                "FACE_TOO_SMALL", "얼굴이 너무 작습니다. 카메라에 더 가까이 촬영하세요."
            )

        import cv2

        face_region = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_mean = float(gray.mean())

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise ModelUnavailableError("얼굴 모델이 임베딩을 반환하지 않았습니다.")

        return EncodedFace(
            embedding=l2_normalize(np.asarray(embedding, dtype=np.float32)),
            quality=FaceQuality(
                detection_score=detection_score,
                face_area_ratio=face_area_ratio,
                blur_score=blur_score,
                brightness_mean=brightness_mean,
                image_width=width,
                image_height=height,
            ),
        )
