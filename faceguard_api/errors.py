"""외부에 안전하게 반환할 얼굴가드 오류."""

from __future__ import annotations


class FaceGuardError(Exception):
    """파일명·경로·임베딩 등 민감정보를 포함하지 않는 API 오류."""

    def __init__(self, code: str, message: str, http_status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class ModelUnavailableError(FaceGuardError):
    def __init__(self, message: str) -> None:
        super().__init__("MODEL_UNAVAILABLE", message, 503)
