"""검증 데이터로 만든 점수 보정 파일을 안전하게 읽고 적용한다."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RiskLevel = Literal["low", "review", "high"]


@dataclass(frozen=True)
class CalibrationResult:
    """API에 공개해도 되는 보정 결과.

    ``calibrated_probability``는 별도의 공개 Gate를 통과한 파일에서만
    채운다. 연구 결과가 있어도 Gate가 실패하면 원점수만 공개한다.
    """

    calibrated_probability: float | None
    calibration_status: str
    calibration_version: str | None
    risk_level: RiskLevel | None
    warning: str


@dataclass(frozen=True)
class ScoreCalibration:
    version: str
    scope: str
    method: Literal["temperature", "platt", "isotonic"]
    parameters: dict[str, object]
    model_fingerprint: str
    low_threshold: float
    high_threshold: float
    review_band_empty: bool
    status: str
    display_approved: bool
    warning: str

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_model_fingerprint: str,
        expected_scope: str,
    ) -> ScoreCalibration | None:
        """선택적 보정 파일을 읽는다. 파일이 없으면 원점수 모드로 둔다."""

        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "1.0":
                raise ValueError("unsupported schema_version")
            if payload.get("model_fingerprint") != expected_model_fingerprint:
                raise ValueError("model fingerprint mismatch")
            if payload.get("scope") != expected_scope:
                raise ValueError("calibration scope mismatch")
            method = payload["selected_method"]
            if method not in {"temperature", "platt", "isotonic"}:
                raise ValueError("unsupported calibration method")
            risk_bands = payload["risk_bands"]
            low_threshold = float(risk_bands["low_max_raw_score"])
            high_threshold = float(risk_bands["high_min_raw_score"])
            if not 0.0 <= low_threshold <= high_threshold <= 1.0:
                raise ValueError("invalid risk-band thresholds")
            review_band_empty = risk_bands["review_band_empty"]
            display_approved = payload["display_approved"]
            gate_overall_pass = payload["gate"]["overall_pass"]
            if type(review_band_empty) is not bool:
                raise ValueError("review_band_empty must be a boolean")
            if type(display_approved) is not bool:
                raise ValueError("display_approved must be a boolean")
            if type(gate_overall_pass) is not bool:
                raise ValueError("gate.overall_pass must be a boolean")
            if display_approved != gate_overall_pass:
                raise ValueError("display approval and overall Gate must match")
            if review_band_empty != (low_threshold == high_threshold):
                raise ValueError("review-band metadata does not match its thresholds")
            calibration = cls(
                version=str(payload["calibration_version"]),
                scope=str(payload["scope"]),
                method=method,
                parameters=dict(payload["parameters"]),
                model_fingerprint=str(payload["model_fingerprint"]),
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                review_band_empty=review_band_empty,
                status=str(payload["calibration_status"]),
                display_approved=display_approved,
                warning=str(payload["warning"]),
            )
            # 로딩 시 대표값도 계산해 파라미터 구조와 유한값을 검증한다.
            calibration._predict(0.5)
            return calibration
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"유효하지 않은 점수 보정 파일입니다: {error}") from error

    def _predict(self, raw_score: float) -> float:
        score = min(max(float(raw_score), 1e-7), 1.0 - 1e-7)
        logit = math.log(score / (1.0 - score))
        if self.method == "temperature":
            temperature = float(self.parameters["temperature"])
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError("temperature must be positive")
            calibrated_logit = logit / temperature
            probability = _sigmoid(calibrated_logit)
        elif self.method == "platt":
            slope = float(self.parameters["slope"])
            intercept = float(self.parameters["intercept"])
            if not math.isfinite(slope) or not math.isfinite(intercept):
                raise ValueError("Platt parameters must be finite")
            probability = _sigmoid(slope * logit + intercept)
        else:
            boundaries = [float(value) for value in self.parameters["boundaries"]]
            values = [float(value) for value in self.parameters["values"]]
            if not boundaries or len(boundaries) != len(values):
                raise ValueError("invalid isotonic parameters")
            if any(
                boundaries[index] > boundaries[index + 1]
                or values[index] > values[index + 1]
                for index in range(len(boundaries) - 1)
            ):
                raise ValueError("isotonic parameters must be monotonic")
            probability = values[-1]
            for boundary, value in zip(boundaries, values, strict=True):
                if score <= boundary:
                    probability = value
                    break
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("calibrated probability must be finite and bounded")
        return probability

    def apply(self, raw_score: float) -> CalibrationResult:
        if not 0.0 <= raw_score <= 1.0:
            raise ValueError("raw_score는 0과 1 사이여야 합니다.")
        risk_level: RiskLevel
        if raw_score < self.low_threshold:
            risk_level = "low"
        elif raw_score >= self.high_threshold:
            risk_level = "high"
        else:
            risk_level = "review"
        probability = self._predict(raw_score) if self.display_approved else None
        return CalibrationResult(
            calibrated_probability=probability,
            calibration_status=self.status,
            calibration_version=self.version,
            risk_level=risk_level,
            warning=self.warning,
        )


def unavailable_calibration_result() -> CalibrationResult:
    return CalibrationResult(
        calibrated_probability=None,
        calibration_status="not_available",
        calibration_version=None,
        risk_level=None,
        warning=(
            "검증 데이터로 만든 점수 보정 파일이 설치되지 않아 원점수만 제공합니다. "
            "원점수를 확률이나 신뢰도 퍼센트로 표시하지 마세요."
        ),
    )


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)
