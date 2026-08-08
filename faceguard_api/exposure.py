"""딥소각 공개 노출 스캔을 비동기 작업으로 관리한다.

이 모듈의 저장소는 데모용 프로세스 메모리다. 원본 사진은 등록 요청이
끝나면 버리고, 풀링한 임베딩과 품질 정보만 TTL 동안 보관한다.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

from .candidate_filter import (
    CandidateFaceDecision,
    CandidateFilterResult,
    CandidateFilterService,
    PreparedReferences,
)
from .errors import FaceGuardError
from .search import SearchQuery, SearchService

ScanStatus = Literal[
    "queued",
    "searching",
    "identity_filtering",
    "deepfake_analyzing",
    "completed",
    "partial_failed",
    "failed",
]
FINAL_SCAN_STATUSES = frozenset({"completed", "partial_failed", "failed"})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class EnrollmentRecord:
    enrollment_id: str
    references: PreparedReferences
    created_at: datetime
    expires_at: datetime


class EphemeralEnrollmentStore:
    """TTL이 지나면 등록 임베딩을 메모리에서 제거한다."""

    def __init__(
        self,
        *,
        ttl_seconds: int,
        maximum_entries: int,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if ttl_seconds <= 0 or maximum_entries <= 0:
            raise ValueError("등록 저장소 제한값은 양수여야 합니다.")
        self.ttl_seconds = ttl_seconds
        self.maximum_entries = maximum_entries
        self._clock = clock
        self._records: dict[str, EnrollmentRecord] = {}
        self._lock = asyncio.Lock()

    def _prune_locked(self, now: datetime) -> None:
        expired = [
            key for key, record in self._records.items() if record.expires_at <= now
        ]
        for key in expired:
            self._records.pop(key, None)

    async def create(self, references: PreparedReferences) -> EnrollmentRecord:
        now = self._clock()
        async with self._lock:
            self._prune_locked(now)
            if len(self._records) >= self.maximum_entries:
                raise FaceGuardError(
                    "ENROLLMENT_CAPACITY_EXCEEDED",
                    "현재 사용 중인 얼굴 등록이 많습니다. 잠시 후 다시 시도해 주세요.",
                    429,
                )
            record = EnrollmentRecord(
                enrollment_id=str(uuid4()),
                references=references,
                created_at=now,
                expires_at=now + timedelta(seconds=self.ttl_seconds),
            )
            self._records[record.enrollment_id] = record
            return record

    async def get(self, enrollment_id: str) -> EnrollmentRecord:
        now = self._clock()
        async with self._lock:
            record = self._records.get(enrollment_id)
            if record is None:
                raise FaceGuardError(
                    "ENROLLMENT_NOT_FOUND",
                    "등록 정보를 찾지 못했습니다. 사진을 다시 등록해 주세요.",
                    404,
                )
            if record.expires_at <= now:
                self._records.pop(enrollment_id, None)
                raise FaceGuardError(
                    "ENROLLMENT_EXPIRED",
                    "등록 유효 시간이 지났습니다. 사진을 다시 등록해 주세요.",
                    410,
                )
            return record

    async def delete(self, enrollment_id: str) -> bool:
        async with self._lock:
            return self._records.pop(enrollment_id, None) is not None


@dataclass
class ExposureProgress:
    searched_candidate_count: int = 0
    analyzed_candidate_count: int = 0
    skipped_candidate_count: int = 0
    identity_match_count: int = 0
    deepfake_completed_count: int = 0
    deepfake_failed_count: int = 0


@dataclass
class ExposureCandidate:
    candidate_id: str
    decision: CandidateFaceDecision


@dataclass
class ExposureScanRecord:
    scan_id: str
    enrollment_id: str
    query: SearchQuery
    request_fingerprint: str
    idempotency_key: str | None
    status: ScanStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: ExposureProgress = field(default_factory=ExposureProgress)
    providers: tuple = ()
    candidates: tuple[ExposureCandidate, ...] = ()
    error_code: str | None = None
    processing_ms: float = 0.0
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    stage_history: list[str] = field(default_factory=lambda: ["queued"])


def scan_request_fingerprint(enrollment_id: str, query: SearchQuery) -> str:
    """멱등성 키가 다른 요청에 재사용되는 실수를 막는 비식별 hash."""

    payload = {
        "enrollment_id": enrollment_id,
        "privacy_mode": query.privacy_mode,
        "web_monitoring_consent": query.web_monitoring_consent,
        "text_query": query.text_query,
        "categories": list(query.categories),
        "language": query.language,
        "safe_search": query.safe_search,
        "maximum_results": query.maximum_results,
        "submitted_candidates": [
            {
                "page_url": item.page_url,
                "media_url": item.media_url,
                "thumbnail_url": item.thumbnail_url,
                "content_sha256": item.content_sha256,
                "perceptual_hash": item.perceptual_hash,
            }
            for item in query.submitted_candidates
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def redacted_search_query(query: SearchQuery) -> SearchQuery:
    """작업 종료 후 사용자 검색어와 제보 입력을 메모리에 남기지 않는다."""

    return SearchQuery(
        privacy_mode=query.privacy_mode,
        web_monitoring_consent=query.web_monitoring_consent,
        submitted_candidates=(),
        text_query=None,
        categories=query.categories,
        language=query.language,
        safe_search=query.safe_search,
        maximum_results=query.maximum_results,
    )


class ExposureScanManager:
    """검색·얼굴 선별·딥페이크 분석을 메모리 작업으로 연결한다."""

    def __init__(
        self,
        *,
        search_service: SearchService,
        candidate_filter_service: CandidateFilterService,
        enrollment_store: EphemeralEnrollmentStore,
        scan_ttl_seconds: int,
        maximum_scans: int,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if scan_ttl_seconds <= 0 or maximum_scans <= 0:
            raise ValueError("스캔 저장소 제한값은 양수여야 합니다.")
        self.search_service = search_service
        self.candidate_filter_service = candidate_filter_service
        self.enrollment_store = enrollment_store
        self.scan_ttl_seconds = scan_ttl_seconds
        self.maximum_scans = maximum_scans
        self._clock = clock
        self._records: dict[str, ExposureScanRecord] = {}
        self._candidate_index: dict[str, tuple[str, ExposureCandidate]] = {}
        self._idempotency_index: dict[str, tuple[str, str]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    def _remove_locked(self, scan_id: str) -> None:
        record = self._records.pop(scan_id, None)
        if record is None:
            return
        for candidate in record.candidates:
            self._candidate_index.pop(candidate.candidate_id, None)
        if record.idempotency_key:
            indexed = self._idempotency_index.get(record.idempotency_key)
            if indexed and indexed[1] == scan_id:
                self._idempotency_index.pop(record.idempotency_key, None)

    def _prune_locked(self, now: datetime) -> None:
        expired = [
            key for key, record in self._records.items() if record.expires_at <= now
        ]
        for key in expired:
            self._remove_locked(key)

    async def create_enrollment(
        self, reference_payloads: list[bytes]
    ) -> EnrollmentRecord:
        prepared = await self.candidate_filter_service.prepare_references(
            reference_payloads
        )
        return await self.enrollment_store.create(prepared)

    async def create_scan(
        self,
        *,
        enrollment_id: str,
        query: SearchQuery,
        idempotency_key: str | None,
    ) -> tuple[ExposureScanRecord, bool]:
        await self.enrollment_store.get(enrollment_id)
        fingerprint = scan_request_fingerprint(enrollment_id, query)
        idempotency_token = (
            hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
            if idempotency_key
            else None
        )
        now = self._clock()
        async with self._lock:
            self._prune_locked(now)
            if idempotency_token:
                previous = self._idempotency_index.get(idempotency_token)
                if previous:
                    previous_fingerprint, scan_id = previous
                    if previous_fingerprint != fingerprint:
                        raise FaceGuardError(
                            "IDEMPOTENCY_KEY_REUSED",
                            "같은 Idempotency-Key를 다른 스캔 요청에 사용할 수 없습니다.",
                            409,
                        )
                    existing = self._records.get(scan_id)
                    if existing is not None:
                        return existing, True
            if len(self._records) >= self.maximum_scans:
                removable = [
                    record
                    for record in self._records.values()
                    if record.status in FINAL_SCAN_STATUSES
                ]
                if not removable:
                    raise FaceGuardError(
                        "SCAN_CAPACITY_EXCEEDED",
                        "현재 실행 중인 스캔이 많습니다. 잠시 후 다시 시도해 주세요.",
                        429,
                    )
                oldest = min(removable, key=lambda record: record.created_at)
                self._remove_locked(oldest.scan_id)
            record = ExposureScanRecord(
                scan_id=str(uuid4()),
                enrollment_id=enrollment_id,
                query=query,
                request_fingerprint=fingerprint,
                idempotency_key=idempotency_token,
                status="queued",
                created_at=now,
                updated_at=now,
                expires_at=now + timedelta(seconds=self.scan_ttl_seconds),
            )
            self._records[record.scan_id] = record
            if idempotency_token:
                self._idempotency_index[idempotency_token] = (
                    fingerprint,
                    record.scan_id,
                )
            task = asyncio.create_task(self._run(record.scan_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return record, False

    async def _transition(self, record: ExposureScanRecord, status: ScanStatus) -> None:
        record.status = status
        record.updated_at = self._clock()
        if not record.stage_history or record.stage_history[-1] != status:
            record.stage_history.append(status)
        await asyncio.sleep(0)

    async def _run(self, scan_id: str) -> None:
        record = self._records.get(scan_id)
        if record is None:
            return
        started = time.perf_counter()
        record.started_at = self._clock()
        stage_started = time.perf_counter()
        try:
            enrollment = await self.enrollment_store.get(record.enrollment_id)
            await self._transition(record, "searching")
            search_result = await self.search_service.search(record.query)
            record.stage_durations_ms["searching"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            record.progress.searched_candidate_count = len(search_result.candidates)
            record.providers = search_result.providers

            stage_started = time.perf_counter()
            await self._transition(record, "identity_filtering")

            async def deepfake_stage_started() -> None:
                nonlocal stage_started
                if record.status != "identity_filtering":
                    return
                record.stage_durations_ms["identity_filtering"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                stage_started = time.perf_counter()
                await self._transition(record, "deepfake_analyzing")

            async def candidate_progress(
                decisions: tuple[CandidateFaceDecision, ...],
            ) -> None:
                analyzed_count = sum(item.status != "skipped" for item in decisions)
                record.progress = ExposureProgress(
                    searched_candidate_count=record.progress.searched_candidate_count,
                    analyzed_candidate_count=analyzed_count,
                    skipped_candidate_count=len(decisions) - analyzed_count,
                    identity_match_count=sum(
                        item.identity_match is True for item in decisions
                    ),
                    deepfake_completed_count=sum(
                        item.deepfake.status == "analyzed" for item in decisions
                    ),
                    deepfake_failed_count=sum(
                        item.deepfake.status in {"failed", "unavailable"}
                        for item in decisions
                    ),
                )
                record.updated_at = self._clock()
                await asyncio.sleep(0)

            filtered = await self.candidate_filter_service.filter_prepared(
                enrollment.references,
                search_result.candidates,
                deepfake_stage_callback=deepfake_stage_started,
                progress_callback=candidate_progress,
            )
            if record.status == "identity_filtering":
                record.stage_durations_ms["identity_filtering"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                stage_started = time.perf_counter()
                await self._transition(record, "deepfake_analyzing")
            self._store_results(record, filtered)
            record.stage_durations_ms["deepfake_analyzing"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            final_status: ScanStatus = (
                "partial_failed"
                if search_result.status == "partial_failed"
                or filtered.skipped_candidate_count
                or filtered.deepfake_failed_candidate_count
                else "completed"
            )
            await self._transition(record, final_status)
        except asyncio.CancelledError:
            record.error_code = "SCAN_CANCELLED"
            await self._transition(record, "failed")
            raise
        except FaceGuardError as error:
            record.error_code = error.code
            await self._transition(record, "failed")
        except Exception:  # noqa: BLE001 - 내부 예외 내용은 응답에 노출하지 않는다.
            record.error_code = "SCAN_INTERNAL_ERROR"
            await self._transition(record, "failed")
        finally:
            record.query = redacted_search_query(record.query)
            record.processing_ms = (time.perf_counter() - started) * 1000.0
            record.completed_at = self._clock()
            record.updated_at = record.completed_at

    def _store_results(
        self, record: ExposureScanRecord, filtered: CandidateFilterResult
    ) -> None:
        candidates = tuple(
            ExposureCandidate(candidate_id=str(uuid4()), decision=decision)
            for decision in filtered.candidates
        )
        record.candidates = candidates
        record.progress = ExposureProgress(
            searched_candidate_count=record.progress.searched_candidate_count,
            analyzed_candidate_count=filtered.analyzed_candidate_count,
            skipped_candidate_count=filtered.skipped_candidate_count,
            identity_match_count=filtered.identity_match_count,
            deepfake_completed_count=filtered.deepfake_analyzed_candidate_count,
            deepfake_failed_count=filtered.deepfake_failed_candidate_count,
        )
        for candidate in candidates:
            self._candidate_index[candidate.candidate_id] = (record.scan_id, candidate)

    async def get_scan(self, scan_id: str) -> ExposureScanRecord:
        now = self._clock()
        async with self._lock:
            record = self._records.get(scan_id)
            if record is None:
                raise FaceGuardError(
                    "SCAN_NOT_FOUND", "스캔 작업을 찾지 못했습니다.", 404
                )
            if record.expires_at <= now:
                self._remove_locked(scan_id)
                raise FaceGuardError(
                    "SCAN_EXPIRED", "스캔 결과 보관 시간이 지났습니다.", 410
                )
            return record

    async def get_candidate(
        self, candidate_id: str
    ) -> tuple[ExposureScanRecord, ExposureCandidate]:
        async with self._lock:
            indexed = self._candidate_index.get(candidate_id)
        if indexed is None:
            raise FaceGuardError(
                "EXPOSURE_CANDIDATE_NOT_FOUND", "노출 후보를 찾지 못했습니다.", 404
            )
        scan_id, candidate = indexed
        return await self.get_scan(scan_id), candidate

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
