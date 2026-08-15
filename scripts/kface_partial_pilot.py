#!/usr/bin/env python3
"""다운로드 중인 K-FACE ZIP의 완성된 인물만 CRC 검증해 파일럿 ZIP으로 묶는다."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from kface_pilot import (
    _assert_safe_member,
    _normalized_member,
    list_subject_archives,
)

LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
LOCAL_HEADER = struct.Struct("<HHHHHIIIHH")
MAXIMUM_INNER_ZIP_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class LocalMember:
    name: str
    flags: int
    compression_method: int
    crc32: int
    compressed_bytes: int
    uncompressed_bytes: int
    data_offset: int
    next_offset: int


def _decode_name(payload: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x800 else "cp437"
    return _assert_safe_member(payload.decode(encoding))


def read_local_member(handle: BinaryIO, offset: int, snapshot_bytes: int) -> LocalMember | None:
    if offset + 30 > snapshot_bytes:
        return None
    handle.seek(offset)
    if handle.read(4) != LOCAL_FILE_SIGNATURE:
        return None
    header = handle.read(LOCAL_HEADER.size)
    if len(header) != LOCAL_HEADER.size:
        return None
    (
        _version,
        flags,
        method,
        _mtime,
        _mdate,
        crc32,
        compressed_bytes,
        uncompressed_bytes,
        name_length,
        extra_length,
    ) = LOCAL_HEADER.unpack(header)
    if flags & 0x1:
        raise ValueError("암호화된 ZIP 멤버는 처리할 수 없습니다.")
    if flags & 0x8:
        raise ValueError("데이터 설명자를 쓰는 ZIP 멤버는 부분 처리할 수 없습니다.")
    if method not in {0, 8}:
        raise ValueError(f"지원하지 않는 ZIP 압축 방식입니다: {method}")
    if compressed_bytes == 0xFFFFFFFF or uncompressed_bytes == 0xFFFFFFFF:
        raise ValueError("ZIP64 내부 멤버는 부분 처리할 수 없습니다.")
    if uncompressed_bytes > MAXIMUM_INNER_ZIP_BYTES:
        raise ValueError("인물별 ZIP의 압축 해제 크기가 안전 한도를 넘었습니다.")
    name_payload = handle.read(name_length)
    if len(name_payload) != name_length:
        return None
    name = _decode_name(name_payload, flags)
    data_offset = offset + 30 + name_length + extra_length
    next_offset = data_offset + compressed_bytes
    if next_offset > snapshot_bytes:
        return None
    return LocalMember(
        name=name,
        flags=flags,
        compression_method=method,
        crc32=crc32,
        compressed_bytes=compressed_bytes,
        uncompressed_bytes=uncompressed_bytes,
        data_offset=data_offset,
        next_offset=next_offset,
    )


def _read_verified_payload(handle: BinaryIO, member: LocalMember) -> bytes:
    handle.seek(member.data_offset)
    compressed = handle.read(member.compressed_bytes)
    if len(compressed) != member.compressed_bytes:
        raise OSError("부분 ZIP에서 완성된 압축 데이터를 읽지 못했습니다.")
    if member.compression_method == 0:
        payload = compressed
    else:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    if len(payload) != member.uncompressed_bytes:
        raise OSError("인물별 ZIP의 압축 해제 크기가 헤더와 다릅니다.")
    if zlib.crc32(payload) & 0xFFFFFFFF != member.crc32:
        raise OSError("인물별 ZIP의 CRC 검증에 실패했습니다.")
    if not zipfile.is_zipfile(BytesIO(payload)):
        raise ValueError("완성된 멤버가 인물별 ZIP 형식이 아닙니다.")
    return payload


def build_pilot_archive(
    partial_path: Path,
    *,
    reference_archive: Path,
    output_path: Path,
    max_subjects: int,
) -> dict[str, int | bool | str]:
    if max_subjects <= 0:
        raise ValueError("파일럿 인물 수는 양수여야 합니다.")
    if output_path.exists():
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {output_path}")
    reference = list_subject_archives(reference_archive)[:max_subjects]
    target_names = {_normalized_member(subject.outer_member) for subject in reference}
    selected: set[str] = set()
    snapshot_bytes = partial_path.stat().st_size
    offset = 0
    scanned_members = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"기존 임시 파일을 덮어쓰지 않습니다: {temporary}")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as output, partial_path.open("rb") as source:
            while offset < snapshot_bytes and len(selected) < len(target_names):
                member = read_local_member(source, offset, snapshot_bytes)
                if member is None:
                    break
                scanned_members += 1
                normalized = _normalized_member(member.name)
                if normalized in target_names:
                    if normalized in selected:
                        raise ValueError("부분 ZIP에 중복된 인물별 ZIP이 있습니다.")
                    output.writestr(normalized, _read_verified_payload(source, member))
                    selected.add(normalized)
                offset = member.next_offset
            missing = target_names - selected
            if missing:
                raise RuntimeError(
                    "부분 다운로드에 필요한 인물 ZIP이 아직 부족합니다: "
                    f"{len(missing)}개"
                )
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "dataset": "K-FACE",
        "source": "verified_completed_prefix",
        "snapshot_bytes": snapshot_bytes,
        "scanned_local_members": scanned_members,
        "selected_subjects": len(selected),
        "pilot_archive_bytes": output_path.stat().st_size,
        "contains_raw_faces": True,
        "private_artifact": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partial", type=Path)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-subjects", type=int, default=30)
    args = parser.parse_args(argv)
    result = build_pilot_archive(
        args.partial,
        reference_archive=args.reference_archive,
        output_path=args.output,
        max_subjects=args.max_subjects,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
