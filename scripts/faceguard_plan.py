#!/usr/bin/env python3
"""FaceGuard resource planning helpers.

All storage units are decimal GB/MB unless the operating system reports bytes.
The command deliberately labels calculated values as estimates; it never treats
them as benchmark measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


BITS_PER_DECIMAL_GB = 8_000_000_000
BITS_PER_MEGABIT = 1_000_000
BYTES_PER_DECIMAL_GB = 1_000_000_000


@dataclass(frozen=True)
class DownloadEstimate:
    link_mbps: float
    theoretical_seconds: float
    expected_seconds: float


def download_seconds(size_gb: float, link_mbps: float, efficiency: float = 1.0) -> float:
    """Return transfer duration from decimal GB, Mbps, and usable link ratio."""
    if size_gb <= 0:
        raise ValueError("size_gb must be positive")
    if link_mbps <= 0:
        raise ValueError("link_mbps must be positive")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return (size_gb * BITS_PER_DECIMAL_GB) / (
        link_mbps * BITS_PER_MEGABIT * efficiency
    )


def download_estimates(
    size_gb: float, speeds_mbps: Iterable[float], efficiency: float
) -> list[DownloadEstimate]:
    return [
        DownloadEstimate(
            link_mbps=speed,
            theoretical_seconds=download_seconds(size_gb, speed),
            expected_seconds=download_seconds(size_gb, speed, efficiency),
        )
        for speed in speeds_mbps
    ]


def process_seconds(size_gb: float, throughput_mbps: float) -> float:
    """Estimate local processing time from decimal GB and measured MB/s."""
    if size_gb <= 0 or throughput_mbps <= 0:
        raise ValueError("size_gb and throughput_mbps must be positive")
    return size_gb * 1_000 / throughput_mbps


def embedding_gb(images: int, dimensions: int = 512, dtype_bytes: int = 4) -> float:
    if images < 0 or dimensions <= 0 or dtype_bytes <= 0:
        raise ValueError("invalid embedding dimensions")
    return images * dimensions * dtype_bytes / BYTES_PER_DECIMAL_GB


def storage_estimate(
    compressed_gb: float,
    unpacked_gb: float,
    preprocessed_gb: float,
    images: int,
    checkpoints_gb: float,
    logs_gb: float,
    headroom: float,
) -> dict[str, float]:
    items = {
        "compressed_gb": compressed_gb,
        "unpacked_gb": unpacked_gb,
        "preprocessed_gb": preprocessed_gb,
        "embeddings_gb": embedding_gb(images),
        "checkpoints_gb": checkpoints_gb,
        "logs_and_results_gb": logs_gb,
    }
    if any(value < 0 for value in items.values()):
        raise ValueError("storage values cannot be negative")
    if headroom < 1:
        raise ValueError("headroom must be >= 1")
    minimum = sum(items.values())
    recommended = max(compressed_gb * 3, minimum * headroom)
    return {**items, "minimum_gb": minimum, "recommended_gb": recommended}


def format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _command_output(command: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _ram_bytes() -> Optional[int]:
    if platform.system() == "Darwin":
        value = _command_output(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return int(pages * page_size)


def _accelerator() -> dict[str, object]:
    result: dict[str, object] = {
        "torch_version": None,
        "cuda_available": False,
        "mps_available": False,
        "gpu": None,
        "vram_bytes": None,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the host
        result["torch_error"] = str(exc)
        return result

    result["torch_version"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    result["mps_available"] = bool(mps and mps.is_available())
    if result["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = props.name
        result["vram_bytes"] = int(props.total_memory)
    elif result["mps_available"]:
        result["gpu"] = "Apple Metal Performance Shaders (unified memory)"
    return result


def environment_snapshot(path: Path) -> dict[str, object]:
    disk = shutil.disk_usage(path.resolve())
    cpu_model = None
    if platform.system() == "Darwin":
        cpu_model = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    if not cpu_model:
        cpu_model = platform.processor() or platform.machine()
    return {
        "status": "measured",
        "path": str(path.resolve()),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "internet_download_mbps": None,
        "internet_download_note": "not measured; run three wired tests and use median",
        "archive_compressed_bytes": None,
        "archive_unpacked_bytes": None,
        **_accelerator(),
    }


def _download_command(args: argparse.Namespace) -> dict[str, object]:
    rows = download_estimates(args.size_gb, args.speeds_mbps, args.efficiency)
    payload: dict[str, object] = {
        "status": "estimated",
        "size_gb": args.size_gb,
        "efficiency_assumption": args.efficiency,
        "downloads": [
            {
                **asdict(row),
                "theoretical_human": format_duration(row.theoretical_seconds),
                "expected_human": format_duration(row.expected_seconds),
            }
            for row in rows
        ],
    }
    if args.checksum_mbps:
        seconds = process_seconds(args.size_gb, args.checksum_mbps)
        payload["checksum"] = {
            "status": "estimated_from_user_throughput",
            "measured_throughput_MBps": args.checksum_mbps,
            "seconds": seconds,
            "human": format_duration(seconds),
        }
    if args.extract_mbps:
        seconds = process_seconds(args.size_gb, args.extract_mbps)
        payload["extract"] = {
            "status": "estimated_from_user_throughput",
            "measured_compressed_input_MBps": args.extract_mbps,
            "seconds": seconds,
            "human": format_duration(seconds),
            "note": "expanded-output write time may add to this estimate",
        }
    return payload


def _storage_command(args: argparse.Namespace) -> dict[str, object]:
    return {
        "status": "estimated",
        "assumptions": {
            "embedding_dimensions": 512,
            "embedding_dtype": "float32",
            "headroom_multiplier": args.headroom,
        },
        **storage_estimate(
            args.compressed_gb,
            args.unpacked_gb,
            args.preprocessed_gb,
            args.images,
            args.checkpoints_gb,
            args.logs_gb,
            args.headroom,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="estimate transfer durations")
    download.add_argument("--size-gb", type=float, default=100)
    download.add_argument(
        "--speeds-mbps", type=float, nargs="+", default=[50, 100, 500, 1_000]
    )
    download.add_argument("--efficiency", type=float, default=0.8)
    download.add_argument("--checksum-mbps", type=float)
    download.add_argument("--extract-mbps", type=float)
    download.set_defaults(handler=_download_command)

    storage = subparsers.add_parser("storage", help="estimate working storage")
    storage.add_argument("--compressed-gb", type=float, default=100)
    storage.add_argument("--unpacked-gb", type=float, default=150)
    storage.add_argument("--preprocessed-gb", type=float, default=30)
    storage.add_argument("--images", type=int, default=1_000_000)
    storage.add_argument("--checkpoints-gb", type=float, default=10)
    storage.add_argument("--logs-gb", type=float, default=5)
    storage.add_argument("--headroom", type=float, default=1.15)
    storage.set_defaults(handler=_storage_command)

    env = subparsers.add_parser("env", help="capture the current runtime inventory")
    env.add_argument("--path", type=Path, default=Path.cwd())
    env.set_defaults(handler=lambda args: environment_snapshot(args.path))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = args.handler(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
