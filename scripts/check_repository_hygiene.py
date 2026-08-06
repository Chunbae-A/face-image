#!/usr/bin/env python3
"""Reject sensitive or non-reproducible artifacts from tracked repository files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable, Sequence


DEFAULT_MAX_FILE_BYTES = 5_000_000
TEXT_SCAN_LIMIT_BYTES = 2_000_000

FORBIDDEN_SUFFIXES = (
    ".avi",
    ".ckpt",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pt",
    ".pth",
    ".tar",
    ".tar.gz",
    ".tflite",
    ".zip",
)

FORBIDDEN_PATH_PREFIXES = (
    "data/aligned/",
    "data/embeddings/",
    "data/interim/",
    "data/raw/",
    "data/rejected/",
    "outputs/",
)

SECRET_PATTERNS = (
    (
        "GitHub token",
        re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    ),
    (
        "private key",
        re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    ),
    (
        "AI-Hub API key assignment",
        re.compile(rb"AIHUB_API_KEY\s*=\s*['\"]?[A-Za-z0-9._-]{12,}"),
    ),
)


@dataclass(frozen=True)
class Violation:
    path: str
    rule: str
    detail: str

    def render(self) -> str:
        return f"{self.path}: [{self.rule}] {self.detail}"


def tracked_files(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def relative_name(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def check_notebook(path: Path, relative: str) -> list[Violation]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [Violation(relative, "notebook-json", str(exc))]

    violations: list[Violation] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            violations.append(
                Violation(relative, "notebook-output", f"code cell {index} contains outputs")
            )
        if cell.get("execution_count") is not None:
            violations.append(
                Violation(
                    relative,
                    "notebook-execution-count",
                    f"code cell {index} has execution_count={cell['execution_count']}",
                )
            )
    return violations


def check_paths(
    root: Path,
    paths: Iterable[Path],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[Violation]:
    root = root.resolve()
    violations: list[Violation] = []

    for path in paths:
        path = path.resolve()
        if not path.is_file():
            continue
        relative = relative_name(path, root)
        lowered = relative.casefold()

        if lowered.endswith(FORBIDDEN_SUFFIXES):
            violations.append(
                Violation(relative, "forbidden-artifact", "file extension is not allowed in Git")
            )

        if relative.startswith(FORBIDDEN_PATH_PREFIXES) and path.name != ".gitkeep":
            violations.append(
                Violation(relative, "forbidden-data-path", "sensitive data path is tracked")
            )

        size = path.stat().st_size
        if size > max_file_bytes:
            violations.append(
                Violation(
                    relative,
                    "large-file",
                    f"{size} bytes exceeds the {max_file_bytes}-byte limit",
                )
            )

        if path.suffix.casefold() == ".ipynb":
            violations.extend(check_notebook(path, relative))

        if size <= TEXT_SCAN_LIMIT_BYTES:
            payload = path.read_bytes()
            for name, pattern in SECRET_PATTERNS:
                if pattern.search(payload):
                    violations.append(Violation(relative, "secret-pattern", name))

    return sorted(violations, key=lambda item: (item.path, item.rule, item.detail))


def check_repository(root: Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> list[Violation]:
    root = root.resolve()
    return check_paths(root, tracked_files(root), max_file_bytes=max_file_bytes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    violations = check_repository(args.root, max_file_bytes=args.max_file_bytes)
    if violations:
        print("Repository hygiene check failed:")
        for violation in violations:
            print(f"- {violation.render()}")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
