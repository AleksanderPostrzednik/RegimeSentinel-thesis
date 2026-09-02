"""Deterministic artifact and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return sha256_file(path)


def git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def runtime_versions(*, rscript: str = "Rscript") -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "scipy", "pandas", "arch", "statsmodels"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    try:
        completed = subprocess.run(
            [rscript, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        versions["Rscript"] = (completed.stdout or completed.stderr).strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        versions["Rscript"] = "unavailable"

    probes = (
        ("MSGARCH", 'cat(as.character(packageVersion("MSGARCH")))'),
        ("R_platform", 'cat(R.version$platform)'),
    )
    for key, expression in probes:
        try:
            completed = subprocess.run(
                [rscript, "-e", expression],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            versions[key] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            versions[key] = "unavailable"
    return versions


def started_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def base_manifest(
    *,
    protocol_id: str,
    protocol_sha256: str,
    snapshot_id: str,
    snapshot_file_sha256: str,
    repo_root: Path,
    seed: int,
    started: str,
    rscript: str = "Rscript",
) -> dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "snapshot_id": snapshot_id,
        "snapshot_file_sha256": snapshot_file_sha256,
        "git_commit": git_commit(repo_root),
        "runtime_versions": runtime_versions(rscript=rscript),
        "seed": seed,
        "started_at_utc": started,
    }
