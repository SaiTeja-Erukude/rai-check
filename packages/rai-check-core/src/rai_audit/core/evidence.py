from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from rai_audit.core.findings import AuditReport
from rai_audit.core.schemas import SCHEMA_VERSION, prepare_document

EVIDENCE_SCHEMA_VERSION = SCHEMA_VERSION
_DEFAULT_PACKAGES = (
    "rai-check-core",
    "rai-check-ml",
    "rai-check-dl",
    "rai-check-genai",
    "rai-check-kit",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def collect_library_versions(packages: tuple[str, ...] = _DEFAULT_PACKAGES) -> dict[str, str]:
    """Capture installed package versions for reproducible evidence bundles."""
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def current_git_sha(directory: str | Path | None = None) -> str | None:
    """Return the current Git commit when the audit runs inside a worktree."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def build_evidence_manifest(
    report: AuditReport,
    *,
    config: dict[str, Any] | None = None,
    inputs: dict[str, str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
    working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Build a portable manifest describing an audit run and its input evidence."""
    input_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted((inputs or {}).items())
    }
    supplied_metadata = metadata or {}
    library_versions = supplied_metadata.get("library_versions") or collect_library_versions()
    manifest = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "audit_id": str(uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": report.project_name,
        "audit_type": report.audit_type,
        "git_sha": supplied_metadata.get("git_sha") or current_git_sha(working_directory),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "library_versions": library_versions,
        "inputs": input_hashes,
        "config_sha256": sha256_json(config) if config is not None else None,
        "artifacts": {},
    }
    if supplied_metadata.get("model_hash"):
        manifest["model_hash"] = supplied_metadata["model_hash"]
    return prepare_document("evidence_manifest", manifest)


def finalize_evidence_manifest(
    manifest: dict[str, Any],
    artifacts: dict[str, str | Path],
) -> dict[str, Any]:
    """Add generated artifact hashes to a manifest."""
    manifest = {**manifest}
    manifest["artifacts"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(artifacts.items())
    }
    return prepare_document("evidence_manifest", manifest)


def write_evidence_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    validated = prepare_document("evidence_manifest", manifest)
    Path(path).write_text(json.dumps(validated, indent=2), encoding="utf-8")


def load_evidence_manifest(path: str | Path) -> dict[str, Any]:
    """Load, migrate, and validate an evidence manifest."""
    return prepare_document(
        "evidence_manifest",
        json.loads(Path(path).read_text(encoding="utf-8")),
    )


def reproducibility_metadata(
    manifest: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate captured manifest evidence into reproducibility check inputs."""
    supplied_metadata = metadata or {}
    input_hashes = {
        name: value["sha256"] for name, value in manifest.get("inputs", {}).items()
    }
    return {
        **supplied_metadata,
        "python_version": sys.version,
        "library_versions": manifest.get("library_versions", {}),
        "data_hash": supplied_metadata.get("data_hash") or sha256_json(input_hashes),
        "model_hash": supplied_metadata.get("model_hash") or manifest.get("model_hash"),
        "git_sha": supplied_metadata.get("git_sha") or manifest.get("git_sha"),
    }
