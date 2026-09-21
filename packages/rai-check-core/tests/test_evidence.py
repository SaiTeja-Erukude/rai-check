import json

import pytest
from rai_audit.core.evidence import load_evidence_manifest, write_evidence_manifest
from rai_audit.core.schemas import SchemaDocumentError


def _manifest() -> dict:
    return {
        "audit_id": "audit-1",
        "generated_at": "2026-05-31T00:00:00+00:00",
        "project_name": "demo",
        "audit_type": "classification",
        "runtime": {},
        "library_versions": {},
        "inputs": {},
        "artifacts": {},
    }


def test_evidence_manifest_is_migrated_when_written_and_loaded(tmp_path):
    path = tmp_path / "manifest.json"

    write_evidence_manifest(_manifest(), path)
    loaded = load_evidence_manifest(path)

    assert loaded["schema_version"] == "1.0"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def test_evidence_manifest_rejects_unknown_version(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {**_manifest(), "schema_version": "99.0"}
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SchemaDocumentError, match="unsupported"):
        load_evidence_manifest(path)
