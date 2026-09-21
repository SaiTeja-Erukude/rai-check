import pytest
from jsonschema import Draft202012Validator
from rai_audit.core.schemas import (
    SCHEMA_VERSION,
    SCHEMAS,
    SchemaDocumentError,
    get_schema,
    prepare_document,
)


def test_registered_schemas_are_valid_json_schemas():
    for schema in SCHEMAS.values():
        Draft202012Validator.check_schema(schema)


def test_legacy_config_is_migrated_to_current_version():
    config = prepare_document("config", {"audit": {"type": "classification"}})

    assert config["schema_version"] == SCHEMA_VERSION


def test_legacy_suite_tests_key_is_migrated_to_cases():
    suite = prepare_document(
        "suite",
        {
            "name": "legacy",
            "tests": [{"id": "case-1", "type": "unsafe_output", "prompt": "Hello"}],
        },
    )

    assert suite["schema_version"] == SCHEMA_VERSION
    assert suite["cases"][0]["id"] == "case-1"
    assert "tests" not in suite


def test_future_schema_version_is_rejected():
    with pytest.raises(SchemaDocumentError, match="unsupported"):
        prepare_document("config", {"schema_version": "99.0", "audit": {}})


def test_schema_validation_reports_document_path():
    with pytest.raises(SchemaDocumentError, match="report.project_name"):
        prepare_document(
            "report",
            {
                "schema_version": SCHEMA_VERSION,
                "project_name": "",
                "audit_type": "classification",
                "findings": [],
                "risk_matrix": [],
                "metadata": {},
            },
        )


def test_get_schema_returns_copy():
    schema = get_schema("config")
    schema["properties"].clear()

    assert get_schema("config")["properties"]
