"""Diagnostics stay usable with optional packages missing or plugins broken."""

import importlib.metadata
import json
from types import SimpleNamespace

import pytest
import typer
from rai_audit.kit import cli, doctor
from typer.testing import CliRunner

_BASE_PACKAGES = {"rai-check-core": "1.0", "rai-check-ml": "1.1", "rai-check-kit": "1.2"}
_runner = CliRunner()


def _versions(monkeypatch, installed):
    def version(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(doctor.importlib.metadata, "version", version)


def _cli_state(monkeypatch, plugins=None, errors=None):
    monkeypatch.setattr(cli, "_plugin_results", plugins or [])
    monkeypatch.setattr(cli, "_plugin_discovery_errors", errors or [])


def test_base_install_json_reports_missing_optional_features(monkeypatch):
    _versions(monkeypatch, _BASE_PACKAGES)
    _cli_state(monkeypatch)
    result = _runner.invoke(cli.app, ["doctor", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["ok"]
    packages = {item["name"]: item for item in report["packages"]}
    assert packages["rai-check-ml"]["version"] == "1.1"
    assert packages["rai-check-dl"]["status"] == "not_installed"
    assert not packages["rai-check-dl"]["required"]
    assert '"rai-check-kit[dl]"' in packages["rai-check-dl"]["install_command"]
    assert [item["extra"] for item in report["optional_dependencies"]] == ["explainability"]
    shap = report["optional_dependencies"][0]["dependencies"][0]
    assert shap["name"] == "shap"
    assert shap["status"] == "not_installed"
    assert '"rai-check-ml[explainability]"' in shap["install_command"]


def test_text_output_has_copyable_installation_guidance(monkeypatch):
    _versions(monkeypatch, _BASE_PACKAGES)
    _cli_state(monkeypatch)
    result = _runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "rai-check-kit: installed (1.2)" in result.stdout
    assert 'python -m pip install "rai-check-kit[genai]"' in result.stdout
    assert 'python -m pip install "rai-check-ml[explainability]"' in result.stdout
    assert "Status: OK" in result.stdout


def test_partially_installed_extra_identifies_missing_dependency(monkeypatch):
    _versions(monkeypatch, {**_BASE_PACKAGES, "rai-check-dl": "1.3", "torch": "2.0"})
    report = doctor.collect_diagnostics([], [])
    extra = next(item for item in report["optional_dependencies"] if item["extra"] == "torch")
    assert [(dep["name"], dep["status"]) for dep in extra["dependencies"]] == [
        ("torch", "installed"),
        ("torchvision", "not_installed"),
    ]
    assert report["ok"]


def test_all_optional_dependencies_installed(monkeypatch):
    installed = {
        **_BASE_PACKAGES,
        "rai-check-dl": "1.0",
        "rai-check-genai": "1.0",
        "shap": "0.44",
        "torch": "2.0",
        "torchvision": "0.15",
        "tensorflow": "2.13",
        "openai": "1.0",
        "anthropic": "0.20",
    }
    _versions(monkeypatch, installed)
    report = doctor.collect_diagnostics([], [])
    assert report["ok"]
    assert len(report["optional_dependencies"]) == 5
    assert all(
        dep["status"] == "installed" and dep["version"] == installed[dep["name"]]
        for extra in report["optional_dependencies"]
        for dep in extra["dependencies"]
    )


def test_broken_plugin_does_not_block_doctor_or_other_plugins(monkeypatch):
    _versions(monkeypatch, _BASE_PACKAGES)

    def broken_load():
        raise ImportError("missing backend")

    def register(parent):
        @parent.command()
        def working():
            typer.echo("healthy plugin still works")

    eps = [
        SimpleNamespace(
            name="broken",
            value="broken.cli:register",
            load=broken_load,
            dist=SimpleNamespace(metadata={"Name": "third-party-audit"}),
        ),
        SimpleNamespace(name="working", value="working.cli:register", load=lambda: register),
    ]
    monkeypatch.setattr(cli.importlib.metadata, "entry_points", lambda **kwargs: eps)
    parent = typer.Typer()
    plugins, errors = cli._load_plugins(parent)
    _cli_state(monkeypatch, plugins, errors)
    parent.command()(cli.doctor)
    result = _runner.invoke(parent, ["doctor", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert [plugin["status"] for plugin in report["plugins"]] == ["error", "loaded"]
    assert "ImportError: missing backend" in report["plugins"][0]["error"]
    assert '"third-party-audit"' in report["plugins"][0]["install_command"]
    result = _runner.invoke(parent, ["working"])
    assert result.exit_code == 0
    assert "healthy plugin still works" in result.stdout


def test_registration_failure_is_recorded(monkeypatch):
    def register(parent):
        raise RuntimeError("registration failed")

    ep = SimpleNamespace(name="broken", value="broken:register", load=lambda: register)
    monkeypatch.setattr(cli.importlib.metadata, "entry_points", lambda **kwargs: [ep])
    plugins, errors = cli._load_plugins(typer.Typer())
    assert not errors
    assert plugins[0]["error"] == "RuntimeError: registration failed"


def test_discovery_failure_is_visible_in_json(monkeypatch):
    _versions(monkeypatch, _BASE_PACKAGES)

    def fail(**kwargs):
        raise RuntimeError("invalid entry point metadata")

    monkeypatch.setattr(cli.importlib.metadata, "entry_points", fail)
    plugins, errors = cli._load_plugins(typer.Typer())
    _cli_state(monkeypatch, plugins, errors)
    result = _runner.invoke(cli.app, ["doctor", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert not report["plugins"]
    assert "Plugin discovery failed" in report["errors"][0]
    assert "invalid entry point metadata" in report["errors"][0]


def test_missing_required_package_fails_with_diagnostics(monkeypatch):
    _versions(monkeypatch, {"rai-check-core": "1.0", "rai-check-kit": "1.2"})
    _cli_state(monkeypatch)
    result = _runner.invoke(cli.app, ["doctor", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert "Required package rai-check-ml is not installed." in report["errors"]


@pytest.mark.parametrize("failed_package", ["rai-check-dl", "shap"])
def test_bad_distribution_metadata_does_not_hide_remaining_results(monkeypatch, failed_package):
    _versions(monkeypatch, _BASE_PACKAGES)
    original_version = doctor.importlib.metadata.version

    def version(name):
        if name == failed_package:
            raise ValueError("corrupt metadata")
        return original_version(name)

    monkeypatch.setattr(doctor.importlib.metadata, "version", version)
    report = doctor.collect_diagnostics([], [])
    assert not report["ok"]
    assert any(
        failed_package in error and "corrupt metadata" in error for error in report["errors"]
    )
    assert len(report["packages"]) == 5
