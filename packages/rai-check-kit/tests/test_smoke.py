"""Smoke tests for rai-check-kit unified CLI."""
from typer.testing import CliRunner

from rai_audit.kit.cli import app

_runner = CliRunner()


def test_kit_help():
    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_core_commands_present():
    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output or "report" in result.output


def test_export_subcommand_present():
    result = _runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "model-card" in result.output
