import typer
from rai_audit.genai.cli import register
from typer.testing import CliRunner


def test_genai_cli_groups_llm_and_agent_commands():
    app = typer.Typer()
    register(app)

    result = CliRunner().invoke(app, ["genai", "--help"])

    assert result.exit_code == 0, result.output
    assert "llm" in result.output
    assert "agents" in result.output
