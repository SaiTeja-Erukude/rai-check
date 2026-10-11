"""
Unified CLI entry point for rai-check-kit.

Assembles the core commands and all available module subcommands
by loading registered plugins via entry points.
"""

from __future__ import annotations

import importlib.metadata
import json

import typer
from rai_audit.core.cli import app as core_app
from rai_audit.core.cli import export_app
from rai_audit.kit.doctor import collect_diagnostics, render_diagnostics

app = typer.Typer(
    name="rai-check",
    help="Responsible AI (RAI) Check Kit - security, reliability, and trustworthy AI checks.",
    no_args_is_help=True,
)

# Mount core commands directly on the kit app
for command in core_app.registered_commands:
    app.registered_commands.append(command)

# Mount export sub-app
app.add_typer(export_app, name="export")


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Print diagnostics as JSON."),
) -> None:
    """Diagnose installed packages, audit plugins, and optional dependencies."""
    diagnostics = collect_diagnostics(_plugin_results, _plugin_discovery_errors)
    output = json.dumps(diagnostics, indent=2) if json_output else render_diagnostics(diagnostics)
    typer.echo(output)
    if not diagnostics["ok"]:
        raise typer.Exit(1)


# Discover and mount plugins once; doctor reports these actual load results.
def _load_plugins(parent: typer.Typer) -> tuple[list[dict], list[str]]:
    results = []
    try:
        eps = importlib.metadata.entry_points(group="rai_audit.plugins")
    except Exception as exc:
        # Discovery failure should be visible to doctor, just like load failures.
        return results, [f"Plugin discovery failed: {type(exc).__name__}: {exc}"]
    for ep in eps:
        result = {
            "name": ep.name,
            "entry_point": ep.value,
            "status": "loaded",
            "error": None,
            "install_command": 'python -m pip install --upgrade "rai-check-kit[all]"',
        }
        try:
            distribution = getattr(ep, "dist", None)
            if distribution is not None:
                name = distribution.metadata["Name"]
                result["install_command"] = (
                    f'python -m pip install --upgrade --force-reinstall "{name}"'
                )
            register_fn = ep.load()
            register_fn(parent)
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            typer.echo(f"[warning] Could not load plugin '{ep.name}': {exc}", err=True)
        results.append(result)
    return results, []


_plugin_results, _plugin_discovery_errors = _load_plugins(app)
