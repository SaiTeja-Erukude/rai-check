"""
Unified CLI entry point for rai-check-kit.

Assembles the core commands and all available module subcommands
by loading registered plugins via entry points.
"""
from __future__ import annotations

import importlib.metadata

import typer

from rai_audit.core.cli import app as core_app
from rai_audit.core.cli import export_app

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

# Discover and mount plugin subcommands registered via entry points
def _load_plugins() -> None:
    try:
        eps = importlib.metadata.entry_points(group="rai_audit.plugins")
    except Exception:
        return
    for ep in eps:
        try:
            register_fn = ep.load()
            register_fn(app)
        except Exception as exc:
            typer.echo(f"[warning] Could not load plugin '{ep.name}': {exc}", err=True)


_load_plugins()
