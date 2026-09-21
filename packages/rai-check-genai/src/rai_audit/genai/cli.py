from __future__ import annotations

import typer

from rai_audit.genai.agents.cli import agents_app
from rai_audit.genai.llm.cli import llm_app

genai_app = typer.Typer(
    name="genai",
    help="Audit generative AI applications, including LLM, RAG, and agentic systems.",
    no_args_is_help=True,
)
genai_app.add_typer(llm_app)
genai_app.add_typer(agents_app)


def register(parent: typer.Typer) -> None:
    parent.add_typer(genai_app)
