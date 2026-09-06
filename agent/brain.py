from __future__ import annotations

import os
import pathlib

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent.tools.progress import progress_tool
from agent.tools.quiz import quiz_tool
from agent.tools.retrieve import retrieve_tool

_SYSTEM_PROMPT = (pathlib.Path(__file__).parent / "prompts" / "tutor_v1.md").read_text()


def _build_agent() -> Agent:
    provider = GoogleProvider(api_key=os.environ["GEMINI_API_KEY"])
    model = GoogleModel("gemini-3.6-flash", provider=provider)
    return Agent(
        model,
        deps_type=dict,
        system_prompt=_SYSTEM_PROMPT,
        tools=[retrieve_tool, quiz_tool, progress_tool],
    )


vidur_agent: Agent = _build_agent()
