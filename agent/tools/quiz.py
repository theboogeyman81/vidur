import os

import google.genai as genai
from pydantic_ai import RunContext


async def quiz_tool(ctx: RunContext[dict], topic: str) -> str:
    """Generate one Socratic UPSC question on the given topic."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = await client.aio.models.generate_content(
        model="gemini-3.6-flash",
        contents=(
            f"Generate one Socratic question for UPSC preparation on: {topic}. "
            "Output only the question, no preamble."
        ),
    )
    return resp.text.strip()
