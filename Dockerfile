FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache -e .

# Pre-download Silero VAD model so cold starts don't hang
RUN python -c "from livekit.plugins import silero; silero.VAD.load()"

COPY agent/ agent/

CMD ["python", "-m", "agent.worker", "start"]
