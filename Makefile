.PHONY: dev lint

dev:
	uv run python -m agent.worker dev

lint:
	uv run ruff check .
	uv run ruff format --check .
