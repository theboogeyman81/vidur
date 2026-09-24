.PHONY: dev lint test

dev:
	uv run python -m agent.worker dev

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest tests -q
