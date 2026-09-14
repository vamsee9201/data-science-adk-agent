.PHONY: dev test

dev:
	uv run uvicorn app.main:app --reload --port 8765

test:
	uv run pytest
