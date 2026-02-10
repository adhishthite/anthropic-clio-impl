.PHONY: install format lint check test clean run

install:
	uv sync --all-groups

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint:
	uv run ruff check --fix src/ tests/

check: format lint

test:
	uv run pytest tests/ -v --tb=short

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage htmlcov
	find src tests -type d -name __pycache__ -exec rm -rf {} +
	find src tests -type d -name .pytest_cache -exec rm -rf {} +

run:
	uv run clio --help
