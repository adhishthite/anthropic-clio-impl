.PHONY: install format lint check test clean run ui-install ui-format ui-lint ui-check ui-test ui-clean ui-dev dev-ui ui-build

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

ui-install:
	$(MAKE) -C clio-ui-v2 install

ui-format:
	$(MAKE) -C clio-ui-v2 format

ui-lint:
	$(MAKE) -C clio-ui-v2 lint

ui-check:
	$(MAKE) -C clio-ui-v2 check

ui-test:
	$(MAKE) -C clio-ui-v2 test

ui-clean:
	$(MAKE) -C clio-ui-v2 clean

ui-dev:
	$(MAKE) -C clio-ui-v2 dev

dev-ui: ui-dev

ui-build:
	$(MAKE) -C clio-ui-v2 build
