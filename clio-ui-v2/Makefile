.PHONY: install format lint check test clean dev build

install:
	bun install

format:
	bunx biome format --write .

lint:
	bunx biome check --fix .

check: format lint

test:
	bun test --pass-with-no-tests

clean:
	rm -rf .next node_modules out build dist

dev:
	bun run dev

build:
	bun run build
