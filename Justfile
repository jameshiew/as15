# Show available recipes
default:
    @just --list

# Sync the environment, including dev tooling
sync:
    uv sync

# Lint
lint:
    uv run ruff check .
    tombi lint

# Lint, applying autofixes
fix:
    uv run ruff check --fix .

# Format in place
fmt:
    uv run ruff format .
    tombi fmt

# Verify formatting without writing (what CI should run)
fmt-check:
    uv run ruff format --check .
    tombi fmt --check

# Type check
types:
    uv run ty check

# Unit tests
test:
    uv run pytest

# Every gate, in the order that gives the most useful first failure
check: lint fmt-check types test

# Format and autofix, then re-run every gate
tidy: fmt fix check
