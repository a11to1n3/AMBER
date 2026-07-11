.PHONY: help install test test-fast test-slow test-coverage clean lint format type-check pre-commit-install pre-commit-run docs package check-dist release-check

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install -e .
	pip install -r requirements.txt

test:  ## Run all tests
	pytest

test-fast:  ## Run only fast tests (exclude slow tests)
	pytest -m "not slow"

test-slow:  ## Run only slow tests
	pytest -m "slow"

test-coverage:  ## Run tests with coverage report
	pytest --cov=src/ambr --cov-report=html --cov-report=term-missing

test-unit:  ## Run only unit tests
	pytest -m "unit"

test-integration:  ## Run only integration tests
	pytest -m "integration"

test-verbose:  ## Run tests with verbose output
	pytest -v

test-debug:  ## Run tests with debug output
	pytest -vvv -s

clean:  ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf .!*.coverage
	rm -rf coverage.xml
	rm -rf dist
	rm -rf build
	rm -rf src/*.egg-info
	rm -rf *.egg-info
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

lint:  ## Run linting checks (ruff primary; black/flake8 optional)
	ruff check src/ambr
	@command -v black >/dev/null && black --check src tests || true

format:  ## Format code
	ruff check src/ambr --fix
	@command -v black >/dev/null && black src tests || true

type-check:  ## Run type checking (gradual module set in pyproject.toml)
	mypy

pre-commit-install:  ## Install local git hooks (nbstripout, ruff)
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

docs:  ## Build Sphinx HTML docs into docs/_build/html
	python -m sphinx -b html docs docs/_build/html

package: clean  ## Build source and wheel distributions
	python -m build

check-dist: package  ## Validate built distributions
	python -m twine check dist/*

release-check: check-dist test  ## Validate package artifacts and tests before tagging

check-all: lint type-check test  ## Run all checks (lint, type-check, test)

dev-install:  ## Install in development mode with all dependencies
	pip install -e ".[dev]"
