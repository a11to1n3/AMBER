.PHONY: help install test test-fast test-slow test-coverage clean lint format type-check docs

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
	pytest --cov=src/amber --cov-report=html --cov-report=term-missing

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
	rm -rf coverage.xml
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

lint:  ## Run linting checks
	flake8 src tests
	black --check src tests

format:  ## Format code
	black src tests

type-check:  ## Run type checking
	mypy src

docs:  ## Generate documentation (if applicable)
	@echo "Documentation generation not yet implemented"

check-all: lint type-check test  ## Run all checks (lint, type-check, test)

dev-install:  ## Install in development mode with all dependencies
	pip install -e ".[dev]" 