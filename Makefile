.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check security-check clean docker-build docker-run docs run dev setup ci deploy

# Default target
help:
	@echo "Lya - Autonomous AGI Agent"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup         - Initial project setup"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make dev           - Run in development mode"
	@echo "  make run           - Run the agent"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-e2e      - Run end-to-end tests"
	@echo "  make lint          - Run linter (ruff)"
	@echo "  make format        - Format code (ruff)"
	@echo "  make type-check    - Run type checker (mypy)"
	@echo "  make security-check - Run security checks"
	@echo "  make ci            - Run all CI checks"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make deploy        - Deploy to Linux VPS (single command)"
	@echo "  make bot           - Run Telegram bot directly"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run in Docker (detached)"
	@echo "  make docker-logs   - View Docker logs"
	@echo "  make docs          - Build documentation"
	@echo ""

# ═══════════════════════════════════════════════════════════════
# Setup & Installation
# ═══════════════════════════════════════════════════════════════

setup:
	@echo "🚀 Setting up Lya development environment..."
	python -m pip install --upgrade pip
	pip install poetry
	poetry config virtualenvs.in-project true
	poetry install --with dev --extras "all"
	cp .env.example .env
	@echo "✅ Setup complete! Edit .env file with your configuration."

install:
	poetry install --no-dev

install-dev:
	poetry install --with dev --extras "all"
	poetry run pre-commit install

# ═══════════════════════════════════════════════════════════════
# Development
# ═══════════════════════════════════════════════════════════════

dev:
	poetry run python -m lya.adapters.cli.main --reload

run:
	poetry run lya run --autonomous

chat:
	poetry run lya chat

api:
	poetry run python -m lya.adapters.api.server

# ═══════════════════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════════════════

test:
	poetry run pytest --cov=src/lya --cov-report=term-missing --cov-report=html

test-unit:
	poetry run pytest tests/unit -v -m unit

test-integration:
	poetry run pytest tests/integration -v -m integration

test-e2e:
	poetry run pytest tests/e2e -v -m e2e

test-fast:
	poetry run pytest -x --timeout=300

test-coverage:
	poetry run pytest --cov=src/lya --cov-report=html --cov-report=xml
	@echo "📊 Coverage report: htmlcov/index.html"

# ═══════════════════════════════════════════════════════════════
# Code Quality
# ═══════════════════════════════════════════════════════════════

lint:
	@echo "🔍 Running linter..."
	poetry run ruff check src tests
	poetry run ruff check --select I src tests

lint-fix:
	@echo "🔧 Fixing lint issues..."
	poetry run ruff check --fix src tests
	poetry run ruff check --select I --fix src tests

format:
	@echo "🎨 Formatting code..."
	poetry run ruff format src tests

type-check:
	@echo "🔍 Running type checker..."
	poetry run mypy src

security-check:
	@echo "🔒 Running security checks..."
	poetry run bandit -r src
	poetry run safety check

pre-commit:
	poetry run pre-commit run --all-files

ci: lint type-check test
	@echo "✅ All CI checks passed!"

# ═══════════════════════════════════════════════════════════════
# Docker
# ═══════════════════════════════════════════════════════════════

deploy:
	@echo "Deploying Lya..."
	bash deploy.sh

bot:
	python run_lya.py

docker-build:
	docker compose -f .docker/docker-compose.yml build

docker-run:
	docker compose -f .docker/docker-compose.yml up -d

docker-dev:
	docker compose -f .docker/docker-compose.yml up

docker-down:
	docker compose -f .docker/docker-compose.yml down

docker-logs:
	docker compose -f .docker/docker-compose.yml logs -f

# ═══════════════════════════════════════════════════════════════
# Documentation
# ═══════════════════════════════════════════════════════════════

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Clean complete!"

update-deps:
	poetry update
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes

seed-data:
	poetry run python scripts/seed_data.py

migrate:
	poetry run python scripts/migrate.py

# ═══════════════════════════════════════════════════════════════
# Release
# ═══════════════════════════════════════════════════════════════

version-patch:
	poetry version patch

version-minor:
	poetry version minor

version-major:
	poetry version major

publish:
	poetry build
	poetry publish
