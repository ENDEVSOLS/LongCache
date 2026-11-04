.PHONY: lint format typecheck test test-cov build clean install docs-serve docs-build

DIST_DIR ?= dist

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src/longcache/ --strict

test:
	pytest -v --tb=short

test-cov:
	pytest --cov=src/longcache --cov-report=term-missing

build:
	python -m build --outdir $(DIST_DIR)
	twine check --strict $(DIST_DIR)/*
	python scripts/check_package_metadata.py --dist-dir $(DIST_DIR)

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage site/
	find . -type d -name __pycache__ -exec rm -rf {} +

install:
	pip install -e ".[all,dev,docs]"

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build --strict
