PYTHON ?= python

.PHONY: test lint format-check check format

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

check: lint format-check test

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
