PYTHON ?= python

.PHONY: test opd-test lint format-check check format

test:
	$(PYTHON) -m pytest

opd-test:
	$(PYTHON) -m pytest \
		tests/test_token_shift.py \
		tests/test_response_mask.py \
		tests/test_sampled_opd_reward.py \
		tests/test_teacher_no_grad.py \
		tests/test_overlap_metric.py \
		tests/test_entropy_metric.py \
		tests/test_weight_sync.py

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

check: lint format-check test

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
