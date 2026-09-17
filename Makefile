PYTHON ?= python

.PHONY: check integrity compile test status

check: integrity compile
	@git diff --check

integrity:
	$(PYTHON) scripts/ci_repo_integrity.py

compile:
	$(PYTHON) -m compileall -q src tests

test:
	$(PYTHON) -m pytest -q

status:
	@git status --short
