.PHONY: test preflight help

help:
	python -m rppg --help

test:
	python -m pytest -q

preflight:
	python scripts/validate_repository.py
	python -m compileall -q src
