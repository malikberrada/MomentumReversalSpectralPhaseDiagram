.PHONY: install test build clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info src/mrspd.egg-info
