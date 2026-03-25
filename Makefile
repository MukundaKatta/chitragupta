.PHONY: test lint clean install

PYTHONPATH := src

test:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest tests/ -v --tb=short

test-quick:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest tests/ -q

lint:
	python3 -m py_compile src/chitragupta/core.py
	python3 -m py_compile src/chitragupta/chunker.py
	python3 -m py_compile src/chitragupta/embedder.py
	python3 -m py_compile src/chitragupta/search.py

install:
	pip install -e .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache build dist *.egg-info
