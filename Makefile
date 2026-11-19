.PHONY: all setup ingest index bench demo test clean help

PYTHON := python
PYTEST := pytest

help:
	@echo "Available targets:"
	@echo "  make setup    - Install dependencies from requirements.txt"
	@echo "  make ingest   - Run data cleaning, deduplication, and sampling"
	@echo "  make index    - Build ChromaDB vector index with SentenceTransformer"
	@echo "  make bench    - Run hash-based sync benchmark and equivalence check"
	@echo "  make demo     - Run side-by-side retrieval comparison demo"
	@echo "  make test     - Execute full unit test suite"
	@echo "  make all      - Run ingest, index, and tests sequentially"
	@echo "  make clean    - Remove build and cache artifacts"

setup:
	pip install -r requirements.txt

ingest:
	$(PYTHON) src/ingest.py

index:
	$(PYTHON) src/index.py

bench:
	$(PYTHON) src/bench_sync.py

demo:
	$(PYTHON) src/demo_comparison.py

test:
	$(PYTEST) -v tests/

all: ingest index test

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__

