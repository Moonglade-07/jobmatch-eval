.PHONY: all setup ingest index bench retrieval-eval llm-eval make-table demo test clean help

PYTHON := python
PYTEST := pytest

help:
	@echo "Available targets:"
	@echo "  make setup          - Install dependencies from requirements.txt"
	@echo "  make ingest         - Run data cleaning, deduplication, and sampling"
	@echo "  make index          - Build ChromaDB vector index with SentenceTransformer"
	@echo "  make bench          - Run hash-based sync benchmark and equivalence check"
	@echo "  make retrieval-eval - Run Phase 2B full retrieval evaluation"
	@echo "  make llm-eval       - Run Phase 2C LLM explanation evaluation (cached/live)"
	@echo "  make make-table     - Inject results tables into README.md & update RESULTS.md"
	@echo "  make demo           - Run side-by-side retrieval comparison demo"
	@echo "  make test           - Execute full unit test suite"
	@echo "  make all            - Run ingest -> index -> sync benchmark -> retrieval eval -> LLM eval -> make README table -> test"
	@echo "  make clean          - Remove build and cache artifacts"

setup:
	pip install -r requirements.txt

ingest:
	$(PYTHON) src/ingest.py

index:
	$(PYTHON) src/index.py

bench:
	$(PYTHON) src/bench_sync.py

retrieval-eval:
	$(PYTHON) src/run_retrieval_eval.py

llm-eval:
	$(PYTHON) src/run_explain_eval.py --full

make-table:
	$(PYTHON) src/make_readme_table.py

demo:
	$(PYTHON) src/demo_comparison.py

test:
	$(PYTEST) -v tests/

all: ingest index bench retrieval-eval llm-eval make-table test

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
