#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Job-Matching Search: Full Pipeline Run"
echo "=========================================="

echo "[1/3] Running ingestion and data cleaning..."
python src/ingest.py

echo "[2/3] Building ChromaDB vector index..."
python src/index.py

echo "[3/4] Running retrieval strategy side-by-side demo..."
python src/demo_comparison.py

echo "[4/4] Running unit test suite..."
pytest -v tests/

echo "Pipeline execution finished successfully."

