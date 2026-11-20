#!/usr/bin/env bash
set -euo pipefail

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    PYTEST=".venv/bin/pytest"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
    PYTEST=".venv/Scripts/pytest.exe"
else
    PYTHON="python"
    PYTEST="pytest"
fi

echo "=========================================="
echo "Job-Matching Search: Full Pipeline Run"
echo "=========================================="

echo "[1/7] Running ingestion and data cleaning..."
$PYTHON src/ingest.py

echo "[2/7] Building ChromaDB vector index..."
$PYTHON src/index.py

echo "[3/7] Running hash-based incremental sync benchmark..."
$PYTHON src/bench_sync.py

echo "[4/7] Running Phase 2B retrieval evaluation..."
$PYTHON src/run_retrieval_eval.py

echo "[5/7] Running Phase 2C LLM explanation evaluation..."
$PYTHON src/run_explain_eval.py --full

echo "[6/7] Injecting results tables into README.md & generating RESULTS.md..."
$PYTHON src/make_readme_table.py

echo "[7/7] Running unit test suite..."
$PYTEST -v tests/

echo "=========================================="
echo "Pipeline execution finished successfully."
echo "=========================================="
