@echo off
setlocal enabledelayedexpansion

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    set "PYTEST=.venv\Scripts\pytest.exe"
) else (
    set "PYTHON=python"
    set "PYTEST=pytest"
)

echo ==========================================
echo Job-Matching Search: Full Pipeline Run
echo ==========================================

echo [1/7] Running ingestion and data cleaning...
%PYTHON% src\ingest.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [2/7] Building ChromaDB vector index...
%PYTHON% src\index.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [3/7] Running hash-based incremental sync benchmark...
%PYTHON% src\bench_sync.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [4/7] Running Phase 2B retrieval evaluation...
%PYTHON% src\run_retrieval_eval.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [5/7] Running Phase 2C LLM explanation evaluation...
%PYTHON% src\run_explain_eval.py --full
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [6/7] Injecting results tables into README.md & generating RESULTS.md...
%PYTHON% src\make_readme_table.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [7/7] Running unit test suite...
%PYTEST% -v tests\
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo ==========================================
echo Pipeline execution finished successfully.
echo ==========================================
