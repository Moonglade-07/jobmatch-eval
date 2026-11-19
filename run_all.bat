@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Job-Matching Search: Full Pipeline Run
echo ==========================================

echo [1/3] Running ingestion and data cleaning...
python src\ingest.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [2/3] Building ChromaDB vector index...
python src\index.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [3/3] Running unit test suite...
pytest -v tests\
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo Pipeline execution finished successfully.

