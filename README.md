# Job-Matching Search with Evaluation

An end-to-end, reproducible ML portfolio project building and comparing retrieval strategies (dense semantic search, lexical BM25, hybrid retrieval, and reciprocal rank fusion) for matching candidate profiles against job postings, accompanied by grounded LLM explanations and rigorous evaluation metrics.

---

## Phase 1A: Data and Indexing

### Dataset Provenance & Attribution
- **Dataset**: LinkedIn Job Postings (2023-2024)
- **Source URL**: [https://huggingface.co/datasets/datastax/linkedin_job_listings](https://huggingface.co/datasets/datastax/linkedin_job_listings)
- **Licence**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) / MIT
- **Integrity**: 100% real postings; no synthetic data generated.

### Measured Pipeline Statistics
*(All metrics generated automatically by `src/generate_readme.py` from `reports/data_stats.json`)*

| Stage | Metric | Count / Value |
|---|---|---|
| Raw Pool | Downloaded Postings | 10000 |
| Cleaned Pool | Valid Postings After Cleaning & Deduplication | 9252 |
| Sampled Output | Final Sampled Dataset (`seed=42`) | 3000 |
| Text Metric | Average Description Characters | 3588.4 chars |
| Text Metric | Average Description Words | 501.1 words |
| Text Metric | Median Words | 446 words |

### Experience Level Distribution

| Seniority Tier | Count | Percentage |
|---|---|---|
| Mid-Senior level | 1040 | 34.67% |
| Entry level | 1020 | 34.0% |
| Not Specified | 739 | 24.63% |
| Director | 101 | 3.37% |
| Internship | 50 | 1.67% |
| Executive | 50 | 1.67% |

### Measured Indexing Performance

| Metric | Measured Value |
|---|---|
| Vector Store | ChromaDB (HNSW Cosine) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension | 384 |
| Total Documents Indexed | 3000 |
| Indexing Time | 153.75 s |
| Indexing Throughput | 19.51 docs/sec |
| Metadata Fields | `job_id, title, company, location, experience_level, content_hash` |


---

## Project Structure

```text
job-search-project/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules (ignoring data/, chroma_db/, .venv/)
├── DECISIONS.md              # Architectural decisions, trade-offs, and justifications
├── Dockerfile                # Reproducible container configuration
├── Makefile                  # Build and execution targets
├── pyproject.toml            # Pytest configuration and project metadata
├── README.md                 # System overview and measured results table
├── requirements.txt          # Pinned dependency manifest
├── run_all.bat               # Windows one-command execution script
├── run_all.sh                # Linux/macOS one-command execution script
├── data/                     # Data directory (gitignored)
│   ├── raw_postings.csv      # Downloaded raw postings pool
│   └── jobs_clean.parquet    # Cleaned, deduplicated, sampled postings
├── reports/                  # Verified execution reports (JSON)
│   ├── data_stats.json       # Ingestion and data cleaning statistics
│   └── index_stats.json      # Vector indexing benchmark statistics
├── src/                      # Production implementation
│   ├── __init__.py
│   ├── generate_readme.py    # Auto-generation of results table from reports
│   ├── index.py              # ChromaDB vector indexing and embeddings
│   └── ingest.py             # Data loading, cleaning, deduplication, sampling
└── tests/                    # Comprehensive unit tests
    ├── test_index.py         # Embedding, hashing, and ChromaDB unit tests
    └── test_ingest.py        # Cleaning, HTML stripping, dedupe, empty row tests
```

---

## Quickstart & Reproducibility

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execute Pipeline
```bash
# Ingestion, cleaning, deduplication, and sampling
python src/ingest.py

# Dense vector indexing in ChromaDB
python src/index.py

# Run full unit test suite
pytest -v tests/

# Update README results table from reports
python src/generate_readme.py
```

### 3. One-Command Execution
- On Linux/macOS: `./run_all.sh` or `make all`
- On Windows: `run_all.bat`
- Via Docker:
  ```bash
  docker build -t job-search .
  docker run --rm job-search
  ```
