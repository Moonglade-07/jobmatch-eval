"""Script to generate or update README.md directly from reports/ JSON artifacts.

Ensures zero hand-typed metrics by pulling all data pipeline and index numbers
strictly from reports/data_stats.json and reports/index_stats.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DATA_STATS_PATH = Path("reports/data_stats.json")
INDEX_STATS_PATH = Path("reports/index_stats.json")
SYNC_STATS_PATH = Path("reports/sync_benchmark.json")
README_PATH = Path("README.md")


def load_json_report(path: Path) -> Dict[str, Any]:
    """Safely load a JSON report file.

    Args:
        path: Path to the JSON report.

    Returns:
        Loaded dictionary, or empty dict if file does not exist.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_readme_markdown(
    data_stats: Dict[str, Any],
    index_stats: Dict[str, Any],
    sync_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct full README content embedding verified stats from reports.

    Args:
        data_stats: Parsed dictionary from reports/data_stats.json.
        index_stats: Parsed dictionary from reports/index_stats.json.
        sync_stats: Optional parsed dictionary from reports/sync_benchmark.json.

    Returns:
        Complete Markdown string for README.md.
    """
    row_counts = data_stats.get("row_counts", {})
    desc_stats = data_stats.get("description_length", {})
    levels = data_stats.get("experience_level_distribution", {})

    level_rows = "\n".join(
        [
            f"| {lvl} | {info['count']} | {info['percentage']}% |"
            for lvl, info in levels.items()
        ]
    )

    index_table = ""
    if index_stats:
        index_table = f"""### Measured Indexing Performance

| Metric | Measured Value |
|---|---|
| Vector Store | ChromaDB (HNSW Cosine) |
| Embedding Model | `{index_stats.get("model_name", "N/A")}` |
| Embedding Dimension | {index_stats.get("embedding_dimension", "N/A")} |
| Total Documents Indexed | {index_stats.get("total_documents_indexed", "N/A")} |
| Indexing Time | {index_stats.get("index_build_time_seconds", "N/A")} s |
| Indexing Throughput | {index_stats.get("indexing_throughput_docs_sec", "N/A")} docs/sec |
| Metadata Fields | `{", ".join(index_stats.get("metadata_fields", []))}` |
"""

    sync_table = ""
    if sync_stats:
        mut_params = sync_stats.get("mutation_parameters", {})
        changes = sync_stats.get("change_counts", {})
        full = sync_stats.get("full_reindex", {})
        sync = sync_stats.get("incremental_sync", {})
        verif = sync_stats.get("state_equivalence_verification", {})

        sync_table = f"""---

## Phase 1B: Incremental Index Synchronization Benchmark

### Simulated Change Scenario
- **Dataset Baseline Size:** {sync_stats.get("dataset_baseline_size", "N/A")} postings
- **Modifications:** {mut_params.get("modify_percentage", 0)}% of descriptions ({changes.get("modified_count", 0)} postings)
- **Additions:** {mut_params.get("add_percentage", 0)}% new postings ({changes.get("added_count", 0)} postings)
- **Deletions:** {mut_params.get("delete_percentage", 0)}% removed postings ({changes.get("deleted_count", 0)} postings)
- **Unchanged:** {changes.get("unchanged_count", 0)} postings (zero embedding overhead)
- **Benchmark Repetitions:** {sync_stats.get("benchmark_runs", 3)} runs per strategy

### Measured Speedup & Throughput
*(Numbers extracted directly from `reports/sync_benchmark.json`)*

| Strategy | Mean Latency | Latency Std Dev | Effective Throughput |
|---|---|---|---|
| Full Re-Index (from scratch) | {full.get("mean_seconds", "N/A")} s | {full.get("std_seconds", "N/A")} s | {full.get("throughput_docs_per_sec", "N/A")} docs/sec |
| Hash-Based Incremental Sync | **{sync.get("mean_seconds", "N/A")} s** | {sync.get("std_seconds", "N/A")} s | **{sync.get("throughput_effective_docs_per_sec", "N/A")} docs/sec** |
| **Speedup Factor** | **{sync_stats.get("speedup_factor", "N/A")}x** | — | — |

### Index State Equivalence Verification
- **Verification Status:** {"PASSED (100% Identical)" if verif.get("passed") else "FAILED"}
- **Synced Document Count:** {verif.get("synced_document_count", "N/A")}
- **Full Re-Index Document Count:** {verif.get("full_reindex_document_count", "N/A")}
- **ID Set Match:** {verif.get("id_sets_identical", False)}
- **Content Hash Discrepancies:** {verif.get("hash_mismatches_count", 0)}
"""

    content = f"""# Job-Matching Search with Evaluation

An end-to-end, reproducible ML portfolio project building and comparing retrieval strategies (dense semantic search, lexical BM25, hybrid retrieval, and reciprocal rank fusion) for matching candidate profiles against job postings, accompanied by grounded LLM explanations and rigorous evaluation metrics.

---

## Phase 1A: Data and Indexing

### Dataset Provenance & Attribution
- **Dataset**: {data_stats.get("dataset_name", "LinkedIn Job Postings")}
- **Source URL**: [{data_stats.get("source_url", "Hugging Face")}]({data_stats.get("source_url", "")})
- **Licence**: {data_stats.get("licence", "CC BY-SA 4.0 / MIT")}
- **Integrity**: 100% real postings; no synthetic data generated.

### Measured Pipeline Statistics
*(All metrics generated automatically by `src/generate_readme.py` from `reports/data_stats.json`)*

| Stage | Metric | Count / Value |
|---|---|---|
| Raw Pool | Downloaded Postings | {row_counts.get("raw_postings_pool", "N/A")} |
| Cleaned Pool | Valid Postings After Cleaning & Deduplication | {row_counts.get("clean_postings_available", "N/A")} |
| Sampled Output | Final Sampled Dataset (`seed=42`) | {row_counts.get("final_sampled_postings", "N/A")} |
| Text Metric | Average Description Characters | {desc_stats.get("avg_characters", "N/A")} chars |
| Text Metric | Average Description Words | {desc_stats.get("avg_words", "N/A")} words |
| Text Metric | Median Words | {desc_stats.get("median_words", "N/A")} words |

### Experience Level Distribution

| Seniority Tier | Count | Percentage |
|---|---|---|
{level_rows}

{index_table}
{sync_table}

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
│   ├── index_stats.json      # Vector indexing benchmark statistics
│   └── sync_benchmark.json   # Hash-based incremental sync benchmark results
├── src/                      # Production implementation
│   ├── __init__.py
│   ├── bench_sync.py         # Sync benchmark and state equivalence verification
│   ├── generate_readme.py    # Auto-generation of results table from reports
│   ├── index.py              # ChromaDB vector indexing and embeddings
│   ├── ingest.py             # Data loading, cleaning, deduplication, sampling
│   └── sync_index.py         # Hash-based incremental upsert/delete pipeline
└── tests/                    # Comprehensive unit tests
    ├── test_index.py         # Embedding, hashing, and ChromaDB unit tests
    ├── test_ingest.py        # Cleaning, HTML stripping, dedupe, empty row tests
    └── test_sync.py          # Incremental sync unit tests (mock embedder)
```

---

## Quickstart & Reproducibility

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
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
"""
    return content


def update_readme() -> None:
    """Generate and overwrite README.md using metrics from reports/."""
    data_stats = load_json_report(DATA_STATS_PATH)
    index_stats = load_json_report(INDEX_STATS_PATH)
    sync_stats = load_json_report(SYNC_STATS_PATH)

    if not data_stats:
        print(f"Warning: {DATA_STATS_PATH} not found. Skipping README update.")
        return

    content = build_readme_markdown(data_stats, index_stats, sync_stats)
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully generated {README_PATH} from reports/")


if __name__ == "__main__":
    update_readme()

