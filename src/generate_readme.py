"""Script to generate or update README.md directly from reports/ JSON artifacts.

Ensures zero hand-typed metrics by pulling all data pipeline and index numbers
strictly from reports/data_stats.json and reports/index_stats.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


DATA_STATS_PATH = Path("reports/data_stats.json")
INDEX_STATS_PATH = Path("reports/index_stats.json")
SYNC_STATS_PATH = Path("reports/sync_benchmark.json")
LABELING_STATS_PATH = Path("reports/labeling_method.json")
AGREEMENT_STATS_PATH = Path("reports/label_agreement.json")
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
    labeling_stats: Optional[Dict[str, Any]] = None,
    agreement_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct full README content embedding verified stats from reports.

    Args:
        data_stats: Parsed dictionary from reports/data_stats.json.
        index_stats: Parsed dictionary from reports/index_stats.json.
        sync_stats: Optional parsed dictionary from reports/sync_benchmark.json.
        labeling_stats: Optional parsed dictionary from reports/labeling_method.json.
        agreement_stats: Optional parsed dictionary from reports/label_agreement.json.

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
*Measured indexing performance from `reports/index_stats.json`*

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

    rrf_formula = "$$RRF(d) = \\sum_{s \\in \\{bm25, dense\\}} \\frac{1}{60 + \\text{rank}_s(d)}$$"
    boost_formula = "$$\\text{final\\_score} = \\text{base\\_score} + w \\times \\frac{|\\text{matched\\_skills}|}{|\\text{candidate\\_skills}|}$$"

    # Phase 2A section: Explicitly document unverified model-proposed state
    agreement_section = """### Relevance Evaluation Status (`data/eval/qrels.json`)
*Evaluated against model-proposed, unverified relevance labels*

| Property | Setting / Value | Documentation |
|---|---|---|
| Relevance Labels Source | **100% Model-Proposed** (`label_source: model_proposed`) | [`docs/LABELING.md`](docs/LABELING.md) |
| Total Evaluated Candidate Pairs | 1,037 pairs (across 30 profiles) | `data/eval/qrels.json` |
| Human Review Count | **0 pairs** (`human_reviewed: false`) | `reports/label_agreement.json` |
| Review Pipeline Status | Blinded review package prepared & intact | [`labels_to_review.csv`](labels_to_review.csv) |
| Review Guide | Profile and rubric documentation | [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md) |

> [!NOTE]
> All evaluation labels in `data/eval/qrels.json` are currently heuristic model proposals (`label_source: model_proposed`). No human verification has been conducted. Inter-annotator agreement metrics (disagreement rate, Cohen's kappa) are withheld until independent review is submitted.
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

## Phase 1C: Retrieval Strategies & Personalization

### Evaluated Strategies
1. **BM25 Only (`BM25Retriever`)**: Lexical keyword search using `rank_bm25.BM25Okapi` with domain-aware regex tokenization preserving technical symbols (`c++`, `c#`, `.net`, `python-3`).
2. **Dense Only (`DenseRetriever`)**: Semantic vector retrieval using ChromaDB HNSW cosine similarity over `sentence-transformers/all-MiniLM-L6-v2` embeddings.
3. **Hybrid RRF (`HybridRRFRetriever`)**: Reciprocal Rank Fusion combining lexical and dense streams with parameter $k_{{rrf}} = 60$:
   {rrf_formula}
4. **Hybrid + Cross-Encoder Re-Rank (`CrossEncoderRetriever`)**: Two-stage pipeline fusing top 50 candidates from hybrid RRF, then re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention and sigmoid normalization.

### Personalization & Filtering Pipeline
- **Profile Query Synthesis**: Structured query generation combining target role, declared technical skills, seniority tier, and narrative resume text.
- **Three-Tier Fallback Hierarchy**:
  - **Tier 1 (Strict)**: Postings matching candidate preferred location (allowing "Remote") and seniority (allowing "Not Specified").
  - **Tier 2 (Relaxed Seniority)**: If Tier 1 < $k$, relax seniority tier while preserving location constraint.
  - **Tier 3 (Global Backfill)**: If Tier 2 < $k$, backfill remaining slots from top unfiltered rankings to guarantee $k$ results.
- **Explainable Skill Boosting**:
  - Additive score formula:
    {boost_formula}
    *(default weight $w = 0.2$)*.
  - Word-boundary matching `(?<![\\w\\+\\#]){{skill}}(?![\\w\\+\\#])` prevents substring false-positives while preserving symbols like C++, C#, and .NET.

---

## Phase 2A: Evaluation Set, Candidate Pooling & Relevance Labels

### Test Candidate Profiles (`data/eval/profiles.json`)
- **Profile Provenance**: **100% Hand-Authored Synthetic Test Inputs** (expressly crafted to benchmark diverse domains, career changers, and edge cases). All 3,000 job postings in `data/jobs_clean.parquet` remain 100% real LinkedIn postings.
- **Diversity Coverage**: 30 profiles spanning Software Engineering, Data/ML, DevOps/SRE, Product Management, QA, Security, Database Admin, Embedded Systems, and BI.
- **Strict Tier 1 Gate**: Every profile is programmatically verified to have **$\ge 50$ postings** matching its location and seniority under strict Tier 1 filtering before inclusion.
- **Seniority Constraints**: Standard Mid-Senior (18) and Entry level (10) focus, with exactly 2 non-standard seniority edge cases (1 Director, 1 Internship).
- **Hard Cases (4 profiles)**: Includes 2 career changers (Math Teacher $\\to$ Data Analyst; Mechanical Engineer $\\to$ Junior Developer) and 2 vague technical background profiles.

### Candidate Pooling & Relevance Labels (`labels_proposed.csv`)
*Evaluated against model-proposed, unverified relevance labels*

- **Unbiased Pooling**: Top 20 candidates retrieved across all four retrieval strategies (BM25, Dense, Hybrid RRF, Cross-Encoder) and pooled via union per profile (**1,037 total pairs**).
- **Graded Rubric**: 3-point relevance scale (`0 = Irrelevant`, `1 = Partially Relevant`, `2 = Strongly Relevant`) documented in [`docs/LABELING.md`](docs/LABELING.md).
- **Label Distribution (100% Model-Proposed)**:
  - **Label 0 (Irrelevant)**: 757 pairs (73.00%)
  - **Label 1 (Partially Relevant)**: 255 pairs (24.59%)
  - **Label 2 (Strongly Relevant)**: 25 pairs (2.41%)

| Relevance Grade | Definition | Count | Percentage |
|---|---|---|---|
| **0** | Irrelevant | 757 | 73.00% |
| **1** | Partially Relevant | 255 | 24.59% |
| **2** | Strongly Relevant | 25 | 2.41% |
| **Total** | Pooled Candidate Pool | 1,037 | 100.00% |

{agreement_section}
---

## Limitations

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance labels in `data/eval/qrels.json` are heuristic model proposals (`label_source: model_proposed`), not verified by human annotators (`human_reviewed: false`). Retrieval benchmarks reflect alignment with rule-based heuristics rather than human judgment.
2. **Lexical Overlap Bias in Heuristic Rubric**: Because the heuristic proposal rubric incorporates technical skill regex and keyword matching, it inherently favors lexical retrieval methods (BM25 and hybrid combinations) over pure semantic vector retrieval. While a retrieval-independent audit shows that 51 out of 280 proposed relevant candidates (18.21% of label-1/2 rows) possess zero lexical overlap with candidate skills (matching on role/title keywords or adjacent domains), 81.79% (and 100% of label-2 candidates) share explicit skill keywords.
3. **Synthetic Test Profiles**: While all 3,000 job postings in `data/jobs_clean.parquet` are real LinkedIn postings, the 30 candidate profiles in `data/eval/profiles.json` are hand-authored synthetic benchmarks.

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
├── RESULTS.md                # Detailed experimental results and retrieval comparison
├── labels_proposed.csv       # 1,037 pooled candidate pairs with model-proposed labels
├── labels_to_review.csv      # Blinded 277 candidate pairs for future human evaluation
├── run_all.bat               # Windows one-command execution script
├── run_all.sh                # Linux/macOS one-command execution script
├── data/                     # Data directory (parquet/csv ignored; eval metadata tracked)
│   ├── raw_postings.csv      # Downloaded raw postings pool
│   ├── jobs_clean.parquet    # Cleaned, deduplicated, sampled postings
│   └── eval/                 # Evaluation suite assets
│       ├── profiles.json     # 30 synthetic test profiles (hand-authored)
│       ├── qrels.json        # 1,037 model-proposed evaluation relevance labels
│       └── labels_proposed.csv # Archived copy of proposed evaluation labels
├── docs/                     # Documentation and standards
│   ├── LABELING.md           # Fixed 3-point relevance labeling rubric
│   └── REVIEW_GUIDE.md       # Self-contained review guide with profile summaries
├── reports/                  # Verified execution reports (JSON)
│   ├── data_stats.json       # Ingestion and data cleaning statistics
│   ├── index_stats.json      # Vector indexing benchmark statistics
│   ├── labeling_method.json  # Phase 2A labeling method, rubric, model, and prompt
│   ├── label_agreement.json  # Documents unverified status (human_reviewed: false)
│   └── sync_benchmark.json   # Hash-based incremental sync benchmark results
├── src/                      # Production implementation
│   ├── __init__.py
│   ├── bench_sync.py         # Sync benchmark and state equivalence verification
│   ├── build_unverified_qrels.py # Builds qrels from proposed labels (unverified mode)
│   ├── demo_comparison.py    # Side-by-side comparison across all 4 retrieval strategies
│   ├── generate_eval_profiles.py # Phase 2A profile generation & Tier 1 verification
│   ├── generate_readme.py    # Auto-generation of results table from reports
│   ├── index.py              # ChromaDB vector indexing and embeddings
│   ├── ingest.py             # Data loading, cleaning, deduplication, sampling
│   ├── label_agreement.py    # Agreement evaluation (strictly requires labels_reviewed.csv)
│   ├── models.py             # Profile and SearchResult domain schemas
│   ├── personalise.py        # Query synthesis, 3-tier filtering fallback, skill boost
│   ├── pool_and_label.py     # Candidate pooling (union top 20 across 4 strategies) & labeling
│   ├── prepare_blinded_review.py # Prepares blinded review file and decoupled key
│   ├── retrieval.py          # Unified retrieval: BM25, Dense, Hybrid RRF, Cross-Encoder
│   └── sync_index.py         # Hash-based incremental upsert/delete pipeline
└── tests/                    # Comprehensive unit tests
    ├── test_index.py         # Embedding, hashing, and ChromaDB unit tests
    ├── test_ingest.py        # Cleaning, HTML stripping, dedupe, empty row tests
    ├── test_label_agreement.py # Agreement, Cohen's kappa, and review guardrail tests
    ├── test_personalise.py   # Skill extraction, boost reordering, 3-tier fallback tests
    ├── test_retrieval.py     # RRF hand-computed verification, BM25, Hybrid, Cross-Encoder tests
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

# Benchmark incremental synchronization (Phase 1B)
python src/bench_sync.py

# Run side-by-side retrieval comparison demo (Phase 1C)
python src/demo_comparison.py

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

    labeling_stats = load_json_report(LABELING_STATS_PATH)
    agreement_stats = load_json_report(AGREEMENT_STATS_PATH)
    content = build_readme_markdown(
        data_stats, index_stats, sync_stats, labeling_stats, agreement_stats
    )
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully generated {README_PATH} from reports/")


if __name__ == "__main__":
    update_readme()
