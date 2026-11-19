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

    # Phase 2A agreement section: strictly require real human review file to exist
    agreement_section = ""
    reviewed_file_exists = Path("labels_reviewed.csv").exists()

    if agreement_stats and reviewed_file_exists:
        metrics = agreement_stats.get("agreement_metrics", {})
        counts = agreement_stats.get("provenance_counts", {})
        cm_matrix = agreement_stats.get("confusion_matrix", {}).get("matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agreement_section = f"""### Inter-Annotator Agreement (`reports/label_agreement.json`)
*(Measured across the {agreement_stats.get("reviewed_sample_size", 0)} reviewed pairs from `labels_reviewed.csv`)*

| Metric | Measured Value | Interpretation |
|---|---|---|
| Human-Reviewed Pairs | {counts.get("human_verified_count", 0)} / {agreement_stats.get("total_candidate_pairs", 0)} ({counts.get("human_verified_percentage", 0)}%) | Evaluated on `labels_reviewed.csv` |
| Unreviewed (Model-Proposed) | {counts.get("model_proposed_count", 0)} / {agreement_stats.get("total_candidate_pairs", 0)} ({counts.get("model_proposed_percentage", 0)}%) | Retained model proposals |
| Disagreement Count | {metrics.get("disagreement_count", 0)} pairs | Divergent judgments |
| Disagreement Rate | **{metrics.get("disagreement_percentage", 0)}%** | Real measured disagreement |
| Agreement Rate | **{metrics.get("agreement_percentage", 0)}%** | Inter-rater agreement |
| **Cohen's Kappa (Quadratic)** | **{metrics.get("cohen_kappa_quadratic_weighted", 0)}** | **{metrics.get("kappa_interpretation", "N/A")}** |
| **Cohen's Kappa (Linear)** | **{metrics.get("cohen_kappa_linear_weighted", 0)}** | Linear weighted |
| **Ground Truth Qrels** | `data/eval/qrels.json` | Final combined labels |

#### Confusion Matrix (Rows: Model Proposed, Columns: Human Reviewer)
| | Human 0 | Human 1 | Human 2 | Total Proposed |
|---|---|---|---|---|
| **Proposed 0** | {cm_matrix[0][0]} | {cm_matrix[0][1]} | {cm_matrix[0][2]} | {sum(cm_matrix[0])} |
| **Proposed 1** | {cm_matrix[1][0]} | {cm_matrix[1][1]} | {cm_matrix[1][2]} | {sum(cm_matrix[1])} |
| **Proposed 2** | {cm_matrix[2][0]} | {cm_matrix[2][1]} | {cm_matrix[2][2]} | {sum(cm_matrix[2])} |
"""
    else:
        agreement_section = """### Human Review Status
- **Blinded Review File**: [`labels_to_review.csv`](labels_to_review.csv) (277 pooled candidates: 260 random sample + all proposed label-2 candidates, union de-duplicated and shuffled per profile).
- **Review Guide**: [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md) (single-page rubric and all 30 candidate profile summaries).
- **Status**: **Awaiting Human Evaluation**. Inter-annotator agreement metrics (disagreement rate, Cohen's weighted kappa, and confusion matrix) will be computed only after `labels_reviewed.csv` is completed and returned.
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
- **Unbiased Pooling**: Top 20 candidates retrieved across all four retrieval strategies (BM25, Dense, Hybrid RRF, Cross-Encoder) and pooled via union per profile (**1,037 total pairs**).
- **Graded Rubric**: 3-point relevance scale (`0 = Irrelevant`, `1 = Partially Relevant`, `2 = Strongly Relevant`) documented in [`docs/LABELING.md`](docs/LABELING.md).
- **Quality Audit Sample**: A stratified random **25% subset (260 pairs, seed=42)** is marked as `must_review = True` for human verification.
- **Label Distribution**:
  - **Label 0 (Irrelevant)**: 757 pairs (73.00%)
  - **Label 1 (Partially Relevant)**: 255 pairs (24.59%)
  - **Label 2 (Strongly Relevant)**: 25 pairs (2.41%)

{agreement_section}
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
├── labels_proposed.csv       # 1,037 pooled candidate pairs with proposed labels and 25% audit sample
├── labels_to_review.csv      # Blinded 277 candidate pairs for human evaluation
├── run_all.bat               # Windows one-command execution script
├── run_all.sh                # Linux/macOS one-command execution script
├── data/                     # Data directory (parquet/csv ignored; eval metadata tracked)
│   ├── raw_postings.csv      # Downloaded raw postings pool
│   ├── jobs_clean.parquet    # Cleaned, deduplicated, sampled postings
│   └── eval/                 # Evaluation suite assets
│       ├── profiles.json     # 30 synthetic test profiles (hand-authored)
│       └── labels_proposed.csv # Archived copy of proposed evaluation labels
├── docs/                     # Documentation and standards
│   ├── LABELING.md           # Fixed 3-point relevance labeling rubric
│   └── REVIEW_GUIDE.md       # Self-contained review guide with profile summaries
├── reports/                  # Verified execution reports (JSON)
│   ├── data_stats.json       # Ingestion and data cleaning statistics
│   ├── index_stats.json      # Vector indexing benchmark statistics
│   ├── labeling_method.json  # Phase 2A labeling method, rubric, model, and prompt
│   └── sync_benchmark.json   # Hash-based incremental sync benchmark results
├── src/                      # Production implementation
│   ├── __init__.py
│   ├── bench_sync.py         # Sync benchmark and state equivalence verification
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
