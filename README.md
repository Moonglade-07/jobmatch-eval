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
| Cleaned Pool | Valid Postings After Cleaning & Deduplication | 8052 |
| Sampled Output | Final Sampled Dataset (`seed=42`) | 3000 |
| Text Metric | Average Description Characters | 3637.38 chars |
| Text Metric | Average Description Words | 505.49 words |
| Text Metric | Median Words | 457 words |

### Experience Level Distribution

| Seniority Tier | Count | Percentage |
|---|---|---|
| Mid-Senior level | 1027 | 34.23% |
| Entry level | 984 | 32.8% |
| Not Specified | 746 | 24.87% |
| Director | 126 | 4.2% |
| Internship | 63 | 2.1% |
| Executive | 54 | 1.8% |

### Measured Indexing Performance
*Measured indexing performance from `reports/index_stats.json`*

| Metric | Measured Value |
|---|---|
| Vector Store | ChromaDB (HNSW Cosine) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension | 384 |
| Total Documents Indexed | 3000 |
| Indexing Time | 159.59 s |
| Indexing Throughput | 18.8 docs/sec |
| Metadata Fields | `job_id, title, company, location, experience_level, content_hash` |

---

## Phase 1B: Incremental Index Synchronization Benchmark

### Simulated Change Scenario
- **Dataset Baseline Size:** 3000 postings
- **Modifications:** 5.0% of descriptions (150 postings)
- **Additions:** 2.0% new postings (60 postings)
- **Deletions:** 2.0% removed postings (60 postings)
- **Unchanged:** 2790 postings (zero embedding overhead)
- **Benchmark Repetitions:** 3 runs per strategy

### Measured Speedup & Throughput
*(Numbers extracted directly from `reports/sync_benchmark.json`)*

| Strategy | Mean Latency | Latency Std Dev | Effective Throughput |
|---|---|---|---|
| Full Re-Index (from scratch) | 159.636 s | 8.408 s | 18.79 docs/sec |
| Hash-Based Incremental Sync | **12.295 s** | 0.973 s | **244.0 docs/sec** |
| **Speedup Factor** | **12.98x** | — | — |

### Index State Equivalence Verification
- **Verification Status:** PASSED (100% Identical)
- **Synced Document Count:** 3000
- **Full Re-Index Document Count:** 3000
- **ID Set Match:** True
- **Content Hash Discrepancies:** 0

---

## Phase 1C: Retrieval Strategies & Personalization

### Evaluated Strategies
1. **BM25 Only (`BM25Retriever`)**: Lexical keyword search using `rank_bm25.BM25Okapi` with domain-aware regex tokenization preserving technical symbols (`c++`, `c#`, `.net`, `python-3`).
2. **Dense Only (`DenseRetriever`)**: Semantic vector retrieval using ChromaDB HNSW cosine similarity over `sentence-transformers/all-MiniLM-L6-v2` embeddings.
3. **Hybrid RRF (`HybridRRFRetriever`)**: Reciprocal Rank Fusion combining lexical and dense streams with parameter $k_{rrf} = 60$:
   $$RRF(d) = \sum_{s \in \{bm25, dense\}} \frac{1}{60 + \text{rank}_s(d)}$$
4. **Hybrid + Cross-Encoder Re-Rank (`CrossEncoderRetriever`)**: Two-stage pipeline fusing top 50 candidates from hybrid RRF, then re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention and sigmoid normalization.

### Personalization & Filtering Pipeline
- **Profile Query Synthesis**: Structured query generation combining target role, declared technical skills, seniority tier, and narrative resume text.
- **Three-Tier Fallback Hierarchy**:
  - **Tier 1 (Strict)**: Postings matching candidate preferred location (allowing "Remote") and seniority (allowing "Not Specified").
  - **Tier 2 (Relaxed Seniority)**: If Tier 1 < $k$, relax seniority tier while preserving location constraint.
  - **Tier 3 (Global Backfill)**: If Tier 2 < $k$, backfill remaining slots from top unfiltered rankings to guarantee $k$ results.
- **Explainable Skill Boosting**:
  - Additive score formula:
    $$\text{final\_score} = \text{base\_score} + w \times \frac{|\text{matched\_skills}|}{|\text{candidate\_skills}|}$$
    *(default weight $w = 0.2$)*.
  - Word-boundary matching `(?<![\w\+\#]){skill}(?![\w\+\#])` prevents substring false-positives while preserving symbols like C++, C#, and .NET.

---

## Phase 2A: Evaluation Set, Candidate Pooling & Relevance Labels

### Test Candidate Profiles (`data/eval/profiles.json`)
- **Profile Provenance**: **100% Hand-Authored Synthetic Test Inputs** (expressly crafted to benchmark diverse domains, career changers, and edge cases). All 3,000 job postings in `data/jobs_clean.parquet` remain 100% real LinkedIn postings.
- **Diversity Coverage**: 30 profiles spanning Software Engineering, Data/ML, DevOps/SRE, Product Management, QA, Security, Database Admin, Embedded Systems, and BI.
- **Strict Tier 1 Gate**: Every profile is programmatically verified to have **$\ge 50$ postings** matching its location and seniority under strict Tier 1 filtering before inclusion.
- **Seniority Constraints**: Standard Mid-Senior (18) and Entry level (10) focus, with exactly 2 non-standard seniority edge cases (1 Director, 1 Internship).
- **Hard Cases (4 profiles)**: Includes 2 career changers (Math Teacher $\to$ Data Analyst; Mechanical Engineer $\to$ Junior Developer) and 2 vague technical background profiles.

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

### Relevance Evaluation Status (`data/eval/qrels.json`)
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
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
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
