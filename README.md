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
| Index Build Time | 159.59 s |
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

> [!NOTE]
> **Scope of Incremental Synchronization**: This benchmark measures incremental synchronization of the **ChromaDB dense vector index** specifically, where unchanged documents bypass embedding inference via SHA-256 content hashes. Because `rank_bm25` does not support incremental updates, the BM25 lexical index and tokenized corpus must be rebuilt in full (~6.0 s) whenever postings are added, modified, or removed. The production pipeline is therefore hybrid: dense vector updates are incremental, while lexical index updates require a complete rebuild.

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
- **Label Distribution (100% Rule-Based Proposed Labels)**:
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
| Relevance Labels Source | **100% Rule-Based Proposed Labels** (`label_source: model_proposed`) | [`docs/LABELING.md`](docs/LABELING.md) |
| Total Evaluated Candidate Pairs | 1,037 pairs (across 30 profiles) | `data/eval/qrels.json` |
| Human Review Count | **0 pairs** (`human_reviewed: false`) | `reports/label_agreement.json` |
| Review Pipeline Status | Blinded review package prepared & intact | [`labels_to_review.csv`](labels_to_review.csv) |
| Review Guide | Profile and rubric documentation | [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md) |

> [!NOTE]
> All evaluation labels in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`) generated via deterministic heuristics; unverified by humans. Inter-annotator agreement metrics (disagreement rate, Cohen's kappa) are withheld until independent review is submitted.

---

## Phase 2B: Retrieval Evaluation & Benchmarking

### Benchmark Scope & Methodology
- **30 Candidate Profiles**: Hand-authored synthetic profiles spanning Software Engineering, Data/ML, DevOps, Product Management, QA, and Security.
- **Graded Relevance Labels**: Evaluated against rule-based proposed labels (`jobmatch-rubric-v1`) across 1,037 pooled candidate pairs.
- **Protocol**: Unjudged documents in top $k$ treated as non-relevant (label 0). Profiles with zero relevant documents (`prof_07`: Mobile Developer, `prof_23`: Computer Vision Engineer) are explicitly excluded from Recall@10 and MRR means (evaluated over 28 profiles).
- **Statistical Significance**: Seeded empirical bootstrap 95% confidence intervals ($N=1,000$ iterations) for all metric macro-averages and paired differences.
- **In-Depth Analysis Report**: Comprehensive query case studies, statistical power analysis, and failure mode breakdowns are documented in [`reports/analysis_retrieval.md`](reports/analysis_retrieval.md).

### 1. Main Setting: Product Pipeline (Filters ON, Skill Boost ON $w=0.2$)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 | CPU p50 Latency | CPU p95 Latency |
|---|---|---|---|---|---|---|---|
| **BM25 Only** | 0.520 [0.41, 0.63] | 0.548 [0.47, 0.63] | 0.887 [0.79, 0.97] | 0.615 [0.52, 0.70] | 0.0% | 136.2 ms | 168.3 ms |
| **Dense Only (ChromaDB)** | 0.427 [0.31, 0.55] | 0.395 [0.31, 0.48] | 0.701 [0.56, 0.85] | 0.423 [0.33, 0.52] | 0.0% | 74.2 ms | 91.3 ms |
| **Hybrid RRF (k=60)** | 0.533 [0.42, 0.65] | 0.546 [0.47, 0.63] | 0.910 [0.82, 0.98] | 0.610 [0.52, 0.68] | 0.0% | 181.5 ms | 206.4 ms |
| **Hybrid + Cross-Encoder** | 0.527 [0.41, 0.63] | 0.544 [0.47, 0.63] | 0.864 [0.76, 0.95] | 0.595 [0.51, 0.67] | 0.0% | 2479.3 ms | 2813.0 ms |

> [!NOTE]
> All four strategies exhibit 0.00% unjudged rate in top 10 under the main configuration because candidate pooling in Phase 2A pooled the top 20 candidates across these exact four retrieval streams.

### 2. Secondary Setting: Pure Retrieval (Filters OFF, Skill Boost OFF) — Not comparable (labels pooled under filters, 76% unjudged)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 |
|---|---|---|---|---|---|
| **BM25 Only** | 0.127 [0.06, 0.20] | 0.142 [0.09, 0.20] | 0.254 [0.14, 0.38] | 0.162 [0.10, 0.23] | 85.3% |
| **Dense Only (ChromaDB)** | 0.140 [0.09, 0.20] | 0.138 [0.09, 0.20] | 0.295 [0.18, 0.44] | 0.161 [0.10, 0.23] | 76.0% |
| **Hybrid RRF (k=60)** | 0.147 [0.08, 0.23] | 0.141 [0.09, 0.20] | 0.292 [0.17, 0.44] | 0.177 [0.10, 0.26] | 82.3% |
| **Hybrid + Cross-Encoder** | 0.160 [0.09, 0.25] | 0.215 [0.15, 0.30] | 0.320 [0.20, 0.45] | 0.186 [0.13, 0.25] | 79.3% |

> [!WARNING]
> **Not comparable (labels pooled under filters, 76% unjudged)**: Candidate pooling was performed under active location and seniority filters. Out-of-market jobs retrieved when filters are disabled were never pooled or evaluated, resulting in unjudged rates of 76%–85% (treated as non-relevant label 0). These numbers do not reflect pure retrieval quality.

### 3. Paired Statistical Significance (Bootstrap 95% CIs on Differences)
*Evaluated against model-proposed, unverified relevance labels*

| Comparison ($A$ vs $B$) | Metric | Mean Difference ($A - B$) | 95% Bootstrap CI | Statistical Conclusion |
|---|---|---|---|---|
| **BM25 vs Dense** | `precision@5` | +0.0933 | [-0.0067, 0.1935] | No clear difference (CI spans 0.0) |
| **BM25 vs Dense** | `recall@10` | +0.1523 | [0.0388, 0.2735] | **Clear difference against rule-based labels** |
| **BM25 vs Dense** | `mrr` | +0.1863 | [0.0640, 0.3027] | **Clear difference against rule-based labels** |
| **BM25 vs Dense** | `ndcg@10` | +0.1916 | [0.1109, 0.2610] | **Clear difference against rule-based labels** |
| **Hybrid RRF vs BM25** | `precision@5` | +0.0133 | [-0.0400, 0.0667] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs BM25** | `recall@10` | -0.0020 | [-0.0494, 0.0450] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs BM25** | `mrr` | +0.0230 | [-0.0366, 0.0884] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs BM25** | `ndcg@10` | -0.0048 | [-0.0502, 0.0384] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs Dense** | `precision@5` | +0.1067 | [0.0400, 0.1800] | **Clear difference against rule-based labels** |
| **Hybrid RRF vs Dense** | `recall@10` | +0.1503 | [0.0618, 0.2597] | **Clear difference against rule-based labels** |
| **Hybrid RRF vs Dense** | `mrr` | +0.2093 | [0.1033, 0.3179] | **Clear difference against rule-based labels** |
| **Hybrid RRF vs Dense** | `ndcg@10` | +0.1868 | [0.1277, 0.2414] | **Clear difference against rule-based labels** |
| **Cross-Encoder vs Hybrid RRF** | `precision@5` | -0.0067 | [-0.0733, 0.0600] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs Hybrid RRF** | `recall@10` | -0.0012 | [-0.0408, 0.0333] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs Hybrid RRF** | `mrr` | -0.0456 | [-0.1337, 0.0340] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs Hybrid RRF** | `ndcg@10` | -0.0146 | [-0.0579, 0.0308] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `precision@5` | +0.0067 | [-0.0800, 0.0800] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `recall@10` | -0.0032 | [-0.0524, 0.0517] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `mrr` | -0.0226 | [-0.1144, 0.0667] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `ndcg@10` | -0.0193 | [-0.0781, 0.0410] | No clear difference (CI spans 0.0) |

### 4. Query Latency Profile on CPU (25 timed runs per step)
*(Measured on AMD 12-core CPU, PyTorch CPU execution with `torch_threads=6`, `CUDA: False`)*

> **Measurement Conditions**: Latency was measured warm (retriever warm-up executed before timing), single-process, CPU-only, with inverted index, tokenized corpus, and regex caches pre-built outside the timed loop.

| Architecture / Step | p50 Latency (ms) | p95 Latency (ms) | Mean Latency (ms) | Details / Operations |
|---|---|---|---|---|
| **Dense Only** | 74.24 | 91.32 | 75.06 | Query embedding (`all-MiniLM-L6-v2`) + ChromaDB HNSW |
| **BM25 Only** | 136.24 | 168.32 | 146.96 | Lexical BM25Okapi scoring over full corpus |
| **Hybrid RRF ($k=60$)** | 181.46 | 206.44 | 177.55 | BM25 + Dense retrieval + RRF fusion |
| **Cross-Encoder: Step 1 (Retrieval)** | 157.2 | 179.94 | 166.46 | Hybrid RRF candidate pool retrieval (top 50) |
| **Cross-Encoder: Step 2 (Neural Scoring)** | 2337.29 | 2622.19 | 2371.4 | CPU Transformer forward pass re-ranking 50 pairs |
| **Cross-Encoder: End-to-End Pipeline** | **2479.28** | **2813.02** | **2537.85** | Total two-stage retrieval and neural re-ranking |

*One-off startup / index build times (outside timed queries): BM25 Index Build = 5992.1 ms, Dense Init = 877.8 ms, Cross-Encoder Load = 242.5 ms.*


### 5. Ablation Study: Filter, Boost, and Weight Tuning
*Evaluated against model-proposed, unverified relevance labels*

| Configuration | Precision@5 | Recall@10 | MRR | nDCG@10 | Unjudged Rate@10 |
|---|---|---|---|---|---|
| Filters ON, Boost ON (w=0.2) [Main] | 0.5200 | 0.5476 | 0.8869 | 0.6147 | 0.0% |
| Filters ON, Boost OFF (w=0.0) | 0.5200 | 0.5464 | 0.8690 | 0.6062 | 0.0% |
| Filters OFF, Boost ON (w=0.2) | 0.1333 | 0.1562 | 0.2750 | 0.1768 | 84.3% |
| Filters OFF, Boost OFF (w=0.0) [Pure] | 0.1267 | 0.1420 | 0.2541 | 0.1622 | 85.3% |
| Filters ON, Boost w=0.1 | 0.5200 | 0.5476 | 0.8869 | 0.6135 | 0.0% |
| Filters ON, Boost w=0.4 | 0.5267 | 0.5476 | 0.8861 | 0.6122 | 0.0% |

### 6. Circularity Audit: Relevant Subset with Zero Literal Skill Overlap (Indicative Only: 51 pairs, 17 profiles)
*Evaluated against model-proposed, unverified relevance labels*

Subset consists of **51 relevant candidate postings across 17 profiles** that share zero literal keyword overlap with candidate declared skills (13 profiles had zero such postings and are excluded from Recall/MRR means):

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) |
|---|---|---|---|---|
| **BM25 Only** | 0.0333 [0.00, 0.08] | 0.2588 [0.12, 0.43] | 0.1751 [0.07, 0.31] | 0.0932 [0.04, 0.16] |
| **Dense Only (ChromaDB)** | 0.0600 [0.02, 0.11] | 0.3931 [0.23, 0.57] | 0.3437 [0.15, 0.53] | 0.1383 [0.07, 0.21] |
| **Hybrid RRF (k=60)** | 0.0800 [0.03, 0.14] | 0.3180 [0.16, 0.49] | 0.2319 [0.10, 0.39] | 0.1124 [0.05, 0.18] |
| **Hybrid + Cross-Encoder** | 0.0467 [0.01, 0.08] | 0.2552 [0.12, 0.42] | 0.3077 [0.13, 0.51] | 0.1098 [0.05, 0.18] |

> [!NOTE]
> **Indicative Finding Only (51 pairs, 17 profiles)**: Evaluated on the subset of 51 relevant candidate postings across 17 profiles with zero literal skill matches (13 profiles have zero relevant items in this slice and are excluded from Recall/MRR means). Dense retrieval achieves Recall@10 of 0.3931 [0.2333, 0.5687] vs BM25 0.2588 [0.1245, 0.4265] and MRR of 0.3437 [0.1546, 0.5314] vs BM25 0.1751 [0.0727, 0.3139]. This indicates that dense vector retrieval can surface conceptually relevant postings when keyword markers are absent, but the small sample size and wide confidence intervals mean this is indicative only.

---

## Limitations

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance labels in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`), generated via deterministic heuristics (`jobmatch-rubric-v1`) and unverified by humans (`human_reviewed: false`). Evaluated retrieval metrics reflect alignment with deterministic rule-based criteria rather than human preference.
2. **Lexical Overlap Bias in Heuristic Rubric**: Because the heuristic proposal rubric incorporates technical skill regex and keyword matching, it inherently favors lexical retrieval methods (BM25 and hybrid combinations) over pure semantic vector retrieval. While the retrieval-independent audit shows that 51 out of 280 proposed relevant candidates (18.21% of label-1/2 rows) possess zero lexical overlap with candidate skills, 81.79% (and 100% of label-2 candidates) share explicit skill keywords. High BM25 scores reflect shared lexical signals with the rubric, not evidence that BM25 is the better retriever in absolute terms.
3. **Retrieval-Independent Check is Indicative Only (51 pairs, 17 profiles)**: When evaluated on the 51 relevant candidates with zero literal skill matches, dense retrieval achieves higher recall (0.3931 vs 0.2588) and MRR (0.3437 vs 0.1751) than BM25, indicating semantic capability when keywords are missing. However, because only 17 profiles have any relevant documents in this slice, confidence intervals are wide and the result is indicative only.
4. **Niche Role Coverage Gap (< 3 Relevant Docs)**: In this 3,000-posting LinkedIn sample, niche technical roles are poorly covered under strict location and seniority criteria. Exactly 6 of the 30 profiles have fewer than 3 relevant documents:
   - `prof_01` (Machine Learning Engineer, Remote): 1 relevant doc
   - `prof_07` (Mobile Developer, San Francisco, CA): 0 relevant docs
   - `prof_09` (Database Administrator, Houston, TX): 2 relevant docs
   - `prof_11` (QA Automation Engineer, Denver, CO): 2 relevant docs
   - `prof_19` (NLP Engineer, Remote): 1 relevant doc
   - `prof_23` (Computer Vision Engineer, Remote): 0 relevant docs
   Broad roles (e.g. Junior Data Analyst with 32 relevant docs, Junior Software Developer with 21) have strong coverage, while specialized disciplines (mobile development, computer vision, NLP) have sparse representation in this dataset slice.
5. **Ranking Inside Pre-Filtered Pool, Not Full-Index Retrieval**: Reported retrieval metrics evaluate ranking within candidate pools constrained by location and seniority filters (with fallback), rather than unconstrained retrieval over all 3,000 raw postings.
6. **Skill Boost Gains Inflated by Label/Signal Overlap**: Because the rule-based labeling rubric (`jobmatch-rubric-v1`) rewards skill regex matches, applying the skill boost directly reinforces the labeling criteria, artificially inflating observed ranking improvements.
7. **BM25 Full Rebuild vs Incremental Dense Indexing**: Latency benchmarks are measured warm in a single process on CPU with all caches pre-built. While ChromaDB supports hash-based incremental updates, `rank_bm25` has no incremental update mechanism; any mutation to the dataset requires rebuilding the BM25 index and tokenized corpus in full (~6.0 s).
8. **Synthetic Test Profiles**: While all 3,000 job postings in `data/jobs_clean.parquet` are real LinkedIn postings, the 30 candidate profiles in `data/eval/profiles.json` are hand-authored synthetic benchmarks.

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
│   ├── analysis_retrieval.md # Comprehensive Phase 2B retrieval evaluation analysis
│   ├── data_stats.json       # Ingestion and data cleaning statistics
│   ├── index_stats.json      # Vector indexing benchmark statistics
│   ├── labeling_method.json  # Phase 2A labeling method, rubric, model, and prompt
│   ├── label_agreement.json  # Documents unverified status (human_reviewed: false)
│   ├── retrieval_ablation.json # Phase 2B ablation grid results
│   ├── retrieval_circularity_check.json # Phase 2B zero-skill-overlap circularity audit
│   ├── retrieval_latency.json # Phase 2B CPU query latency profile (p50, p95)
│   ├── retrieval_metrics_main.json # Phase 2B main pipeline retrieval metrics & CIs
│   ├── retrieval_metrics_pure.json # Phase 2B pure retrieval metrics & CIs
│   ├── retrieval_paired_comparisons.json # Phase 2B paired bootstrap comparisons
│   ├── retrieval_per_profile_main.csv # Phase 2B per-profile main results
│   ├── retrieval_per_profile_pure.csv # Phase 2B per-profile pure results
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
│   ├── metrics.py            # Precision@k, Recall@k, MRR, nDCG@k evaluation metrics
│   ├── models.py             # Profile and SearchResult domain schemas
│   ├── personalise.py        # Query synthesis, 3-tier filtering fallback, skill boost
│   ├── pool_and_label.py     # Candidate pooling (union top 20 across 4 strategies) & labeling
│   ├── prepare_blinded_review.py # Prepares blinded review file and decoupled key
│   ├── retrieval.py          # Unified retrieval: BM25, Dense, Hybrid RRF, Cross-Encoder
│   ├── run_retrieval_eval.py # Phase 2B full benchmark, ablation, and latency runner
│   └── sync_index.py         # Hash-based incremental upsert/delete pipeline
└── tests/                    # Comprehensive unit tests
    ├── test_index.py         # Embedding, hashing, and ChromaDB unit tests
    ├── test_ingest.py        # Cleaning, HTML stripping, dedupe, empty row tests
    ├── test_label_agreement.py # Agreement, Cohen's kappa, and review guardrail tests
    ├── test_metrics.py       # P@k, Recall@k, MRR, nDCG@k unit tests with manual cases
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

# Run Phase 2B full retrieval evaluation across all strategies
python src/run_retrieval_eval.py

# Run full unit test suite
pytest -v tests/

# Update README and RESULTS results tables directly from reports
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
