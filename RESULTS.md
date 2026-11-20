# Experimental Results & Provenance Ledger

This document lists every measured number across all experimental phases,
accredited with its exact producing command and source JSON artifact in `reports/`.

## Metric Provenance Registry

| Pipeline Stage | Primary Producing Command | Source Report File | Measured Entities |
|---|---|---|---|
| **Phase 1A: Ingestion & Cleaning** | `python src/ingest.py` | `reports/data_stats.json` | Row counts, text length statistics, seniority distribution |
| **Phase 1A: Dense Vector Indexing** | `python src/index.py` | `reports/index_stats.json` | Vector dimension, indexed count, build duration, throughput |
| **Phase 1B: Incremental Sync Benchmark** | `python src/bench_sync.py` | `reports/sync_benchmark.json` | Full re-index latency, sync latency, speedup factor (12.98x) |
| **Phase 2A: Pooling & Label Proposals** | `python src/pool_and_label.py` | `reports/labeling_method.json` | 1,037 candidate pairs, 3-tier label counts (757/255/25) |
| **Phase 2A: Relevance Agreement Guard** | `python src/build_unverified_qrels.py` | `reports/label_agreement.json` | Unverified status, human verified count (0) |
| **Phase 2B: Main Pipeline Retrieval** | `python src/run_retrieval_eval.py` | `reports/retrieval_metrics_main.json` | P@5, Recall@10, MRR, nDCG@10, bootstrap 95% CIs |
| **Phase 2B: Pure Retrieval Setting** | `python src/run_retrieval_eval.py` | `reports/retrieval_metrics_pure.json` | Unconstrained retrieval metrics (76%-85% unjudged rate) |
| **Phase 2B: Paired Statistical Comparisons** | `python src/run_retrieval_eval.py` | `reports/retrieval_paired_comparisons.json` | Mean paired differences and 95% bootstrap CIs |
| **Phase 2B: CPU Latency Profile** | `python src/run_retrieval_eval.py` | `reports/retrieval_latency.json` | p50, p95, mean latency, one-off index build timings |
| **Phase 2B: Ablation Grid** | `python src/run_retrieval_eval.py` | `reports/retrieval_ablation.json` | Filter and skill boost ablation combinations |
| **Phase 2B: Circularity Subset Audit** | `python src/run_retrieval_eval.py` | `reports/retrieval_circularity_check.json` | Zero-skill-overlap evaluation (51 pairs, 17 profiles) |
| **Phase 2C: LLM Explanation Quality** | `python src/run_explain_eval.py --full` | `reports/llm_quality.json` | Schema validity, baseline & adjusted groundedness, leakage |

---

## 2. Ingestion & Indexing Pipeline Benchmarks

### Ingestion & Cleaning Metrics
*(Data source: `reports/data_stats.json` | Command: `python src/ingest.py`)*

| Stage | Metric | Count / Value |
|---|---|---|
| Raw Pool | Downloaded Postings | 10,000 |
| Cleaned Pool | Valid Postings After Cleaning & Deduplication | 8,052 |
| Sampled Output | Final Sampled Dataset (`seed=42`) | 3,000 |
| Text Metric | Average Description Characters | 3637.38 chars |
| Text Metric | Average Description Words | 505.49 words |
| Text Metric | Median Words | 457 words |

### Experience Level Distribution
*(Data source: `reports/data_stats.json`)*

| Seniority Tier | Count | Percentage |
|---|---|---|
| Mid-Senior level | 1027 | 34.23% |
| Entry level | 984 | 32.80% |
| Not Specified | 746 | 24.87% |
| Director | 126 | 4.20% |
| Internship | 63 | 2.10% |
| Executive | 54 | 1.80% |

### Measured Indexing Performance
*(Data source: `reports/index_stats.json` | Command: `python src/index.py`)*

| Metric | Measured Value |
|---|---|
| Vector Store | ChromaDB (HNSW Cosine) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension | 384 |
| Total Documents Indexed | 3,000 |
| Index Build Time | 159.59 s |
| Indexing Throughput | 18.8 docs/sec |
| Metadata Fields | `job_id, title, company, location, experience_level, content_hash` |

### Incremental Synchronization vs Full Re-Index (Phase 1B)
*(Data source: `reports/sync_benchmark.json` | Command: `python src/bench_sync.py` | 5% modified, 2% added, 2% deleted)*

| Strategy | Mean Latency (3 runs) | Latency Std Dev | Effective Throughput |
|---|---|---|---|
| Full Re-Index (from scratch) | 159.636 s | 8.408 s | 18.79 docs/sec |
| Hash-Based Incremental Sync | **12.295 s** | 0.973 s | **244.0 docs/sec** |
| **Measured Speedup** | **12.98x** | — | — |

> [!NOTE]
> **Scope of Incremental Synchronization**: This benchmark measures incremental synchronization of the **ChromaDB dense vector index** specifically, where unchanged documents bypass embedding inference via SHA-256 content hashes. Because `rank_bm25` does not support incremental updates, the BM25 lexical index and tokenized corpus must be rebuilt in full (~6.0 s) whenever postings are added, modified, or removed. The production pipeline is therefore hybrid: dense vector updates are incremental, while lexical index updates require a complete rebuild.

---

## 3. Candidate Pooling & Relevance Distribution (Phase 2A)

### Candidate Pooling Strategy
To eliminate retrieval strategy bias, candidate pools were constructed per profile by taking the set union of the top 20 candidates retrieved by each of the four retrievers (BM25, Dense, Hybrid RRF, and Cross-Encoder). Across all 30 profiles, this yielded **1,037 unique candidate pairs** (mean 34.6 candidates per profile).

### Pooled Relevance Label Distribution
*Evaluated against model-proposed, unverified relevance labels*

| Relevance Grade | Description | Pair Count | Percentage |
|---|---|---|---|
| **0** | Irrelevant | 757 | 73.00% |
| **1** | Partially Relevant | 255 | 24.59% |
| **2** | Strongly Relevant | 25 | 2.41% |
| **Total** | Pooled Candidate Pool | 1,037 | 100.00% |

### Relevance Evaluation Status (`data/eval/qrels.json`)
*Evaluated against model-proposed, unverified relevance labels*

| Property | Status / Configuration | Reference Artifact |
|---|---|---|
| Relevance Source | **100% Rule-Based Proposed Labels** (`label_source: model_proposed`) | [`docs/LABELING.md`](docs/LABELING.md) |
| Total Judgments | 1,037 pairs across 30 candidate profiles | `data/eval/qrels.json` |
| Human Review Count | **0 pairs** (`human_reviewed: false`) | `reports/label_agreement.json` |
| Labeling Model / Engine | Deterministic Rubric Alignment (`jobmatch-rubric-v1 (Deterministic rule-based heuristic alignment)`) | `reports/labeling_method.json` |
| Quality Control Package | Blinded 277 candidate pairs intact | [`labels_to_review.csv`](labels_to_review.csv) |
| Review Documentation | Standalone rubric and candidate profile guide | [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md) |

---

## 4. Phase 2B: Retrieval Evaluation & Benchmarking

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
| **Cross-Encoder: Step 1 (Retrieval)** | 157.20 | 179.94 | 166.46 | Hybrid RRF candidate pool retrieval (top 50) |
| **Cross-Encoder: Step 2 (Neural Scoring)** | 2337.29 | 2622.19 | 2371.40 | CPU Transformer forward pass re-ranking 50 pairs |
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
> **Indicative Finding Only (51 pairs, 17 profiles)**: Evaluated on the subset of 51 relevant candidate postings across 17 profiles with zero literal skill matches (13 profiles have zero relevant items in this slice and are excluded from Recall/MRR means). Dense retrieval achieves Recall@10 of 0.3931 [0.23, 0.57] vs BM25 0.2588 [0.12, 0.43] and MRR of 0.3437 [0.15, 0.53] vs BM25 0.1751 [0.07, 0.31]. This indicates that dense vector retrieval can surface conceptually relevant postings when keyword markers are absent, but the small sample size and wide confidence intervals mean this is indicative only.

---

## 5. Phase 2C: LLM Explanation Quality & Groundedness Evaluation

### Architecture & Protocol
- **Objective**: Explain top-5 job matches from the best retrieval strategy (BM25 Only, nDCG@10 = 0.6147) for each candidate profile without relying on an LLM judge.
- **Model**: `gemini-3.1-flash-lite` via Google AI Studio API (Date checked: 2026-11-20, [Official Docs](https://ai.google.dev/gemini-api/docs/models)).
- **Prompt Version**: `explain-v1.0` with structured Pydantic schema validation:
  - `matched_skills: list[str]` (technical skills explicitly appearing in job text)
  - `missing_skills: list[str]` (qualifications required by job that candidate lacks)
  - `summary: str` (strictly enforced single-sentence rationale)
- **Retry & Recovery Protocol**: Exactly one repair prompt retry allowed on JSON/schema validation error. Persistent disk caching in `data/cache/explanations/`.
- **Evaluation Methodology**: 100% programmatic verification against raw job posting text. Zero LLM judge and zero human annotations.

### 1. Schema Validity & Fallback Robustness
*Programmatic checks against job text; no human or LLM judge*

| Metric | Observed Count | Measured Value | 95% Confidence / Upper Bound | Description |
|---|---|---|---|---|
| **JSON Valid First Try Rate** | 150/150 (0 failures) | **100.00%** | <= 2.00% failures (95% rule-of-three: <= 3/150) | Responses passing Pydantic schema on initial attempt |
| **Retries Needed** | 0/150 calls | **0.00%** | — | Total calls requiring repair prompt |
| **JSON Valid After Retry Rate** | 150/150 | **100.00%** | — | Cumulative validity after allowed single retry ($\ge$ first-try rate) |
| **Fallback Rate** | 0/150 calls | **0.00%** | <= 2.00% (95% rule-of-three: <= 3/150) | Calls returning fallback (`explanation=null`) |

> **Fallback Breakdown**: `invalid_json: 0`, `validation_error: 0`, `api_error: 0`, `timeout: 0`, `quota: 0` across all 150 calls.

### 2. Literal Groundedness & Profile Alignment
*Programmatic checks against job text; no human or LLM judge*

| Metric | Pooled Rate (Skills) | Macro Profile Average | 95% Cluster Bootstrap CI | Definition / Verification Scope |
|---|---|---|---|---|
| **Main Groundedness (Baseline)** | **91.70%** (254/277) | **91.79%** | [0.88, 0.96] | Boundary-safe regex `(?<![\w\+\#]){skill}(?![\w\+\#])` with symbol support (`C++`, `C#`, `.NET`) |
| **Main Groundedness (Adjusted for Joined Words)** | **94.22%** (261/277) | **94.31%** | [0.90, 0.98] | Recovers 7 skills joined without whitespace in raw HTML postings (e.g. `JavaScriptUnderstanding`, `TableauExperience`) |
| **Strict Groundedness (Exact Substring)** | **79.42%** (220/277) | **77.58%** | [0.69, 0.85] | Case-sensitive literal substring match without normalization |
| **Profile Support Rate** | **92.42%** | **93.06%** | [0.89, 0.97] | Claimed matched skills verified to exist in candidate profile |
| **Contradiction Leakage Rate** | **0.31%** (4/1278) | **0.36%** | [0.00, 0.01] | Self-contradiction check: skills claimed missing that appear in both job text and candidate profile |

> [!NOTE]
> **Scope of Contradiction Leakage**: Contradiction leakage measures internal self-contradiction only (instances where the model mistakenly claims a skill is missing despite it being present in both the candidate's profile and the job posting). It does not evaluate the overall completeness or correctness of the `missing_skills` list from the employer's perspective.

### 3. Operational Performance & Resource Utilization
*(Measured live over 125 API calls; 25 calls served from disk cache)*

| Metric | Measured Value | Operational Context |
|---|---|---|
| **Total Explanation Calls** | 150 calls across 30 candidate profiles | Top-5 BM25 matches per profile |
| **Cache Hit Rate** | 16.7% (25 hits, 125 live calls) | Persistent on-disk JSON cache (`data/cache/explanations/`) |
| **Live Call Latency (Mean)** | 3022.6 ms | Sub-second base model inference with backoff |
| **Live Call Latency (p95)** | 10332.5 ms | Tail latency under rate-controlled execution |
| **Prompt Tokens Consumed** | 113,032 tokens | Job description + candidate profile context |
| **Candidate Tokens Generated** | 15,563 tokens | Structured JSON explanations |
| **Total Tokens Consumed** | 128,595 tokens | Verified via API `usageMetadata` |

### 4. Qualitative Audit & Sample Explanations
A full qualitative inspection containing 10 randomly sampled explanations (seed = 42) and the 5 worst cases by literal groundedness is documented in [`reports/llm_samples.md`](reports/llm_samples.md).