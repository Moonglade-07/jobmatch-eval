# Experimental Results & Retrieval Evaluation Benchmark

An empirical benchmark comparing retrieval architectures, candidate pooling, personalization strategies, and relevance evaluation for matching job seekers against real job postings.

---

## 1. Retrieval Strategies Benchmark Overview

Four retrieval strategies are evaluated across 30 test candidate profiles spanning diverse engineering domains, seniority tiers, and edge cases (such as career changers and vague profiles):

1. **BM25 Only (`BM25Retriever`)**: Lexical keyword search using BM25Okapi with technical tokenization (`c++`, `c#`, `.net`, `python-3`).
2. **Dense Only (`DenseRetriever`)**: Semantic dense vector search in ChromaDB using `all-MiniLM-L6-v2` embeddings and HNSW cosine distance.
3. **Hybrid RRF (`HybridRRFRetriever`)**: Reciprocal Rank Fusion ($k_{rrf}=60$) combining reciprocal ranks from BM25 and Dense streams.
4. **Hybrid + Cross-Encoder (`CrossEncoderRetriever`)**: Two-stage pipeline taking the top 50 candidates from Hybrid RRF and re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention scores.

---

## 2. Ingestion & Indexing Pipeline Benchmarks

### Ingestion & Cleaning Metrics
*(Data source: `reports/data_stats.json`)*

| Metric | Measured Value |
|---|---|
| Raw Downloaded Postings | 3,365 |
| Postings After Cleaning & Deduplication | 3,000 |
| Final Sampled Postings (`seed=42`) | 3,000 |
| Average Description Length | 1,463.3 characters (219.8 words) |
| Median Description Words | 196 words |

### Dense Vector Indexing Benchmark
*(Data source: `reports/index_stats.json`)*

| Metric | Measured Value |
|---|---|
| Vector Database | ChromaDB (HNSW Cosine) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension | 384 |
| Total Indexed Postings | 3,000 |
| Index Build Time | 30.15 seconds |
| Indexing Throughput | 99.49 docs/sec |

### Incremental Synchronization vs Full Re-Index (Phase 1B)
*(Data source: `reports/sync_benchmark.json` | Simulated mutation: 5% modified, 2% added, 2% deleted)*

| Strategy | Mean Latency (3 runs) | Latency Std Dev | Effective Throughput |
|---|---|---|---|
| Full Re-Index (from scratch) | 30.82 s | 0.45 s | 97.3 docs/sec |
| Hash-Based Incremental Sync | **3.88 s** | 0.12 s | **773.2 docs/sec** |
| **Measured Speedup** | **7.94x** | — | — |

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

---

## 4. Relevance Evaluation Status (`data/eval/qrels.json`)
*Evaluated against model-proposed, unverified relevance labels*

| Property | Status / Configuration | Reference Artifact |
|---|---|---|
| Relevance Source | **100% Rule-Based Proposed Labels** (`label_source: model_proposed`) | [`docs/LABELING.md`](docs/LABELING.md) |
| Total Judgments | 1,037 pairs across 30 candidate profiles | `data/eval/qrels.json` |
| Human Review Count | **0 pairs** (`human_reviewed: false`) | `reports/label_agreement.json` |
| Labeling Model / Engine | Deterministic Rubric Alignment (`jobmatch-rubric-v1`) | `reports/labeling_method.json` |
| Quality Control Package | Blinded 277 candidate pairs intact | [`labels_to_review.csv`](labels_to_review.csv) |
| Review Documentation | Standalone rubric and candidate profile guide | [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md) |

---

## 5. Retrieval-Independent Lexical Overlap Analysis

Because the heuristic labeling rubric incorporates technical skill regex and keyword matching, a retrieval-independent analysis was performed to measure the exact degree of lexical skill overlap across all proposed relevance tiers:

*Evaluated against model-proposed, unverified relevance labels*

| Relevance Grade | Total Candidates | Zero Skill Overlap | $\ge 1$ Skill Overlap | % with Zero Skill Overlap |
|---|---|---|---|---|
| **Label 0 (Irrelevant)** | 757 | 466 | 291 | 61.56% |
| **Label 1 (Partially Relevant)** | 255 | 51 | 204 | 20.00% |
| **Label 2 (Strongly Relevant)** | 25 | 0 | 25 | 0.00% |
| **Relevant Candidates (1 or 2)** | **280** | **51** | **229** | **18.21%** |

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

## Phase 2C: LLM Explanation Quality & Groundedness Evaluation

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
| **Main Groundedness (Boundary Regex)** | **91.70%** (254/277) | **91.79%** | [0.88, 0.96] | Boundary-safe regex `(?<![\w\+\#]){skill}(?![\w\+\#])` with symbol support (`C++`, `C#`, `.NET`) |
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

---

## 7. Limitations

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance labels in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`), with zero human evaluations conducted (`human_reviewed: false`). Evaluated retrieval metrics benchmark retriever alignment against rule-based heuristic criteria rather than human preference.
2. **Lexical Overlap Bias in Heuristic Rubric**: Because the heuristic proposal rubric incorporates technical skill regex and keyword matching, it inherently favors lexical retrieval methods (BM25 and hybrid combinations) over pure semantic vector retrieval. High BM25 scores reflect shared lexical signals with the rubric, not evidence that BM25 is the better retriever in absolute terms.
3. **Retrieval-Independent Check is Indicative Only (51 pairs, 17 profiles)**: When evaluated on the 51 relevant candidates with zero literal skill matches, dense retrieval achieves higher recall and MRR than BM25, indicating semantic capability when keywords are missing. However, because only 17 profiles have any relevant documents in this slice, confidence intervals are wide and the result is indicative only.
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
9. **Literal Groundedness vs Contextual Correctness (Phase 2C)**: Groundedness evaluates literal presence in job descriptions via boundary-safe regex, not semantic suitability or contextual sentiment. A skill mentioned in an exclusionary clause would still register as literally grounded.
10. **Contradiction Leakage Scope (Phase 2C)**: Contradiction leakage measures self-contradiction only (skills claimed as missing that appear in both the job text and candidate profile). It does not assess the general completeness or relevance of the `missing_skills` list from an employer standpoint.
11. **Prompt and Model Version Specificity (Phase 2C)**: LLM explanation outputs, schema validity, and groundedness metrics reflect `gemini-3.1-flash-lite` using prompt `explain-v1.0`; alternative models or prompt iterations will produce different performance profiles.
12. **Programmatic Verification Without LLM or Human Judge (Phase 2C)**: All explanation quality signals were audited via deterministic string and regex checks; no qualitative human evaluation or secondary LLM judge was utilized.
13. **Offline Benchmark Scope**: Evaluated in an offline batch setting without live user interaction, feedback loops, or click-through session tracking.
