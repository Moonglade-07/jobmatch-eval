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
- **Protocol**: Unjudged documents in top $k$ treated as non-relevant (label 0). Profiles with zero relevant documents (`prof_07`, `prof_23`) are explicitly excluded from Recall@10 and MRR means (evaluated over 28 profiles).
- **Statistical Significance**: Seeded empirical bootstrap 95% confidence intervals ($N=1,000$ iterations) for all metric macro-averages and paired differences.
- **In-Depth Analysis Report**: Comprehensive query case studies, statistical power analysis, and failure mode breakdowns are documented in [`reports/analysis_retrieval.md`](reports/analysis_retrieval.md).

### 1. Main Setting: Product Pipeline (Filters ON, Skill Boost ON $w=0.2$)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 | CPU p50 Latency | CPU p95 Latency |
|---|---|---|---|---|---|---|---|
| **BM25 Only** | 0.520 [0.41, 0.63] | 0.548 [0.47, 0.63] | 0.887 [0.79, 0.97] | 0.615 [0.52, 0.70] | 0.0% | 5450.5 ms | 5872.0 ms |
| **Dense Only (ChromaDB)** | 0.427 [0.31, 0.55] | 0.395 [0.31, 0.48] | 0.701 [0.56, 0.85] | 0.423 [0.33, 0.52] | 0.0% | 449.1 ms | 544.8 ms |
| **Hybrid RRF (k=60)** | 0.533 [0.42, 0.65] | 0.540 [0.47, 0.62] | 0.910 [0.82, 0.98] | 0.607 [0.52, 0.68] | 0.0% | 1249.4 ms | 1488.8 ms |
| **Hybrid + Cross-Encoder** | 0.527 [0.41, 0.63] | 0.539 [0.47, 0.62] | 0.864 [0.76, 0.95] | 0.593 [0.50, 0.67] | 0.0% | 3636.7 ms | 3843.8 ms |

> [!NOTE]
> All four strategies exhibit 0.00% unjudged rate in top 10 under the main configuration because candidate pooling in Phase 2A pooled the top 20 candidates across these exact four retrieval streams.

### 2. Secondary Setting: Pure Retrieval (Filters OFF, Skill Boost OFF)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 |
|---|---|---|---|---|---|
| **BM25 Only** | 0.127 [0.06, 0.20] | 0.142 [0.09, 0.20] | 0.254 [0.14, 0.38] | 0.162 [0.10, 0.23] | 85.3% |
| **Dense Only (ChromaDB)** | 0.140 [0.09, 0.20] | 0.138 [0.09, 0.20] | 0.295 [0.18, 0.44] | 0.161 [0.10, 0.23] | 76.0% |
| **Hybrid RRF (k=60)** | 0.147 [0.08, 0.23] | 0.141 [0.09, 0.20] | 0.293 [0.17, 0.45] | 0.177 [0.10, 0.26] | 82.3% |
| **Hybrid + Cross-Encoder** | 0.160 [0.09, 0.25] | 0.215 [0.15, 0.30] | 0.320 [0.20, 0.45] | 0.186 [0.13, 0.25] | 79.3% |

> [!IMPORTANT]
> When location and seniority filters are disabled, the unjudged rate surges to **76% – 85%**. Postings retrieved from out-of-market locations across the broader index were never pooled or judged, and are treated as non-relevant (label 0). This confirms that domain filtering is essential for precision in production job matching.

### 3. Paired Statistical Significance (Bootstrap 95% CIs on Differences)
*Evaluated against model-proposed, unverified relevance labels*

| Comparison ($A$ vs $B$) | Metric | Mean Difference ($A - B$) | 95% Bootstrap CI | Statistical Conclusion |
|---|---|---|---|---|
| **BM25 vs Dense** | `precision@5` | +0.0933 | [-0.0067, 0.1935] | No clear difference (CI spans 0.0) |
| **BM25 vs Dense** | `recall@10` | +0.1523 | [0.0388, 0.2735] | **Clear difference** |
| **BM25 vs Dense** | `mrr` | +0.1863 | [0.0640, 0.3027] | **Clear difference** |
| **BM25 vs Dense** | `ndcg@10` | +0.1916 | [0.1109, 0.2610] | **Clear difference** |
| **Hybrid RRF vs BM25** | `precision@5` | +0.0133 | [-0.0400, 0.0667] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs BM25** | `recall@10` | -0.0071 | [-0.0556, 0.0405] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs BM25** | `mrr` | +0.0230 | [-0.0366, 0.0884] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs BM25** | `ndcg@10` | -0.0080 | [-0.0534, 0.0352] | No clear difference (CI spans 0.0) |
| **Hybrid RRF vs Dense** | `precision@5` | +0.1067 | [0.0400, 0.1800] | **Clear difference** |
| **Hybrid RRF vs Dense** | `recall@10` | +0.1452 | [0.0538, 0.2519] | **Clear difference** |
| **Hybrid RRF vs Dense** | `mrr` | +0.2093 | [0.1033, 0.3179] | **Clear difference** |
| **Hybrid RRF vs Dense** | `ndcg@10` | +0.1836 | [0.1223, 0.2389] | **Clear difference** |
| **Cross-Encoder vs Hybrid RRF** | `precision@5` | -0.0067 | [-0.0733, 0.0600] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs Hybrid RRF** | `recall@10` | -0.0012 | [-0.0408, 0.0333] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs Hybrid RRF** | `mrr` | -0.0456 | [-0.1337, 0.0340] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs Hybrid RRF** | `ndcg@10` | -0.0140 | [-0.0579, 0.0311] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `precision@5` | +0.0067 | [-0.0800, 0.0800] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `recall@10` | -0.0083 | [-0.0571, 0.0458] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `mrr` | -0.0226 | [-0.1144, 0.0667] | No clear difference (CI spans 0.0) |
| **Cross-Encoder vs BM25** | `ndcg@10` | -0.0220 | [-0.0807, 0.0383] | No clear difference (CI spans 0.0) |

### 4. Query Latency Profile on CPU (25 timed runs per step)
*(Measured on AMD 12-core CPU, PyTorch CPU execution with `torch_threads=6`, `CUDA: False`)*

| Architecture / Step | p50 Latency (ms) | p95 Latency (ms) | Mean Latency (ms) | Details / Operations |
|---|---|---|---|---|
| **Dense Only** | 449.15 | 544.77 | 453.1 | Query embedding (`all-MiniLM-L6-v2`) + ChromaDB HNSW |
| **Hybrid RRF ($k=60$)** | 1249.39 | 1488.81 | 1249.79 | BM25 + Dense retrieval + RRF fusion |
| **BM25 Only** | 5450.48 | 5872.0 | 5441.48 | Lexical BM25Okapi scoring over full corpus |
| **Cross-Encoder: Step 1 (Retrieval)** | 1236.08 | 1501.1 | 1248.36 | Hybrid RRF candidate pool retrieval (top 50) |
| **Cross-Encoder: Step 2 (Neural Scoring)** | 2405.34 | 2574.84 | 2391.62 | CPU Transformer forward pass re-ranking 50 pairs |
| **Cross-Encoder: End-to-End Pipeline** | **3636.65** | **3843.8** | **3639.98** | Total two-stage retrieval and neural re-ranking |


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

### 6. Circularity Audit: Relevant Subset with Zero Literal Skill Overlap
*Evaluated against model-proposed, unverified relevance labels*

Subset consists of **51 relevant candidate postings across 17 profiles** that share zero literal keyword overlap with candidate declared skills (13 profiles had zero such postings and are excluded from Recall/MRR means):

| Strategy | Precision@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| **BM25 Only** | 0.0333 | 0.2588 | 0.1751 | 0.0932 |
| **Dense Only (ChromaDB)** | 0.0600 | 0.3931 | 0.3437 | 0.1383 |
| **Hybrid RRF (k=60)** | 0.0800 | 0.3180 | 0.2319 | 0.1124 |
| **Hybrid + Cross-Encoder** | 0.0467 | 0.2552 | 0.3077 | 0.1098 |

> [!TIP]
> When literal skill keywords are removed, **Dense Only outperforms BM25 Only** (+51.9% relative Recall@10, +96.3% relative MRR, and +48.4% relative nDCG@10). This directly confirms that dense embeddings discover semantic relevance when keyword markers are absent.

---

## 7. Limitations

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance labels in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`), with zero human evaluations conducted (`human_reviewed: false`). Evaluated retrieval metrics benchmark retriever alignment against rule-based heuristic criteria rather than human preference.
2. **Lexical Overlap Bias in Heuristic Rubric**: Because the heuristic proposal rubric incorporates technical skill regex and keyword matching, it inherently favors lexical retrieval methods (BM25 and hybrid combinations) over pure semantic vector retrieval. While the retrieval-independent audit identified 51 relevant candidates (18.21%) with zero skill overlap, 81.79% of relevant candidates share explicit skill keywords.
3. **Synthetic Test Profiles**: While all 3,000 job postings in `data/jobs_clean.parquet` are real LinkedIn postings, the 30 candidate profiles in `data/eval/profiles.json` are hand-authored synthetic benchmarks.
4. **Offline Benchmark Scope**: Evaluated in an offline batch setting without live user interaction, feedback loops, or click-through session tracking.
