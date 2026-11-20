# Job-Matching Search with Evaluation

An end-to-end, reproducible ML portfolio project building and comparing retrieval strategies (dense semantic search, lexical BM25, hybrid retrieval, and reciprocal rank fusion) for matching candidate profiles against job postings, accompanied by grounded LLM explanations and rigorous evaluation metrics.

---

## 1. System Architecture & Dataset Provenance

### Dataset Provenance & Attribution
- **Dataset Name**: LinkedIn Job Postings (2023-2024)
- **Source URL**: [https://huggingface.co/datasets/datastax/linkedin_job_listings](https://huggingface.co/datasets/datastax/linkedin_job_listings)
- **Licence**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) / MIT
- **Dataset Size**: 10,000 raw downloaded postings, cleaned and deduplicated to 8,052 valid postings, with a deterministic sample of 3,000 postings (`seed=42`) used for indexing and benchmarking.
- **Integrity**: 100% real LinkedIn postings; no synthetic job descriptions generated.

### Retrieval Architectures
1. **BM25 Only (`BM25Retriever`)**: Lexical keyword search using `rank_bm25.BM25Okapi` with domain-aware regex tokenization preserving technical symbols (`c++`, `c#`, `.net`, `python-3`).
2. **Dense Only (`DenseRetriever`)**: Semantic dense vector search in ChromaDB using `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions) and HNSW cosine distance.
3. **Hybrid RRF (`HybridRRFRetriever`)**: Reciprocal Rank Fusion combining lexical and dense streams with parameter $k_{rrf} = 60$:
   $$RRF(d) = \sum_{s \in \{bm25, dense\}} \frac{1}{60 + \text{rank}_s(d)}$$
4. **Hybrid + Cross-Encoder Re-Rank (`CrossEncoderRetriever`)**: Two-stage pipeline taking the top 50 candidates from Hybrid RRF and re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention and sigmoid normalization.

### Conversational Career Agent (Phase 3A)
- **Architecture**: Multi-turn conversational agent (`src/agent.py`) leveraging Gemini Function Calling (`gemini-3.1-flash-lite`) with 3 tools: `search_jobs`, `get_job_details`, and `refine_filters`.
- **Underlying Retriever Selection**: The agent uses BM25 because it scored highest on nDCG@10 against rule-based labels that share its signals; the four retrieval strategies were **not compared inside the agent**.
- **Operational Bounds**: Persistent session state on disk (`data/sessions/{session_id}.json`), strict 5-step loop cap per turn (`MAX_STEPS_PER_TURN = 5`), candidate dislike filtering, and disk response caching (`data/cache/agent/`).

### Personalization & Filtering Pipeline
- **Profile Query Synthesis**: Structured query generation combining target role, declared technical skills, seniority tier, and narrative resume text.
- **Three-Tier Fallback Hierarchy**:
  - **Tier 1 (Strict)**: Postings matching candidate preferred location (allowing "Remote") and seniority (allowing "Not Specified").
  - **Tier 2 (Relaxed Seniority)**: If Tier 1 < $k$, relax seniority tier while preserving location constraint.
  - **Tier 3 (Global Backfill)**: If Tier 2 < $k$, backfill remaining slots from top unfiltered rankings to guarantee $k$ results.
- **Explainable Skill Boosting**:
  $$\text{final\_score} = \text{base\_score} + w \times \frac{|\text{matched\_skills}|}{|\text{candidate\_skills}|}$$
  *(default weight $w = 0.2$)*. Word-boundary regex matching `(?<![\w\+\#]){skill}(?![\w\+\#])` prevents substring false-positives while preserving symbols like C++, C#, and .NET.

### Relevance Evaluation & Label Provenance
- **Candidate Pooling**: Top 20 candidates retrieved across all four retrieval strategies (BM25, Dense, Hybrid RRF, Cross-Encoder) and pooled via set union per profile (**1,037 total pairs across 30 candidate profiles**).
- **Labeling Methodology**: Relevance labels were produced by a deterministic rule-based heuristic rubric (`jobmatch-rubric-v1`) based on title-token overlap, technical skill regex, location, and seniority criteria; unverified by humans.
- **Human Verification Rate**: **0% verified by humans** (`human_reviewed: false`, `human_verified_count: 0`).
- **Disagreement Rate & Inter-Annotator Agreement**: Inter-annotator agreement metrics (disagreement rate, Cohen's kappa) are withheld because human review was skipped and no human labels were submitted.
- **Methodological Disclosure**: Because the rule-based labeling heuristic incorporates keyword matching and skill regex, lexical retrieval methods (BM25 and hybrid combinations) and the skill boost may be favored because the labels share their signals. High BM25 scores reflect shared lexical signals with the rubric, not evidence that BM25 is the better retriever in absolute terms.

---

<!-- RESULTS:START -->
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
| **Measured Speedup (Dense Index Only)** | **12.98x** (13x faster dense-index sync: 5% modified, 2% added, 2% deleted) | — | — |

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

---

## 6. Phase 3A: Conversational Agent & API Evaluation

### Multi-Turn Scripted Scenarios Evaluation
*Programmatic checks against scenario constraints; no human or LLM judge*

| Metric | Measured Value | Operational Context |
|---|---|---|
| **Evaluated Model** | `gemini-3.1-flash-lite` | Gemini Function Calling (`tools`) |
| **Total Scenarios** | 10 | Scripted multi-turn scenarios (28 total turns) |
| **Behavioral Scenarios Passed** | **9/10 (90.0%)** | Satisfied all behavioral assertions across multi-turn interactions |
| **First-Pass Scenarios Passed** | **8/10 (80.0%)** | Passed on initial run without infrastructure failures |
| **Infrastructure Failures** | **1** (`scenario_08` Attempt 1) | Transient upstream API failure (`Gemini API request failed after retries`) |
| **Behavioral Failures** | **1** (`scenario_10`) | Pivoted to Remote query when 0 Chicago jobs found; Mid-Senior applied via memory |
| **Tool Call Success Rate** | **96.9%** (31/32 tool calls) | 1 expected failure in `scenario_06` testing non-existent job ID |
| **Memory Retention Rate** | **100.0%** (23/23 memory checks) | Constraints preserved across distracting/follow-up turns |
| **Mean Tool Steps per Turn** | **2.14 steps** | Capped strictly at 5 steps per turn |
| **Mean Turn Latency** | **10828.3 ms** | End-to-end turn latency (live + cached) |
| **Max Turn Latency** | **41697.0 ms** | Worst-case single turn latency |
| **Cache Hit Rate** | 33.3% (1 hits, 2 live calls) | Persistent on-disk JSON cache (`data/cache/agent/`) |

### Scenario Breakdown
*Programmatic checks against scenario constraints; no human or LLM judge*

| Scenario ID | Name | Turns | Status | Tools Called | Failure / Inspection Notes |
|---|---|---|---|---|---|
| `scenario_01` | location_persistence | 3 | **PASS** | `get_job_details`, `search_jobs` | — |
| `scenario_02` | location_update_remote | 3 | **PASS** | `search_jobs` | — |
| `scenario_03` | seniority_exclusion_persistence | 3 | **PASS** | `refine_filters`, `search_jobs` | — |
| `scenario_04` | dislike_exclusion | 3 | **PASS** | `refine_filters`, `search_jobs` | — |
| `scenario_05` | vague_request_no_hallucination | 2 | **PASS** | `search_jobs` | — |
| `scenario_06` | tool_error_handling | 2 | **PASS** | `get_job_details`, `search_jobs` | — |
| `scenario_07` | explicit_filter_refinement | 2 | **PASS** | `refine_filters`, `search_jobs` | — |
| `scenario_08` | skills_persistence_across_turns | 3 | **PASS** (Attempt 2) | `refine_filters`, `search_jobs` | Attempt 1: Infrastructure failure (Turn 2: Turn crashed with error: Gemini API request failed after retries.). Attempt 2: Passed with full skills persistence. |
| `scenario_09` | job_inspection_then_location_switch | 3 | **PASS** | `get_job_details`, `search_jobs` | — |
| `scenario_10` | multi_constraint_compounding | 4 | **FAIL** | `refine_filters`, `search_jobs` | Turn 4: Tool 'search_jobs' was not called with argument 'location' containing 'Chicago, IL'.; Turn 4: Tool 'search_jobs' was not called with argument 'level' containing 'Mid-Senior level'.; Inspection note: In Turn 4, the model retained 'Chicago, IL' and 'Mid-Senior level' in session memory. However, because earlier turns returned zero Chicago postings in this 3,000-job sample, the model pivoted to remote postings by passing location='Remote' to search_jobs, while Mid-Senior level was applied through session memory (as seen by mid-senior postings returned). The scenario failed strictly because the programmatic assertion demanded explicit tool arguments: location='Chicago, IL' and level='Mid-Senior level'. |

<!-- RESULTS:END -->

---

## 6. Analysis Summary

### Ingestion & Incremental Synchronization (Phase 1A & 1B)
- **Dense Vector Sync Speedup**: Hash-based incremental synchronization of the ChromaDB dense vector index achieved **13x faster dense-index sync (5% modified, 2% added, 2% deleted)** (12.30 s vs 159.64 s full re-index; 12.98x), while preserving 100% state equivalence (0 hash discrepancies). This applies exclusively to the dense vector store; the lexical BM25 index must be rebuilt in full (~6.0 s) on any change.
- **BM25 Lexical Index Asymmetry**: Because `rank_bm25` lacks an incremental update API, the lexical index and tokenized corpus must be rebuilt in full (~6.0 s) on every data mutation. The operational system is hybrid: dense vectors update incrementally, while lexical indices require complete rebuilds.

### Retrieval Performance & Signal Alignment (Phase 2B)
- **BM25 Alignment vs Vector Retrieval**: Under the main product pipeline (Filters ON, Skill Boost ON $w=0.2$), BM25 achieved an nDCG@10 of 0.6147 [0.52, 0.70] and MRR of 0.8869 [0.79, 0.97], compared to Dense retrieval nDCG@10 of 0.4231 [0.33, 0.52] and MRR of 0.7006 [0.56, 0.85].
- **Signal Overlap Disclosure**: The superior ranking of BM25 against the evaluation set is directly attributable to the fact that the rule-based labeling rubric (`jobmatch-rubric-v1`) shares lexical signals (token overlap and skill regex) with BM25. This demonstrates retriever-rubric alignment, not absolute retriever superiority for human job seekers.
- **Pure Retrieval Collapse**: When location and seniority filters were disabled, unjudged rates surged to 76.0%–85.3% across all strategies. Because candidate pooling was performed under active filters, out-of-market jobs were never judged (treated as non-relevant label 0), rendering pure retrieval metrics incomparable to the main setting.
- **Circularity Audit on Zero-Skill Overlap (51 pairs, 17 profiles)**: On the subset of 51 relevant candidate postings with zero literal skill keyword matches, Dense retrieval achieved higher Recall@10 (0.3931 [0.23, 0.57] vs 0.2588 [0.12, 0.43]) and higher MRR (0.3437 [0.15, 0.53] vs 0.1751 [0.07, 0.31]) than BM25. This indicates semantic vector capabilities when keywords are missing, but the small sample size (17 profiles) means this finding is indicative only.

### LLM Explanation Quality (Phase 2C)
- **Schema Reliability**: Across all 150 explanation calls (30 profiles $\times$ top-5 BM25 matches) using `gemini-3.1-flash-lite`, the system achieved **100.00% first-try schema validity** (0 retries needed, 0 fallbacks), establishing a 95% failure upper bound of $\le 2.00\%$ via the rule-of-three ($\le 3/150$).
- **Groundedness Transparency**:
  - **Baseline Groundedness**: 91.70% pooled (254/277 skills) and 91.79% macro-average [0.88, 0.96] under standard boundary-safe regex.
  - **Adjusted Groundedness**: 94.22% pooled (261/277 skills) and 94.31% macro-average [0.90, 0.98] after accounting for 7 skills joined without whitespace in raw HTML postings (e.g. `JavaScriptUnderstanding`, `TableauExperience`).
  - **Truly Ungrounded Claims**: Exactly 16 claimed skills (5.78%) were absent from the job text, primarily involving high-level conceptual inferences (e.g. `PowerBI`, `Strategic Planning`, `Budgeting`, `Roadmapping`).
- **Profile Support & Contradiction Leakage**: 92.42% of claimed matched skills were verified against candidate profiles. Contradiction leakage (claiming a skill as missing when it appears in both the job description and candidate profile) was restricted to **0.31%** (4/1,278 missing skills).

---

## 7. Limitations

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance labels in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`), generated via deterministic heuristics (`jobmatch-rubric-v1`) and unverified by humans (`human_reviewed: false`). Evaluated retrieval metrics reflect alignment with deterministic rule-based criteria rather than human preference.
2. **Lexical Overlap Bias in Heuristic Rubric**: Because the heuristic proposal rubric incorporates technical skill regex and keyword matching, it inherently favors lexical retrieval methods (BM25 and hybrid combinations) over pure semantic vector retrieval. High BM25 scores reflect shared lexical signals with the rubric, not evidence that BM25 is the better retriever in absolute terms.
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
9. **Literal Groundedness vs Contextual Correctness (Phase 2C)**: Groundedness evaluates literal presence in job descriptions via boundary-safe regex, not semantic suitability or contextual sentiment. A skill mentioned in an exclusionary clause (e.g., "no PHP required") would still register as literally grounded.
10. **Contradiction Leakage Scope (Phase 2C)**: Contradiction leakage measures internal self-contradiction only (skills claimed as missing that appear in both the job text and candidate profile). It does not assess the general completeness or relevance of the `missing_skills` list from an employer standpoint.
11. **Prompt and Model Version Specificity (Phase 2C)**: LLM explanation outputs, schema validity, and groundedness metrics reflect `gemini-3.1-flash-lite` using prompt `explain-v1.0`; alternative models or prompt iterations will produce different performance profiles.
12. **Programmatic Verification Without LLM or Human Judge (Phase 2C)**: All explanation quality signals were audited via deterministic string and regex checks; no qualitative human evaluation or secondary LLM judge was utilized.
13. **Free-Tier Rate Limits & Quotas (Phase 2C)**: High-throughput execution without billing requires rate-controlled pacing and persistent disk caching (`data/cache/explanations/`) to prevent rate-limit exhaustion under Google AI Studio project quotas.
14. **Agent Retrieval Strategy Scope (Phase 3A)**: The conversational agent integrates `BM25Retriever` exclusively because it scored highest on nDCG@10 against rule-based proposed labels (`jobmatch-rubric-v1`). However, because those labels share lexical signals with BM25, this choice reflects heuristic alignment rather than absolute retrieval superiority. The four retrieval strategies (BM25, Dense, Hybrid RRF, Cross-Encoder) were not compared inside the conversational agent.
15. **Offline Scripted Agent Evaluation & Sample Size (Phase 3A)**: The conversational agent's multi-turn capabilities and memory retention are evaluated programmatically across 10 hand-authored multi-turn scenarios (28 total turns) rather than live human interactions. This small scripted sample size limits statistical generalization and does not capture open-domain user dialogue variability.
16. **Agent Latency & Single Model Scope (Phase 3A)**: Conversational turn latency averages about 10 s (observed mean ~10.8 s), with worst-case turns reaching ~37 s to ~42 s during multi-step tool execution loops, backoff retries, and network delays. Furthermore, evaluation was conducted exclusively on a single model (`gemini-3.1-flash-lite`); performance across alternative models or parameter scales has not been benchmarked.


---

## 8. Project Structure

```text
job-search-project/
├── .env.example              # Environment variables template
├── .dockerignore             # Excluded files for Docker image build
├── .gitignore                # Git ignore rules (ignoring data/, chroma_db/, .venv/)
├── app.py                    # Streamlit interactive search and agent demonstration
├── DECISIONS.md              # Architectural decisions, trade-offs, and justifications
├── Dockerfile                # Reproducible container configuration
├── Makefile                  # Build and execution targets
├── pyproject.toml            # Pytest configuration and project metadata
├── README.md                 # System overview and measured results table
├── requirements.txt          # Pinned dependency manifest
├── RESULTS.md                # Provenance ledger with commands and source files
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
│   ├── CLAIMS.md             # Supported vs unsupported empirical claims
│   ├── INTERVIEW_QA.md       # Honest technical interview questions and answers
│   ├── LABELING.md           # Fixed 3-point relevance labeling rubric
│   └── REVIEW_GUIDE.md       # Self-contained review guide with profile summaries
├── reports/                  # Verified execution reports (JSON)
│   ├── analysis_retrieval.md # Comprehensive Phase 2B retrieval evaluation analysis
│   ├── data_stats.json       # Ingestion and data cleaning statistics
│   ├── index_stats.json      # Vector indexing benchmark statistics
│   ├── labeling_method.json  # Phase 2A labeling method, rubric, model, and prompt
│   ├── label_agreement.json  # Documents unverified status (human_reviewed: false)
│   ├── llm_outputs.jsonl     # Phase 2C logged explanations (cache hits, latency, tokens)
│   ├── llm_quality.json      # Phase 2C schema validity, groundedness, leakage metrics
│   ├── llm_samples.md        # Phase 2C qualitative samples (10 random + 5 worst cases)
│   ├── relevant_docs_per_profile.json # Relevant doc counts per candidate profile
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
│   ├── explain.py            # Phase 2C LLM explanation generation & groundedness evaluator
│   ├── generate_eval_profiles.py # Phase 2A profile generation & Tier 1 verification
│   ├── generate_readme.py    # Auto-generation of results table from reports
│   ├── index.py              # ChromaDB vector indexing and embeddings
│   ├── ingest.py             # Data loading, cleaning, deduplication, sampling
│   ├── label_agreement.py    # Agreement evaluation (strictly requires labels_reviewed.csv)
│   ├── make_readme_table.py  # Results tables injector (no hand-typed numbers)
│   ├── metrics.py            # Precision@k, Recall@k, MRR, nDCG@k evaluation metrics
│   ├── models.py             # Profile and SearchResult domain schemas
│   ├── personalise.py        # Query synthesis, 3-tier filtering fallback, skill boost
│   ├── pool_and_label.py     # Candidate pooling (union top 20 across 4 strategies) & labeling
│   ├── prepare_blinded_review.py # Prepares blinded review file and decoupled key
│   ├── retrieval.py          # Unified retrieval: BM25, Dense, Hybrid RRF, Cross-Encoder
│   ├── run_explain_eval.py   # Phase 2C pipeline runner and CLI evaluator
│   ├── run_retrieval_eval.py # Phase 2B full benchmark, ablation, and latency runner
│   └── sync_index.py         # Hash-based incremental upsert/delete pipeline
└── tests/                    # Comprehensive unit tests
    ├── test_explain.py       # Phase 2C explanation validation & groundedness unit tests
    ├── test_index.py         # Embedding, hashing, and ChromaDB unit tests
    ├── test_ingest.py        # Cleaning, HTML stripping, dedupe, empty row tests
    ├── test_label_agreement.py # Agreement, Cohen's kappa, and review guardrail tests
    ├── test_metrics.py       # P@k, Recall@k, MRR, nDCG@k unit tests with manual cases
    ├── test_personalise.py   # Skill extraction, boost reordering, 3-tier fallback tests
    ├── test_retrieval.py     # RRF hand-computed verification, BM25, Hybrid, Cross-Encoder tests
    └── test_sync.py          # Incremental sync unit tests (mock embedder)
```

---

## 9. Quickstart & Reproducibility

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### One-Command Pipeline Execution
To execute the end-to-end pipeline (`ingest -> index -> sync benchmark -> retrieval eval -> LLM eval -> make README table -> test`):
- **On Linux/macOS**: `./run_all.sh` or `make all`
- **On Windows**: `run_all.bat`
- **Via Docker**:
  ```bash
  docker build -t job-search .
  docker run --rm job-search
  ```

### Individual Step Execution
```bash
# 1. Ingestion, cleaning, deduplication, and sampling
python src/ingest.py

# 2. Dense vector indexing in ChromaDB
python src/index.py

# 3. Benchmark incremental synchronization (Phase 1B)
python src/bench_sync.py

# 4. Phase 2B full retrieval evaluation across all strategies
python src/run_retrieval_eval.py

# 5. Phase 2C LLM explanation evaluation (cached full run)
python src/run_explain_eval.py --full

# 6. Inject generated results tables into README.md & generate RESULTS.md
python src/make_readme_table.py

# 7. Execute test suite
pytest -v tests/
```

---

## 10. Run locally

> [!IMPORTANT]
> **Deployment Status**: This project is **NOT deployed** to any cloud host and has **no live URL**. All benchmarks, interactive interfaces, REST APIs, and containerized workflows run entirely locally on your machine.

### 1. Clone & Environment Configuration
```bash
# Clone the repository
git clone https://github.com/Moonglade-07/jobmatch-eval.git
cd jobmatch-eval

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (free-tier key from Google AI Studio: ai.google.dev)
```

### 2. One-Command Pipeline Execution
To execute the entire reproducible pipeline (`ingest -> index -> sync benchmark -> retrieval eval -> LLM eval -> make README table -> test`):
- **On Linux/macOS**: `./run_all.sh` or `make all`
- **On Windows**: `run_all.bat`

### 3. Interactive Streamlit Application
Launch the dual-interface web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
- **Search Tab**: Profile form (target role, location, seniority, skills, resume snippet), strategy selector (BM25, Dense, Hybrid RRF, Cross-Encoder), and grounded Gemini match explanations with visible fallback banners.
- **Conversational Agent Tab**: Multi-turn job search assistant backed by `JobSearchAgent` with persistent session memory and real-time tool inspection.
- **Local API Key Safeguards**: Built-in rate limits (20 requests per session, 200 requests per day) protect your personal Google AI Studio free-tier API key from unintended quota exhaustion or runaway loops.

### 4. FastAPI REST API Service
Launch the backend REST API:
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.
- `POST /search`: Ranked job retrieval with optional grounded match explanations.
- `POST /chat`: Multi-turn conversational search turn with persistent memory.
- `GET /health`: System health and dynamic index length check.

### 5. Local Docker Container Build & Run
Build and run the self-contained container locally on port 8501:
```bash
# Build self-contained Docker image (bakes Chroma index inside image at build time)
docker build -t jobmatch-ai .

# Run container locally with your Gemini API key
docker run -p 8501:8501 -e GEMINI_API_KEY="your-key-here" jobmatch-ai
```

### 6. Local Memory & Cold-Start Measurements
Measured locally on single-process CPU execution (AMD 12-core, Windows):
- **Cold-Start Time**: **21.26 s** (BM25 + Dense + Hybrid RRF), or **23.36 s** with Cross-Encoder.
- **RAM Footprint (RSS)**:
  - BM25: 319.7 MB (+301.8 MB for tokenized corpus)
  - Dense (ChromaDB + `all-MiniLM-L6-v2`): 811.4 MB (+491.7 MB for vector database & bi-encoder weights)
  - Combined Pipeline (BM25 + Dense + Hybrid RRF): **811.4 MB** steady-state RAM.
  - Cross-Encoder Re-Ranker (`ms-marco-MiniLM-L-6-v2`): 904.6 MB (+93.2 MB model weights), peaking at ~1.1 GB+ during 50-candidate cross-attention passes.
- **Local Resource Knobs**: On memory-constrained local systems, set `DISABLE_CROSS_ENCODER=1` to disable the cross-encoder and cap RAM at ~811 MB, defaulting retrieval to Hybrid RRF.
- **Dataset Attribution**: 3,000 clean sample postings licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Code licensed under MIT. No live user base, production traffic, or commercial usage is claimed.

