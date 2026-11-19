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
| Relevance Source | **100% Model-Proposed** (`label_source: model_proposed`) | [`docs/LABELING.md`](docs/LABELING.md) |
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

### Key Observations:
- **Zero Skill Overlap in Relevant Candidates**: 51 out of 280 proposed relevant postings (18.21%) share **zero declared technical skills** with the candidate profile. These candidates were recognized as partially relevant through role/title keyword affinity (e.g., target role tokens like "Data Analyst" or "Frontend Developer" matching jobs where specific tech stack keywords were absent or described generically) or adjacent technical domain transitions.
- **Strong Fit Concentration**: 100% of the 25 strongly relevant (Label 2) postings contain at least one explicit skill match in addition to direct role and seniority alignment.
- **Irrelevant Postings**: 61.56% of irrelevant postings had zero technical skill matches; the remaining 38.44% contained incidental keyword mentions (e.g. "SQL" or "Python") in distant, unrelated, or senior roles.

---

## 6. Limitations

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance labels in `data/eval/qrels.json` are heuristic model proposals (`label_source: model_proposed`), with zero human evaluations conducted (`human_reviewed: false`). Evaluated retrieval metrics benchmark retriever alignment against rule-based heuristic criteria rather than human preference.
2. **Lexical Overlap Bias in Heuristic Rubric**: Because the heuristic proposal rubric incorporates technical skill regex and keyword matching, it inherently favors lexical retrieval methods (BM25 and hybrid combinations) over pure semantic vector retrieval. While the retrieval-independent audit identified 51 relevant candidates (18.21%) with zero skill overlap, 81.79% of relevant candidates share explicit skill keywords.
3. **Synthetic Test Profiles**: While all 3,000 job postings in `data/jobs_clean.parquet` are real LinkedIn postings, the 30 candidate profiles in `data/eval/profiles.json` are hand-authored synthetic benchmarks.
4. **Offline Benchmark Scope**: Evaluated in an offline batch setting without live user interaction, feedback loops, or click-through session tracking.
