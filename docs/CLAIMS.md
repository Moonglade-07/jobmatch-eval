# Supported vs. Unsupported Claims

This document provides a transparent, empirical breakdown of claims that are supported by measured data in `jobmatch-eval` versus claims that are unsupported, misleading, or methodologically unproven.

Every claim is cross-referenced with exact numbers, files in `reports/`, and documented experimental constraints.

---

## 1. Supported Claims (Empirically Verified)

### Claim 1.1: Incremental synchronization of dense vectors achieves a 13x speedup over full re-indexing.
- **Status**: **SUPPORTED BY EMPIRICAL EVIDENCE**
- **Measured Number & File**:
  - Full re-index latency: **159.636 s** $\pm$ 8.408 s (18.79 docs/sec).
  - Incremental sync latency: **12.295 s** $\pm$ 0.973 s (244.0 effective docs/sec).
  - Measured speedup: **12.98x** (13x faster dense-index sync: 5% modified, 2% added, 2% deleted) (`reports/sync_benchmark.json`).
  - State equivalence: Exactly 0 hash discrepancies across 3,000 documents.

### Claim 1.2: Structured Pydantic validation delivers 100% first-try schema validity.
- **Status**: **SUPPORTED BY EMPIRICAL EVIDENCE**
- **Measured Number & File**:
  - First-try valid rate: **100.00%** (150/150 valid calls, 0 retries, 0 fallbacks) (`reports/llm_quality.json`).
  - Rule-of-three 95% upper bound on failure rate: **$\le 2.00\%$** ($\le 3/150$).

### Claim 1.3: Claimed matched skills in LLM explanations exceed 90% literal groundedness against job postings.
- **Status**: **SUPPORTED BY EMPIRICAL EVIDENCE**
- **Measured Number & File**:
  - Baseline groundedness: **91.70% pooled** (254/277 skills) and **91.79% macro-average** [0.88, 0.96] (`reports/llm_quality.json`).
  - Adjusted groundedness (recovering 7 joined-word HTML artifacts): **94.22% pooled** (261/277 skills) and **94.31% macro-average** [0.90, 0.98] (`reports/llm_quality.json`).
  - Profile support: **92.42% pooled** of claimed skills verified in candidate profile (`reports/llm_quality.json`).
  - Contradiction leakage: **0.31% pooled** (4/1,278 missing skills) (`reports/llm_quality.json`).

### Claim 1.4: Multi-turn conversational agent preserves constraints with 100% memory retention across scripted turns.
- **Status**: **SUPPORTED BY EMPIRICAL EVIDENCE**
- **Measured Number & File**:
  - Behavioral scenario pass rate: **9/10 (90.0%)** across 28 turns (`reports/agent_eval.json`).
  - Memory retention rate: **100.0%** (23/23 programmatic memory assertions passed) (`reports/agent_eval.json`).
  - Tool call success rate: **96.9%** (31/32 calls; 1 expected failure on non-existent job ID) (`reports/agent_eval.json`).
  - Mean tool steps per turn: **2.14 steps** (capped strictly at 5 steps per turn) (`reports/agent_eval.json`).

---

## 2. Unsupported Claims (Explicit Disclosures & Non-Claims)

### Claim 2.1: The project is a deployed or live service with a public URL.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: This project is **NOT deployed** to any cloud host, cluster, or live web URL. It has no live public endpoint. The Streamlit interface (`app.py`), FastAPI service (`src/api.py`), and Docker container are designed and verified strictly for local workstation execution.

### Claim 2.2: The system has real users or serves active job seekers.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: The system has **zero real users**. No job seekers or recruiters have interacted with this software. All evaluations were conducted offline on 30 hand-authored synthetic candidate profiles (`data/eval/profiles.json`) and 10 scripted agent scenarios (`tests/agent_scenarios.json`).

### Claim 2.3: The system operates at production scale.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: The project has not been tested at **production scale**. Benchmarks were executed locally on a single machine (AMD 12-core CPU) over a sampled subset of 3,000 job postings. High-concurrency query loads, distributed vector indexing, real-time streaming updates, and multi-tenant isolation have not been implemented or evaluated.

### Claim 2.4: Evaluation relevance labels are human-verified reference labels.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: Exactly **0% of labels are verified by humans** (`human_reviewed: false`, `human_verified_count: 0` in `reports/label_agreement.json`). All 1,037 relevance labels in `data/eval/qrels.json` were generated using deterministic rule-based heuristics (`jobmatch-rubric-v1`). They do not represent human judgments.

### Claim 2.5: Empirical findings generalise beyond this 3,000-posting dataset.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: **Generalisation beyond this dataset is unproven**. The evaluation was performed on a single 3,000-posting slice from a 2023 LinkedIn dataset. Niche technical roles (such as mobile engineering and computer vision) have sparse or zero representation. Performance on other job boards, different industries, or current market postings has not been measured.

### Claim 2.6: BM25, or any other strategy, is the better retriever for job matching.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: No strategy has been proven to be the **better retriever** for human job seekers. While BM25 achieved an nDCG@10 of 0.6147 vs Dense 0.4231 on the primary evaluation (`reports/retrieval_paired_comparisons.json`), this occurs because the rule-based labeling rubric shares lexical tokens and skill regex with BM25. On the circularity audit subset of 51 candidate postings with zero literal skill overlap, Dense retrieval outperformed BM25 in Recall@10 (0.3931 vs 0.2588, `reports/retrieval_circularity_check.json`), indicating that each retriever has distinct failure modes and neither is universally superior.

### Claim 2.7: The entire retrieval pipeline updates incrementally.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: Incremental synchronization applies **only to the ChromaDB dense vector store**. The lexical BM25 index relies on `rank_bm25`, which lacks an incremental update API. Any dataset mutation requires rebuilding the BM25 inverted index in full (~6.0 s CPU latency, `reports/retrieval_latency.json`).

### Claim 2.8: Cross-Encoder neural re-ranking improves retrieval quality over Hybrid RRF.
- **Status**: **UNSUPPORTED / NOT SUPPORTED**
- **Fact**: Two-stage re-ranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` achieved an nDCG@10 of 0.5954 vs 0.6100 for Hybrid RRF (`reports/retrieval_metrics_main.json`). The mean difference is -0.0146 with a 95% bootstrap CI [-0.0579, 0.0308] spanning zero (`reports/retrieval_paired_comparisons.json`), showing no statistically significant ranking improvement while adding 2,371.4 ms of CPU Transformer forward passes.
