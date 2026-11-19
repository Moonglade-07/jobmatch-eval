# Retrieval Strategy Evaluation & Analysis

> **Evaluated against model-proposed, unverified relevance labels**  
> *Relevance labels are rule-based proposed labels (jobmatch-rubric-v1) generated via deterministic heuristics; unverified by humans.*

---

## 1. Executive Summary

This report presents the retrieval evaluation for the Job-Matching Search system (`jobmatch-eval`) across all four retrieval architectures over a benchmark set of 30 candidate profiles:
1. **BM25 Only (`BM25Retriever`)**: Lexical keyword search using `rank_bm25.BM25Okapi` with domain-aware symbol tokenization.
2. **Dense Only (`DenseRetriever`)**: Semantic vector retrieval using ChromaDB HNSW cosine similarity over `sentence-transformers/all-MiniLM-L6-v2` embeddings.
3. **Hybrid RRF (`HybridRRFRetriever`)**: Reciprocal Rank Fusion ($k_{rrf} = 60$) combining BM25 and Dense ranking streams.
4. **Hybrid + Cross-Encoder (`CrossEncoderRetriever`)**: Two-stage pipeline combining Hybrid RRF retrieval (top 50 candidate pool) with neural cross-encoder re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### Core Evaluation Findings
- **Main Product Setting Performance**: Under the main product pipeline (with three-tier location and seniority filter fallback and explainable skill boost $w=0.2$), **BM25 Only** achieves a macro-average nDCG@10 of **0.615** (95% CI: [0.518, 0.696]), closely followed by **Hybrid RRF** at **0.607** (95% CI: [0.518, 0.680]) and **Hybrid + Cross-Encoder** at **0.593** (95% CI: [0.505, 0.665]). **Dense Only** achieves an nDCG@10 of **0.423** (95% CI: [0.326, 0.522]).
- **Paired Statistical Significance**: Paired bootstrap comparisons ($N=1,000$ resamples) reveal that while BM25 and Hybrid RRF demonstrate statistically significant gains over Dense Only (+0.192 and +0.184 nDCG@10 respectively), there is **no clear difference** among BM25, Hybrid RRF, and Cross-Encoder (all 95% confidence intervals on paired differences span 0.0).
- **Label Origin Bias**: The relevance labels were produced by a deterministic heuristic (`jobmatch-rubric-v1`) based on title-token overlap, technical skill regex, location, and seniority rules. **Lexical methods and the skill boost are inherently favoured by these labels because the evaluation rubric shares their exact keyword and skill match signals.**
- **Circularity Audit**: When restricted to the 51 relevant candidate postings with **zero literal skill match** to candidate profiles, the ranking reverses: **Dense Only outperforms BM25 Only** (Recall@10: 0.393 vs 0.259; MRR: 0.344 vs 0.175; nDCG@10: 0.138 vs 0.093), confirming that dense retrieval captures semantic relevance when literal keywords are absent, despite the heuristic's lexical bias.
- **Latency vs Quality Trade-Off**: Cross-Encoder scoring requires a p50 latency of **2,405 ms** (p95: 2,575 ms) on CPU for scoring 50 pairs, yielding an end-to-end pipeline latency of **3,637 ms**. In contrast, Dense Only executes at **449 ms** p50 and Hybrid RRF at **1,249 ms** p50. Given that Cross-Encoder shows no statistically significant metric improvement over Hybrid RRF on this dataset, Hybrid RRF represents the superior production operating point.

---

## 2. Retrieval Benchmark Results

All metrics are macro-averaged across the 30 candidate profiles. Queries with 0 relevant documents in the candidate pool (`prof_07` and `prof_23`) are explicitly excluded from Recall@10 and MRR means (evaluated over 28 profiles), while Precision@5, nDCG@10, and unjudged rates are computed across all 30 profiles. Documents retrieved but absent from the pooled qrels are treated as non-relevant (label 0).

### Main Setting: Product Pipeline (Filters ON, Skill Boost ON $w=0.2$)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 | CPU p50 Latency | CPU p95 Latency |
|---|---|---|---|---|---|---|---|
| **BM25 Only** | 0.520 [0.413, 0.633] | **0.548** [0.465, 0.627] | 0.887 [0.792, 0.970] | **0.615** [0.518, 0.696] | 0.00% | 5,450.5 ms | 5,872.0 ms |
| **Dense Only** | 0.427 [0.307, 0.547] | 0.395 [0.313, 0.476] | 0.701 [0.556, 0.849] | 0.423 [0.326, 0.522] | 0.00% | **449.2 ms** | **544.8 ms** |
| **Hybrid RRF ($k=60$)** | **0.533** [0.420, 0.647] | 0.541 [0.466, 0.621] | **0.910** [0.823, 0.982] | 0.607 [0.518, 0.680] | 0.00% | 1,249.4 ms | 1,488.8 ms |
| **Hybrid + Cross-Encoder** | 0.527 [0.413, 0.633] | 0.539 [0.467, 0.619] | 0.864 [0.763, 0.954] | 0.593 [0.505, 0.665] | 0.00% | 3,636.7 ms | 3,843.8 ms |

*Note: Unjudged rate in top 10 is 0.00% across all strategies because candidate pooling pooled the top 20 results from these exact four strategies under the main product configuration.*

### Secondary Setting: Pure Retrieval (Filters OFF, Skill Boost OFF)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 |
|---|---|---|---|---|---|
| **BM25 Only** | 0.127 [0.060, 0.200] | 0.142 [0.091, 0.204] | 0.254 [0.144, 0.376] | 0.162 [0.098, 0.231] | 85.33% |
| **Dense Only** | 0.140 [0.087, 0.200] | 0.138 [0.086, 0.199] | 0.295 [0.180, 0.438] | 0.161 [0.102, 0.231] | **76.00%** |
| **Hybrid RRF ($k=60$)** | 0.147 [0.080, 0.227] | 0.141 [0.087, 0.201] | 0.293 [0.169, 0.446] | 0.177 [0.104, 0.264] | 82.33% |
| **Hybrid + Cross-Encoder** | **0.160** [0.093, 0.247] | **0.216** [0.146, 0.304] | **0.320** [0.205, 0.449] | **0.186** [0.130, 0.252] | 79.33% |

#### Key Insight on Pure Retrieval & Unjudged Documents
When location and seniority filters are disabled, the unjudged rate in the top 10 surges to **76% – 85%**. This occurs because candidate pooling in Phase 2A was performed using the product pipeline (where candidates were filtered to match candidate location and seniority constraints). When pure retrieval is executed without filters, models retrieve postings from disparate geographic locations across the US/world that were never pooled or judged, and are therefore treated as non-relevant (label 0). This highlights the crucial role that metadata filtering plays in practical job search.

---

## 3. Paired Statistical Comparisons

To determine whether metric differences between strategies represent meaningful improvements or stochastic noise, we computed paired bootstrap differences across all 30 profiles ($N=1,000$ iterations, seed 42). Whenever the 95% confidence interval spans 0.0, we conclude **no clear difference**.

*Evaluated against model-proposed, unverified relevance labels*

| Comparison ($A$ vs $B$) | Metric | Mean Difference ($A - B$) | 95% Bootstrap CI | Statistical Conclusion |
|---|---|---|---|---|
| **BM25 vs Dense** | Precision@5 | +0.0933 | [-0.0067, +0.1935] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | **+0.1523** | [+0.0388, +0.2735] | **Clear difference** (BM25 superior) |
| | MRR | **+0.1863** | [+0.0640, +0.3027] | **Clear difference** (BM25 superior) |
| | nDCG@10 | **+0.1916** | [+0.1109, +0.2610] | **Clear difference** (BM25 superior) |
| **Hybrid RRF vs BM25** | Precision@5 | +0.0133 | [-0.0400, +0.0667] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | -0.0071 | [-0.0556, +0.0405] | **No clear difference** (CI spans 0.0) |
| | MRR | +0.0230 | [-0.0366, +0.0884] | **No clear difference** (CI spans 0.0) |
| | nDCG@10 | -0.0080 | [-0.0534, +0.0352] | **No clear difference** (CI spans 0.0) |
| **Hybrid RRF vs Dense** | Precision@5 | **+0.1067** | [+0.0400, +0.1800] | **Clear difference** (Hybrid superior) |
| | Recall@10 | **+0.1452** | [+0.0538, +0.2519] | **Clear difference** (Hybrid superior) |
| | MRR | **+0.2093** | [+0.1033, +0.3179] | **Clear difference** (Hybrid superior) |
| | nDCG@10 | **+0.1836** | [+0.1223, +0.2389] | **Clear difference** (Hybrid superior) |
| **Cross-Encoder vs Hybrid RRF** | Precision@5 | -0.0067 | [-0.0733, +0.0600] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | -0.0012 | [-0.0408, +0.0333] | **No clear difference** (CI spans 0.0) |
| | MRR | -0.0456 | [-0.1337, +0.0340] | **No clear difference** (CI spans 0.0) |
| | nDCG@10 | -0.0140 | [-0.0579, +0.0311] | **No clear difference** (CI spans 0.0) |
| **Cross-Encoder vs BM25** | Precision@5 | +0.0067 | [-0.0800, +0.0800] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | -0.0083 | [-0.0571, +0.0458] | **No clear difference** (CI spans 0.0) |
| | MRR | -0.0226 | [-0.1144, +0.0667] | **No clear difference** (CI spans 0.0) |
| | nDCG@10 | -0.0220 | [-0.0807, +0.0383] | **No clear difference** (CI spans 0.0) |

### Effective Sample Size & Correlation Analysis
The evaluation dataset contains 30 candidate profiles. However, the effective sample size is smaller than 30 due to correlation among profiles that share role families, locations, and seniority tiers:
- **Location Clustering**: 10 profiles specify `Remote` and 4 specify `San Francisco, CA`. Remote profiles draw from an identical pool of remote postings, causing query results to covary.
- **Seniority Clustering**: 18 of the 30 profiles target `Mid-Senior level`.
- **Zero-Relevant Queries**: Exactly 2 profiles (`prof_07` and `prof_23`, targeting Mobile Dev in SF and Remote) have 0 relevant items in the index, reducing the effective sample size for Recall and MRR to 28 queries.
Because queries are correlated, confidence intervals are wider than would be expected under 30 independent and identically distributed queries.

---

## 4. In-Depth Query Analysis: Three Concrete Case Studies

To understand the qualitative mechanics driving these numbers, we examine three queries from `reports/retrieval_per_profile_main.csv`.

### Case Study 1: BM25 Beats Dense (`prof_02` — Senior Backend Engineer)
- **Profile Profile**:
  - Target Role: Senior Backend Engineer
  - Seniority: Mid-Senior level
  - Preferred Location: Austin, TX
  - Declared Skills: Go, Python, PostgreSQL, Redis, Kubernetes, Docker, gRPC, Distributed Systems
- **Evaluation Performance**:
  - **BM25 Only**: Precision@5 = **1.000**, Recall@10 = **0.500**, MRR = **1.000**, nDCG@10 = **0.6489**
  - **Dense Only**: Precision@5 = **0.200**, Recall@10 = **0.300**, MRR = **0.3333**, nDCG@10 = **0.2431**
- **Mechanical Breakdown**:
  - Backend engineering job postings in this dataset frequently require exact programming language and database keywords (e.g., `Go`, `Golang`, `PostgreSQL`, `Kubernetes`).
  - BM25 matched the explicit keyword tokens `Go` and `Kubernetes` in the title and requirements section, placing 5 relevant Austin backend roles in the top 5 (ranks 1 through 5 all relevant).
  - Dense embedding search using `all-MiniLM-L6-v2` mapped the query into a broad semantic region covering general software development, retrieving generic IT analyst and frontend postings that lacked the strict backend systems infrastructure requirements.

### Case Study 2: Dense Beats BM25 (`prof_20` — Technical Product Manager)
- **Profile Profile**:
  - Target Role: Technical Product Manager
  - Seniority: Mid-Senior level
  - Preferred Location: Remote
  - Declared Skills: Product Roadmap, Agile, Scrum, JIRA, SQL, User Stories, PRD, Stakeholder Management
- **Evaluation Performance**:
  - **Dense Only**: Precision@5 = **1.000**, Recall@10 = **0.7273**, MRR = **1.000**, nDCG@10 = **0.8701**
  - **BM25 Only**: Precision@5 = **0.400**, Recall@10 = **0.2727**, MRR = **1.000**, nDCG@10 = **0.4284**
- **Mechanical Breakdown**:
  - Product management job descriptions suffer from high vocabulary mismatch. Postings describe responsibilities using varied terminology such as "lead cross-functional delivery", "drive product vision", "define KPI roadmaps", and "partner with engineering squads" rather than repeating verbatim skill acronyms like `PRD` or `User Stories`.
  - BM25 failed to retrieve relevant PM postings whose descriptions used conversational leadership prose rather than specific tool keywords, achieving only 40% Precision@5.
  - Dense semantic retrieval captured the conceptual intent of product leadership, retrieving 5 relevant postings in the top 5 and capturing 8 of the 11 total relevant postings in the top 10 (Recall@10 = 0.7273).

### Case Study 3: Cross-Encoder Promotes Relevant Postings (`prof_01` — ML Engineer)
- **Profile Profile**:
  - Target Role: Machine Learning Engineer
  - Seniority: Mid-Senior level
  - Preferred Location: Remote
  - Declared Skills: Python, PyTorch, NLP, Transformers, FastAPI, Docker, ChromaDB
- **Evaluation Performance**:
  - **Hybrid RRF**: Precision@5 = **0.000**, MRR = **0.1429** (first relevant item at rank 7), nDCG@10 = **0.3333**
  - **Cross-Encoder**: Precision@5 = **0.200**, MRR = **0.5000** (first relevant item promoted to rank 2), nDCG@10 = **0.6309**
- **Mechanical Breakdown**:
  - Under Hybrid RRF, the single relevant ML Engineer posting (`job_id: 3886216113`) was placed at rank 7 because competing postings had high BM25 keyword density for generic terms like `Python` and `Docker`.
  - The cross-encoder (`ms-marco-MiniLM-L-6-v2`) performed cross-attention between the candidate's full profile narrative ("deploying neural search, vector databases, and transformer models to production microservices") and the job's full description.
  - The cross-encoder scored the actual ML engineering passage with a high neural relevance score, promoting it from **rank 7 to rank 2**. This single promotion increased MRR from 0.143 to 0.500 and almost doubled nDCG@10 from 0.333 to 0.631.

---

## 5. Ablation Study: Filter, Boost, and Weight Grid

We executed an ablation grid on the top-performing strategy (**BM25 Only**) across filter toggles, skill boost toggles, and boost weights $w \in \{0.0, 0.1, 0.2, 0.4\}$.

*Evaluated against model-proposed, unverified relevance labels*

| Configuration | Location & Seniority Filters | Skill Boost ($w$) | Precision@5 | Recall@10 | MRR | nDCG@10 | Unjudged Rate@10 |
|---|---|---|---|---|---|---|---|
| **Filters ON, Boost ON ($w=0.2$) [Main]** | **ON** | **0.2** | 0.5200 | **0.5476** | **0.8869** | **0.6147** | 0.00% |
| **Filters ON, Boost OFF ($w=0.0$)** | **ON** | 0.0 | 0.5200 | 0.5464 | 0.8690 | 0.6062 | 0.00% |
| **Filters ON, Boost $w=0.1$** | **ON** | 0.1 | 0.5200 | **0.5476** | **0.8869** | 0.6135 | 0.00% |
| **Filters ON, Boost $w=0.4$** | **ON** | 0.4 | **0.5267** | **0.5476** | 0.8861 | 0.6122 | 0.00% |
| **Filters OFF, Boost ON ($w=0.2$)** | OFF | 0.2 | 0.1333 | 0.1562 | 0.2750 | 0.1768 | 84.33% |
| **Filters OFF, Boost OFF ($w=0.0$) [Pure]** | OFF | 0.0 | 0.1267 | 0.1420 | 0.2541 | 0.1622 | 85.33% |

### Ablation Insights
1. **Metadata Filters Are Critical**: Turning off location and seniority filters causes nDCG@10 to collapse from **0.615 to 0.162** (-73.6% relative drop). This is because candidates overwhelmingly require jobs in their target location and seniority, and unjudged out-of-market postings dominate the unfiltered top 10.
2. **Skill Boost Tuning ($w$)**:
   - Adding min-max normalized skill boosting ($w=0.2$) improves nDCG@10 from **0.6062 to 0.6147** and MRR from **0.8690 to 0.8869**, successfully promoting skill-aligned postings into the top 3.
   - At $w=0.1$, the boost improves nDCG to 0.6135.
   - At $w=0.4$, the boost slightly increases Precision@5 (0.5267) but causes a slight dip in nDCG@10 (0.6122) as heavy skill boosting occasionally promotes postings with high skill overlap but weaker primary role alignment over closer role matches. The chosen default $w=0.2$ provides the optimal balance.

---

## 6. Circularity Audit: Zero Literal Skill Match Subset

To examine whether the apparent superiority of lexical retrieval was solely an artifact of the heuristic labeling rubric (which checked technical skill keywords), we isolated all relevant candidate pairs having **zero literal matches** of candidate declared skills in the posting title or description.

### Subset Characteristics
- **Total Zero-Overlap Relevant Pairs**: 51 postings (out of 280 total relevant pairs across all profiles).
- **Profile Coverage**: Only 17 of the 30 candidate profiles have at least one relevant document with zero skill matches; 13 profiles have zero such documents.
- **Sample Size Warning**: This subset is very small and carries high sampling variance. The 13 queries with zero relevant items are excluded from Recall and MRR macro means.

### Performance on Zero-Overlap Subset
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| **BM25 Only** | 0.0333 | 0.2588 | 0.1751 | 0.0932 |
| **Dense Only** | **0.0600** | **0.3931** | **0.3437** | **0.1383** |
| **Hybrid RRF ($k=60$)** | **0.0800** | 0.3180 | 0.2319 | 0.1124 |
| **Hybrid + Cross-Encoder** | 0.0467 | 0.2552 | 0.3077 | 0.1098 |

### Circularity Conclusion
When literal skill keywords are absent, **Dense Only outperforms BM25 Only** across all metrics:
- Recall@10: **0.3931** (Dense) vs 0.2588 (BM25) — a **+51.9% relative advantage for Dense**.
- MRR: **0.3437** (Dense) vs 0.1751 (BM25) — a **+96.3% relative advantage for Dense**.
- nDCG@10: **0.1383** (Dense) vs 0.0932 (BM25) — a **+48.4% relative advantage for Dense**.

This directly validates that dense semantic vectors successfully discover conceptual job matches that completely evade lexical search. Conversely, in the full benchmark where 81.8% of relevant postings contain explicit skill tokens, BM25 and skill boosting perform strongly because the heuristic labels share those lexical markers.

---

## 7. Latency Benchmarks & Production Trade-Offs

Latency was profiled on an AMD 12-core CPU under PyTorch CPU execution (`torch.set_num_threads(6)`, `CUDA: False`) with warm-up enabled and 25 timed repetitions across diverse queries.

*Evaluated against model-proposed, unverified relevance labels*

| Component / Strategy | p50 Latency (ms) | p95 Latency (ms) | Mean Latency (ms) | Std Dev (ms) | Measured Operations |
|---|---|---|---|---|---|
| **Dense Only** | **449.15** | **544.77** | 453.10 | 55.22 | Query embedding (MiniLM-L6) + ChromaDB HNSW query |
| **Hybrid RRF ($k=60$)** | **1,249.39** | **1,488.81** | 1,249.79 | 122.58 | BM25 search + Dense search + RRF fusion |
| **BM25 Only** | **5,450.48** | **5,872.00** | 5,441.48 | 303.18 | Lexical Okapi scoring + filtering over candidate pool |
| **Cross-Encoder: Step 1 (Retrieval)** | 1,236.08 | 1,501.10 | 1,248.36 | 120.40 | Hybrid RRF retrieval of top 50 candidates |
| **Cross-Encoder: Step 2 (Neural Scoring)** | 2,405.34 | 2,574.84 | 2,391.62 | 108.90 | Transformer forward pass re-ranking 50 pairs on CPU |
| **Cross-Encoder: End-to-End Pipeline** | **3,636.65** | **3,843.80** | 3,639.98 | 149.23 | Total two-stage retrieval and re-ranking |

### Architectural Recommendation
1. **Hybrid RRF is the Optimal Production Strategy**: Hybrid RRF achieves an nDCG@10 of 0.607 with an end-to-end latency of ~1.2 seconds, combining the exact keyword precision of BM25 with the semantic recall of dense vectors.
2. **Cross-Encoder on CPU is Prohibitive**: Cross-Encoder scoring adds ~2.4 seconds of CPU compute per query (3.6s total). Because its nDCG@10 (0.593) is statistically indistinguishable from Hybrid RRF (95% CI on difference: [-0.058, +0.031]), deploying Cross-Encoder on CPU is not justifiable without GPU acceleration or student-model distillation.

---

## 8. Limitations & Methodological Integrity

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance judgments in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`), not human verified (`human_reviewed: false`). The evaluations reflect how closely each retrieval algorithm aligns with the deterministic heuristic rubric (`jobmatch-rubric-v1`).
2. **Shared Signal Bias**: Because the proposal heuristic evaluated title tokens and regex skill matches, lexical search (BM25) and the skill boost operate with an inherent advantage on the primary benchmark.
3. **Synthetic Test Profiles**: The 30 candidate profiles are hand-authored synthetic benchmarks designed to test specific edge cases, while the 3,000 job postings in `data/jobs_clean.parquet` remain real LinkedIn postings.
4. **Effective Sample Size**: Due to regional and seniority correlations among test profiles, the effective sample size is lower than 30, requiring wide confidence intervals when interpreting statistical significance.

