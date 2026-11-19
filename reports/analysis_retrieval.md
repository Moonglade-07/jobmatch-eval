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
- **Main Product Setting Performance**: Under the main product pipeline (with three-tier location and seniority filter fallback and explainable skill boost $w=0.2$), **BM25 Only** achieves a macro-average nDCG@10 of **0.615** (95% CI: [0.518, 0.696]), closely followed by **Hybrid RRF** at **0.610** (95% CI: [0.520, 0.684]) and **Hybrid + Cross-Encoder** at **0.595** (95% CI: [0.507, 0.668]). **Dense Only** achieves an nDCG@10 of **0.423** (95% CI: [0.326, 0.522]).
- **Paired Statistical Comparisons**: Paired bootstrap comparisons ($N=1,000$ resamples) indicate that while BM25 and Hybrid RRF yield higher metric scores than Dense Only (+0.192 and +0.184 nDCG@10 respectively) against these heuristic labels, this is evaluated against rule-based proposed labels that share lexical signals with BM25, and is not evidence that BM25 is the better retriever in absolute terms. Furthermore, there is **no clear difference** among BM25, Hybrid RRF, and Cross-Encoder (all 95% confidence intervals on paired differences span 0.0).
- **Label Origin Bias**: The relevance labels were produced by a deterministic heuristic (`jobmatch-rubric-v1`) based on title-token overlap, technical skill regex, location, and seniority rules. **Lexical methods and the skill boost are inherently favoured by these labels because the evaluation rubric shares their exact keyword and skill match signals.**
- **Circularity Audit (Indicative Only: 51 pairs, 17 profiles)**: When restricted to the small subset of 51 relevant candidate postings with **zero literal skill match** to candidate profiles, dense vector retrieval achieves higher recall and MRR than BM25 (Recall@10: 0.393 [0.233, 0.569] vs 0.259 [0.125, 0.427]; MRR: 0.344 [0.155, 0.531] vs 0.175 [0.073, 0.314]). This suggests that dense retrieval can capture semantic relevance when literal keywords are absent, but the small sample size and wide confidence intervals mean this finding is indicative only.
- **Latency vs Quality Trade-Off**: CPU query latency profiling (with index builds and corpus tokenization executed once at startup outside the timed region) measures a p50 latency of **136.2 ms** for BM25 Only, **74.2 ms** for Dense Only, and **181.5 ms** for Hybrid RRF. Cross-Encoder scoring adds **2,337.3 ms** p50 on CPU (scoring 50 pairs), yielding an end-to-end pipeline latency of **2,479.3 ms** p50. Given that Cross-Encoder shows no statistically significant metric improvement over Hybrid RRF, Hybrid RRF represents the practical production operating point.

---

## 2. Retrieval Benchmark Results

All metrics are macro-averaged across the 30 candidate profiles. Queries with 0 relevant documents in the candidate pool (`prof_07`: Mobile Developer, `prof_23`: Computer Vision Engineer) are explicitly excluded from Recall@10 and MRR means (evaluated over 28 profiles), while Precision@5, nDCG@10, and unjudged rates are computed across all 30 profiles. Documents retrieved but absent from the pooled qrels are treated as non-relevant (label 0).

### Main Setting: Product Pipeline (Filters ON, Skill Boost ON $w=0.2$)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 | CPU p50 Latency | CPU p95 Latency |
|---|---|---|---|---|---|---|---|
| **BM25 Only** | 0.520 [0.413, 0.633] | **0.548** [0.465, 0.627] | 0.887 [0.792, 0.970] | **0.615** [0.518, 0.696] | 0.00% | 136.2 ms | 168.3 ms |
| **Dense Only** | 0.427 [0.307, 0.547] | 0.395 [0.313, 0.476] | 0.701 [0.556, 0.849] | 0.423 [0.326, 0.522] | 0.00% | **74.2 ms** | **91.3 ms** |
| **Hybrid RRF ($k=60$)** | **0.533** [0.420, 0.647] | 0.546 [0.468, 0.626] | **0.910** [0.823, 0.982] | 0.610 [0.520, 0.684] | 0.00% | 181.5 ms | 206.4 ms |
| **Hybrid + Cross-Encoder** | 0.527 [0.413, 0.633] | 0.544 [0.469, 0.626] | 0.864 [0.763, 0.954] | 0.595 [0.507, 0.668] | 0.00% | 2,479.3 ms | 2,802.1 ms |

*Note: Unjudged rate in top 10 is 0.00% across all strategies because candidate pooling pooled the top 20 results from these exact four strategies under the main product configuration.*

### Secondary Setting: Pure Retrieval (Filters OFF, Skill Boost OFF) — Not comparable (labels pooled under filters, 76% unjudged)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 |
|---|---|---|---|---|---|
| **BM25 Only** | 0.127 [0.060, 0.200] | 0.142 [0.091, 0.204] | 0.254 [0.144, 0.376] | 0.162 [0.098, 0.231] | 85.33% |
| **Dense Only** | 0.140 [0.087, 0.200] | 0.138 [0.086, 0.199] | 0.295 [0.180, 0.438] | 0.161 [0.102, 0.231] | **76.00%** |
| **Hybrid RRF ($k=60$)** | 0.147 [0.080, 0.227] | 0.141 [0.087, 0.201] | 0.293 [0.169, 0.446] | 0.177 [0.104, 0.264] | 82.33% |
| **Hybrid + Cross-Encoder** | **0.160** [0.093, 0.247] | **0.216** [0.146, 0.304] | **0.320** [0.205, 0.449] | **0.186** [0.130, 0.252] | 79.33% |

> [!WARNING]
> **Not comparable (labels pooled under filters, 76% unjudged)**: Because candidate pooling was performed under active location and seniority filters, out-of-market jobs retrieved when filters are off were never pooled or evaluated, causing unjudged rates of 76%–85% (treated as non-relevant label 0). These numbers do not reflect pure retrieval quality, and no conclusions about retriever quality should be drawn from them.

---

## 3. Paired Statistical Comparisons

To determine whether metric differences between strategies represent meaningful variations or stochastic noise, we computed paired bootstrap differences across all 30 profiles ($N=1,000$ iterations, seed 42). Whenever the 95% confidence interval spans 0.0, we conclude **no clear difference**.

*Evaluated against model-proposed, unverified relevance labels*

| Comparison ($A$ vs $B$) | Metric | Mean Difference ($A - B$) | 95% Bootstrap CI | Statistical Conclusion |
|---|---|---|---|---|
| **BM25 vs Dense** | Precision@5 | +0.0933 | [-0.0067, +0.1935] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | **+0.1523** | [+0.0388, +0.2735] | **Clear difference against rule-based labels** (not evidence of better retriever) |
| | MRR | **+0.1863** | [+0.0640, +0.3027] | **Clear difference against rule-based labels** (not evidence of better retriever) |
| | nDCG@10 | **+0.1916** | [+0.1109, +0.2610] | **Clear difference against rule-based labels** (not evidence of better retriever) |
| **Hybrid RRF vs BM25** | Precision@5 | +0.0133 | [-0.0400, +0.0667] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | -0.0016 | [-0.0494, +0.0458] | **No clear difference** (CI spans 0.0) |
| | MRR | +0.0230 | [-0.0366, +0.0884] | **No clear difference** (CI spans 0.0) |
| | nDCG@10 | -0.0049 | [-0.0505, +0.0389] | **No clear difference** (CI spans 0.0) |
| **Hybrid RRF vs Dense** | Precision@5 | **+0.1067** | [+0.0400, +0.1800] | **Clear difference against rule-based labels** (not evidence of better retriever) |
| | Recall@10 | **+0.1507** | [+0.0583, +0.2588] | **Clear difference against rule-based labels** (not evidence of better retriever) |
| | MRR | **+0.2093** | [+0.1033, +0.3179] | **Clear difference against rule-based labels** (not evidence of better retriever) |
| | nDCG@10 | **+0.1866** | [+0.1245, +0.2429] | **Clear difference against rule-based labels** (not evidence of better retriever) |
| **Cross-Encoder vs Hybrid RRF** | Precision@5 | -0.0067 | [-0.0733, +0.0600] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | -0.0018 | [-0.0417, +0.0333] | **No clear difference** (CI spans 0.0) |
| | MRR | -0.0456 | [-0.1337, +0.0340] | **No clear difference** (CI spans 0.0) |
| | nDCG@10 | -0.0152 | [-0.0592, +0.0298] | **No clear difference** (CI spans 0.0) |
| **Cross-Encoder vs BM25** | Precision@5 | +0.0067 | [-0.0800, +0.0800] | **No clear difference** (CI spans 0.0) |
| | Recall@10 | -0.0034 | [-0.0531, +0.0500] | **No clear difference** (CI spans 0.0) |
| | MRR | -0.0226 | [-0.1144, +0.0667] | **No clear difference** (CI spans 0.0) |
| | nDCG@10 | -0.0201 | [-0.0789, +0.0401] | **No clear difference** (CI spans 0.0) |

### Effective Sample Size & Profile Verification Details
The evaluation dataset contains 30 candidate profiles. However, the effective sample size is smaller than 30 due to correlation among profiles that share role families, locations, and seniority tiers:
- **Location Clustering**: 10 profiles specify `Remote` and 4 specify `San Francisco, CA`. Remote profiles draw from an identical pool of remote postings, causing query results to covary.
- **Seniority Clustering**: 18 of the 30 profiles target `Mid-Senior level`.
- **Zero-Relevant Queries**: Exactly 2 profiles have 0 relevant items in the index, reducing the effective sample size for Recall and MRR to 28 queries.

#### Profile Audit: Clarifying `prof_07` and `prof_23`
In an earlier conversational turn, `prof_07` and `prof_23` were erroneously referenced as "Staff Security Architect" and "Lead DevOps SRE". The ground dataset in `data/eval/profiles.json` defines them as follows:

1. **Profile `prof_07`**:
   - **Target Role**: `Mobile Developer`
   - **Seniority**: `Mid-Senior level`
   - **Preferred Location**: `San Francisco, CA`
   - **Declared Skills**: `['Swift', 'iOS', 'Kotlin', 'Android', 'React Native', 'Mobile UI', 'REST APIs']`
   - **Resume Text**: *"Mobile software engineer with 5 years shipping consumer-facing iOS and Android applications. Deep knowledge of Swift, Kotlin, offline sync caching, and native performance optimization."*
   - **Judged Candidate Pairs**: 33 pairs
   - **Relevant Documents (Grade $\ge 1$)**: **0**

2. **Profile `prof_23`**:
   - **Target Role**: `Computer Vision Engineer`
   - **Seniority**: `Mid-Senior level`
   - **Preferred Location**: `Remote`
   - **Declared Skills**: `['Python', 'OpenCV', 'PyTorch', 'YOLO', 'Image Processing', 'Object Detection', 'CUDA']`
   - **Resume Text**: *"Computer Vision Engineer with 3.5 years training real-time object detection models (YOLO), video stream feature extraction using OpenCV, and edge device deployment."*
   - **Judged Candidate Pairs**: 31 pairs
   - **Relevant Documents (Grade $\ge 1$)**: **0**

Both profiles had 0 postings passing the strict rubric thresholds in the 3,000-job sample and are properly excluded from Recall@10 and MRR calculations to prevent undefined division by zero.

---

## 4. In-Depth Query Analysis: Three Concrete Case Studies

To understand the qualitative mechanics driving these numbers, we examine three queries from `reports/retrieval_per_profile_main.csv`.

### Case Study 1: BM25 Scores Higher Against Rule-Based Labels (`prof_02` — Senior Backend Engineer)
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
  - *Context*: Because `jobmatch-rubric-v1` specifically checked for the presence of skill keywords, BM25 aligns directly with the evaluation rubric in this query.

### Case Study 2: Dense Captures Semantic Intent When Lexical Vocabulary Varies (`prof_20` — Technical Product Manager)
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

We executed an ablation grid on **BM25 Only** across filter toggles, skill boost toggles, and boost weights $w \in \{0.0, 0.1, 0.2, 0.4\}$.

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
1. **Candidate Pool Pre-Filtering**: When location and seniority filters are disabled, scores collapse because candidate pooling was conducted under filters, causing out-of-market retrieved jobs to be unjudged (treated as label 0). This confirms that evaluation numbers outside the filtered pool are uninformative without unconstrained labeling.
2. **Skill Boost Tuning ($w$)**:
   - Adding min-max normalized skill boosting ($w=0.2$) improves nDCG@10 from **0.6062 to 0.6147** and MRR from **0.8690 to 0.8869** against the rule-based rubric.
   - However, because the rubric directly rewards skill regex matches, this observed boost gain is inflated by signal overlap.
   - At $w=0.4$, the boost slightly increases Precision@5 (0.5267) but causes a slight dip in nDCG@10 (0.6122) as heavy skill boosting occasionally promotes postings with high skill overlap but weaker primary role alignment over closer role matches. The chosen default $w=0.2$ provides a balanced weighting.

---

## 6. Circularity Audit: Zero Literal Skill Match Subset (Indicative Only: 51 pairs, 17 profiles)

To evaluate retrieval behavior when lexical cues are missing, we isolated all relevant candidate pairs having **zero literal matches** of candidate declared skills in the posting title or description.

### Subset Characteristics & Scope
- **Total Zero-Overlap Relevant Pairs**: 51 postings (out of 280 total relevant pairs across all profiles).
- **Profile Coverage**: Only 17 of the 30 candidate profiles have at least one relevant document with zero skill matches; 13 profiles have zero such documents.
- **Sample Size Warning (Indicative Only)**: This sample size is small (51 pairs across 17 profiles). The 13 queries with zero relevant items are excluded from Recall and MRR macro means. Consequently, confidence intervals are wide and findings are indicative only.

### Performance on Zero-Overlap Subset
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) |
|---|---|---|---|---|
| **BM25 Only** | 0.0333 [0.0000, 0.0800] | 0.2588 [0.1245, 0.4265] | 0.1751 [0.0727, 0.3139] | 0.0932 [0.0375, 0.1632] |
| **Dense Only** | **0.0600** [0.0200, 0.1067] | **0.3931** [0.2333, 0.5687] | **0.3437** [0.1546, 0.5314] | **0.1383** [0.0736, 0.2139] |
| **Hybrid RRF ($k=60$)** | **0.0800** [0.0333, 0.1400] | 0.3180 [0.1552, 0.4869] | 0.2319 [0.0984, 0.3879] | 0.1124 [0.0535, 0.1788] |
| **Hybrid + Cross-Encoder** | 0.0467 [0.0133, 0.0933] | 0.2552 [0.1114, 0.4137] | 0.3077 [0.1444, 0.5055] | 0.1098 [0.0558, 0.1731] |

### Indicative Takeaway
When literal skill keywords are absent, Dense Only achieves higher recall (0.3931 vs 0.2588) and MRR (0.3437 vs 0.1751) than BM25 Only. This indicates that dense vector retrieval can surface conceptually relevant postings when keyword markers are absent. However, due to the small sample size (51 pairs across 17 profiles) and wide confidence intervals, this finding is indicative only and does not establish statistical superiority. Conversely, on the full benchmark where 81.8% of relevant postings contain explicit skill tokens, BM25 and skill boosting perform strongly because the heuristic labels share those lexical markers.

---

## 7. Latency Benchmarks & Production Trade-Offs

Latency was profiled on an AMD 12-core CPU under PyTorch CPU execution (`torch.set_num_threads(6)`, `CUDA: False`) with warm-up enabled and 25 timed repetitions across diverse queries. Index building, tokenization, and model loading were executed once at startup outside the timed region.

*Evaluated against model-proposed, unverified relevance labels*

### One-Off Startup & Index Build Times
- **BM25 Index Build & Tokenized Corpus**: 5,992.1 ms (one-off startup)
- **Dense Retriever Collection Initialization**: 877.8 ms (one-off startup)
- **Cross-Encoder Model Loading**: 242.5 ms (one-off startup)

### Timed Query Execution Latency
| Component / Strategy | p50 Latency (ms) | p95 Latency (ms) | Mean Latency (ms) | Std Dev (ms) | Measured Operations |
|---|---|---|---|---|---|
| **Dense Only** | **74.24** | **91.32** | 75.06 | 7.28 | Query embedding (MiniLM-L6) + ChromaDB HNSW query |
| **BM25 Only** | **136.24** | **168.32** | 146.96 | 52.89 | BM25Okapi query scoring + fast candidate filtering |
| **Hybrid RRF ($k=60$)** | **181.46** | **206.44** | 177.55 | 16.94 | BM25 search + Dense search + RRF fusion |
| **Cross-Encoder: Step 1 (Retrieval)** | 157.20 | 179.94 | 166.46 | 13.20 | Hybrid RRF retrieval of top 50 candidates |
| **Cross-Encoder: Step 2 (Neural Scoring)** | 2,337.29 | 2,622.19 | 2,371.40 | 114.50 | Transformer forward pass re-ranking 50 pairs on CPU |
| **Cross-Encoder: End-to-End Pipeline** | **2,479.30** | **2,802.10** | 2,537.86 | 120.30 | Total two-stage retrieval and re-ranking |

### Architectural Recommendation
1. **Hybrid RRF is the Optimal Production Strategy**: Hybrid RRF achieves an nDCG@10 of 0.610 with an end-to-end latency of **181.5 ms** p50, combining the exact keyword precision of BM25 with the semantic recall of dense vectors.
2. **Cross-Encoder on CPU is Prohibitive**: Cross-Encoder scoring adds ~2.3 seconds of CPU compute per query (2.5s total). Because its nDCG@10 (0.595) is statistically indistinguishable from Hybrid RRF (95% CI on difference: [-0.059, +0.030]), deploying Cross-Encoder on CPU is not justifiable without GPU acceleration or student-model distillation.

---

## 8. Limitations & Methodological Integrity

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance judgments in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`), not human verified (`human_reviewed: false`). The evaluations reflect how closely each retrieval algorithm aligns with the deterministic heuristic rubric (`jobmatch-rubric-v1`).
2. **Shared Signal Bias**: Because the proposal heuristic evaluated title tokens and regex skill matches, lexical search (BM25) and the skill boost operate with an inherent advantage on the primary benchmark. High BM25 scores reflect shared lexical signals with the rubric, not evidence that BM25 is the better retriever in absolute terms.
3. **Ranking Inside Pre-Filtered Pool, Not Full-Index Retrieval**: Reported retrieval metrics evaluate ranking within candidate pools constrained by location and seniority filters (with fallback), rather than unconstrained retrieval over all 3,000 raw postings.
4. **Skill Boost Gains Inflated by Label/Signal Overlap**: Because the rule-based labeling rubric (`jobmatch-rubric-v1`) rewards skill regex matches, applying the skill boost directly reinforces the labeling criteria, artificially inflating observed ranking improvements.
5. **Synthetic Test Profiles**: The 30 candidate profiles are hand-authored synthetic benchmarks designed to test specific edge cases, while the 3,000 job postings in `data/jobs_clean.parquet` remain real LinkedIn postings.
6. **Effective Sample Size**: Due to regional and seniority correlations among test profiles, the effective sample size is lower than 30, requiring wide confidence intervals when interpreting statistical significance.
