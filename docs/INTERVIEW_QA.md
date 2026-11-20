# Technical Interview Q&A: Candid Engineering & Evaluation Deep-Dive

This document contains five candid, in-depth technical interview questions and answers regarding the architecture, empirical results, and methodological limitations of `jobmatch-eval`. Every answer uses verified numbers from `reports/` and directly addresses engineering trade-offs.

---

### Question 1: In your retrieval benchmark, BM25 achieved an nDCG@10 of 0.615 compared to 0.423 for Dense semantic retrieval. Does this prove that BM25 is the superior retriever for matching jobs to candidates?

**Answer:**
No, absolutely not. Stating that BM25 is "better" would be a fundamental misinterpretation of the evaluation methodology.

Here is what the numbers actually show:
Under the main pipeline configuration (location and seniority filters active, skill boost $w=0.2$), BM25 achieved an nDCG@10 of **0.6147 [0.52, 0.70]** versus Dense retrieval's **0.4231 [0.33, 0.52]**, representing a statistically significant difference of **+0.1916 [0.1109, 0.2610]** (`reports/retrieval_paired_comparisons.json`).

However, the reason for this gap lies in how the evaluation labels were constructed:
1. **Shared Lexical Signals**: All 1,037 relevance labels in `data/eval/qrels.json` were generated using a deterministic rule-based rubric (`jobmatch-rubric-v1`). This rubric assigns relevance based on title-token overlap, location matching, seniority matching, and technical skill regex counts.
2. **Retriever-Rubric Circularity**: BM25 scores documents based on exact token term frequencies and inverse document frequency. Because the rubric and BM25 share the exact same lexical markers, BM25's high ranking reflects alignment with the heuristic rubric, not superior matching for human job seekers.
3. **The Semantic Counter-Proof (Circularity Audit)**: To test whether dense retrieval possesses value that the rubric masks, we audited the subset of 51 relevant candidate postings across 17 profiles that share **zero literal keyword overlap** with candidate declared skills (`reports/retrieval_circularity_check.json`). On this slice, Dense retrieval achieved higher Recall@10 (**0.3931 [0.23, 0.57] vs BM25 0.2588 [0.12, 0.43]**) and higher MRR (**0.3437 [0.15, 0.53] vs BM25 0.1751 [0.07, 0.31]**).

**Conclusion**: BM25 wins on the main benchmark because the benchmark is lexically biased. Dense retrieval surfaces relevant opportunities when candidates use different terminology than the job description, but verifying this at scale requires human annotators who evaluate semantic equivalence rather than keyword presence.

---

### Question 2: You report a 12.98x speedup from incremental indexing. Does this mean your entire search system updates in under 13 seconds when new jobs arrive?

**Answer:**
No. The 12.98x speedup applies exclusively to the **ChromaDB dense vector index**, not the complete search system.

Here are the exact benchmark measurements:
In `reports/sync_benchmark.json`, we benchmarked a simulated 9% daily mutation scenario over 3,000 postings (150 descriptions modified, 60 new postings added, 60 deleted, and 2,790 unchanged):
- **Full Re-Index from Scratch**: Took **159.636 s $\pm$ 8.408 s** (18.79 docs/sec) because all 3,000 postings had to pass through `sentence-transformers/all-MiniLM-L6-v2` embedding inference.
- **Hash-Based Incremental Sync**: Took **12.295 s $\pm$ 0.973 s** (effective throughput of **244.0 docs/sec**), achieving a verified **12.98x speedup** with 100% verified state equivalence (0 hash mismatches).

**The Architectural Bottleneck**:
`rank_bm25.BM25Okapi` is an in-memory Python implementation that does not provide an incremental update, document insertion, or document deletion API. Whenever a mutation occurs:
- The dense index updates in 12.30 s via SHA-256 content hashes, recalculating embeddings only for the 210 mutated/added documents.
- But the BM25 inverted index and tokenized corpus must be **rebuilt in full**, which incurs a one-off build latency of **~5,992.1 ms (~6.0 s)** on CPU (`reports/retrieval_latency.json`).

**Production Reality**: The system is hybrid. Dense updates are incremental, but lexical updates require a full rebuild. If an indexing architecture were scaled to serve 1,000,000 postings, `rank_bm25` would be completely unviable and would need to be replaced with an incremental inverted index engine like Tantivy or Lucene.

---

### Question 3: Why did you evaluate retrieval within candidate pools constrained by filters, and what happened when you turned the filters off?

**Answer:**
We evaluated retrieval inside filtered candidate pools because in product search, unconstrained retrieval over an entire corpus produces operationally irrelevant results (e.g. surfacing a Senior Director role in Tokyo for an Entry-level candidate in Chicago). However, doing so exposed a major evaluation vulnerability when filters were removed.

**The Candidate Pooling Setup**:
In Phase 2A, we pooled the top 20 candidates retrieved by each of our four retrievers (BM25, Dense, Hybrid RRF, Cross-Encoder) under active location and seniority filters (with fallback), yielding **1,037 unique candidate pairs** across 30 profiles (`reports/labeling_method.json`).

**The Evaluation Collapse (Pure Retrieval)**:
When we disabled location and seniority filters in the "pure retrieval" setting (`reports/retrieval_metrics_pure.json`):
- Top-10 results across all strategies experienced an unjudged rate of **76.0% to 85.3%** (BM25: 85.3%, Dense: 76.0%, Hybrid RRF: 82.3%, Cross-Encoder: 79.3%).
- In standard information retrieval benchmarks, unjudged documents are treated as non-relevant (label 0).
- Consequently, Precision@5 plummeted from ~0.52 to **0.127–0.160**, and nDCG@10 dropped from ~0.61 to **0.161–0.186**.

**Key Takeaway**:
These low "pure retrieval" numbers do not mean that the retrievers failed to identify technically relevant jobs. Rather, out-of-market jobs that were retrieved had never been pooled or judged in Phase 2A. This demonstrates that offline IR evaluation metrics are valid only within the exact candidate pooling distribution under which judgments were collected.

---

### Question 4: You evaluated LLM explanation quality without an LLM judge or human annotators. How did you do this, what were the results, and what caused the groundedness failures?

**Answer:**
We evaluated explanation quality using 100% deterministic programmatic verification against the raw job posting text, avoiding both the circularity of an LLM judge and the subjectivity of unblinded human review.

**Evaluation Framework**:
Using `gemini-3.1-flash-lite` on the top-5 BM25 matches across all 30 profiles (150 calls, logged in `reports/llm_outputs.jsonl`):
1. **Schema Validity**: Enforced structured Pydantic output (`matched_skills`, `missing_skills`, and a single-sentence `summary`). First-try validity was **100.00%** (150/150 valid, 0 retries, 0 fallbacks), establishing a 95% failure upper bound of **$\le 2.00\%$** via the rule of three ($\le 3/150$).
2. **Groundedness**: We verified whether every skill claimed in `matched_skills` literally existed in the job description using boundary-safe regex `(?<![\w\+\#]){skill}(?![\w\+\#])`.
3. **Profile Support**: We checked whether claimed matched skills existed in the candidate's declared profile (**92.42% verified**).
4. **Contradiction Leakage**: We measured whether the model mistakenly claimed a skill was "missing" when it actually existed in both the candidate profile and the job description (**0.31%**, 4/1,278 skills).

**Groundedness Failure Analysis (277 Claimed Skills)**:
- **Baseline Groundedness**: **91.70% pooled** (254/277 skills) and **91.79% macro-average** [0.88, 0.96].
- **Joined-Word Formatting Artifacts (7 skills)**: In the raw LinkedIn HTML data, adjacent HTML tags were occasionally joined without spaces during upstream scraping (e.g. `JavaScriptUnderstanding`, `TableauExperience`, `REST APIsExperience`). When we accounted for these joined words, groundedness rose to **94.22% pooled** (261/277 skills) and **94.31% macro-average** [0.90, 0.98].
- **True Hallucinations / Leaps (16 skills, 5.78%)**: The model occasionally inferred high-level business capabilities not mentioned in the text (e.g. claiming `Strategic Planning`, `Budgeting`, `Roadmapping`, or `PowerBI` for general product or engineering management roles).

**The Major Limitation**:
Boundary-safe regex tests literal presence, not semantic polarity. If a job posting states "We do not use Kubernetes", Kubernetes registers as literally grounded. Full contextual verification requires Natural Language Inference (NLI) models or human reviewers.

---

### Question 5: If you had 3 more months and dedicated budget to transition this research prototype to an interactive system, what architectural decisions would you change, and what would you rebuild first?

**Answer:**
If given 3 months and dedicated resources, I would prioritize four fundamental architectural upgrades:

1. **Conduct Real Human Relevance Evaluations**:
   - The single largest limitation of this project is that all 1,037 relevance labels are rule-based heuristics (`human_reviewed: false`).
   - I would execute the blinded review package (`labels_to_review.csv`) with recruited technical recruiters and software engineers, calculate Cohen's weighted kappa to measure true inter-annotator agreement, and train a learned reranker on human preferences.

2. **Replace `rank_bm25` with an Incremental Search Engine**:
   - The inability of `rank_bm25` to support incremental updates forces a 6-second complete index rebuild whenever postings change.
   - I would replace `rank_bm25` with **Tantivy** (a fast, Rust-based Lucene alternative with Python bindings) or **Elasticsearch/OpenSearch**, allowing true incremental document insertion, update, and deletion across both lexical and dense streams.

3. **Solve Niche Domain Data Sparsity**:
   - In our 3,000-job sample, 6 of 30 profiles had fewer than 3 relevant documents (`prof_07` Mobile Developer and `prof_23` Computer Vision had 0 relevant documents under strict filtering).
   - I would scale the corpus from 3,000 to 100,000+ postings with stratified sampling across technical disciplines to ensure robust statistical power for specialized engineering domains.

4. **NLI-Based Groundedness & Candidate Engagement Analytics**:
   - I would replace literal regex groundedness with an entailment-based NLI model (e.g., DeBERTa-v3-large) to detect negation and exclusionary requirements.
   - Finally, I would implement an online A/B testing framework to measure Click-Through Rates (CTR), application conversion rates, and interview invitation rates rather than relying solely on offline nDCG metrics.

---

### Question 6: How does your conversational agent preserve candidate preferences across multiple turns, and what were the measured results?

**Answer:**
The conversational agent (`src/agent.py`) integrates explicit session state management decoupled from the LLM context window.

**Session Memory Architecture**:
- Each conversation is assigned a unique `session_id` backed by a persistent session file in `data/sessions/{session_id}.json`.
- State tracking maintains an `active_filters` dictionary capturing:
  - `location`: e.g., `"Chicago, IL"`, `"Remote"`
  - `level`: e.g., `"Entry level"`, `"Mid-Senior level"`
  - `excluded_companies`: blacklist populated via negative statements (e.g. `"not interested in sales or defense contractors"`)
  - `preferred_skills`: technical competencies declared or emphasized across conversation turns
- When a candidate modifies a constraint (e.g. stating *"actually, only show me remote positions"*), the agent invokes `refine_filters` or passes updated parameters to `search_jobs`, updating the persistent state.
- In subsequent turns, even when the user asks distracting or follow-up questions (e.g., inspecting specific job details or inquiring about interview preparation), the agent merges persisted filters with new query intents.

**Empirical Evaluation**:
- Across 10 scripted scenarios comprising 28 total turns (`reports/agent_eval.json`), session memory was evaluated programmatically against 23 explicit state assertions.
- **Memory Retention Rate**: **100.0% (23/23 checks passed)**. The agent never lost location constraints, seniority boundaries, or company exclusions across distracting turns.

---

### Question 7: Why did you enforce a strict 5-step loop cap on the agent tool execution, and how did it perform in practice?

**Answer:**
Autonomous tool-calling agents are prone to runaway loops, cascading retries, and unbounded latency when facing ambiguous user queries or empty tool results.

**The Engineering Rationale**:
1. **Preventing Autonomous Thrashing**: If a search returns zero results, an unconstrained model might autonomously attempt repeated searches with minor keyword variations, burning through API rate limits and generating runaway token costs.
2. **Predictable Turn Latency**: Every tool step incurs upstream API inference and execution time. Capping the loop at `MAX_STEPS_PER_TURN = 5` guarantees a deterministic worst-case computational bound per turn.
3. **Safety Fallback**: If the agent reaches 5 tool invocations without synthesizing a final user response, the turn safely halts, outputs the best accumulated results, and returns control to the candidate rather than hanging or crashing.

**Empirical Performance**:
- Across all evaluated turns in `reports/agent_eval.json`:
  - **Mean Tool Steps per Turn**: **2.14 steps** (typically 1 filter update or search followed by 1 inspection call).
  - **Maximum Observed Steps**: Exactly 3 steps in any turn (verified distribution across 28 turns: 2 turns with 1 step, 20 turns with 2 steps, 6 turns with 3 steps), well below the 5-step ceiling.
  - **Turn Latency**: Mean end-to-end turn latency was **10,828.3 ms (~10.8 s)** with a p50 of ~5.9 s, though worst-case multi-step turns with retries reached **41,697.0 ms (~41.7 s)**.
  - **Tool Call Success Rate**: **96.9%** (31/32 calls; the single failure was an intentional negative test verifying graceful error handling when querying an invalid job ID).

---

### Question 8: Why did you choose scripted multi-turn scenarios instead of open-ended conversational evaluation or an LLM-as-a-judge?

**Answer:**
We evaluated the agent using 10 deterministic, scripted multi-turn scenarios (`tests/agent_scenarios.json`, 28 total turns) rather than subjective LLM-as-a-judge grading or unconstrained chat logs.

**Why Scripted Programmatic Evaluation**:
1. **Deterministic Verification**: LLM-as-a-judge suffers from prompt sensitivity, self-enhancement bias, and lack of reproducibility. By authoring scripted scenarios with programmatic assertions, we directly inspect:
   - Tool call arguments (e.g., verifying `location` matches exactly).
   - Session state persistence across turns.
   - Company exclusion blacklists.
   - Resiliency when tools return HTTP 404 or invalid IDs.
2. **Repeatability**: Scenarios can be re-run across model checkpoints, prompt revisions, or temperature configurations with identical pass/fail grading.

**Honest Evaluation Numbers**:
- **8/10 on the first run**:
  - Across the 10 scenarios (28 total turns), the agent achieved **8/10 (80.0%) on the first run**.
  - `scenario_08` failed on Turn 2 due to a transient upstream API quota error (`Gemini API request failed after retries`) and passed on rerun with identical inputs, yielding a **9/10 (90.0%) pass rate on rerun**.
  - `scenario_10` suffered a genuine behavioral failure.
- **Methodological Limitations**:
  - 10 scenarios and 28 turns represent a small, hand-crafted test suite.
  - While it rigorously verifies boundary logic and memory state, it cannot capture the infinite diversity, typos, idioms, and conversational ambiguity of natural human dialogue.

---

### Question 9: What happened in Scenario 10 where the agent relaxed the location constraint to Remote without asking the user to confirm, and why did you keep the failure instead of modifying the test?

**Answer:**
`scenario_10` evaluates multi-constraint compounding across 4 turns. In this scenario, the synthetic user specifies Mid-Senior level roles in Chicago, IL, adds skill preferences, and asks for recommendations.

**The Empirical Finding**:
1. **Zero-Result Context**: In our 3,000-job sample from LinkedIn, there are zero Mid-Senior software engineering postings located in Chicago, IL. The model had noted the lack of Chicago results in earlier turns.
2. **Constraint Relaxation**:
   - In Turn 4, the session state correctly preserved both `location: "Chicago, IL"` and `level: "Mid-Senior level"`.
   - However, facing an empty result set for Chicago, the model relaxed the location constraint to Remote without asking the user to confirm: it passed `location="Remote"` to `search_jobs`, while Mid-Senior level was applied through session memory (returning remote mid-senior postings).
3. **The Programmatic Failure**:
   - The test assertion strictly required:
     `tool_arg_contains: {"location": "Chicago, IL", "level": "Mid-Senior level"}`.
   - Because `location` was `"Remote"`, the programmatic assertion failed.

**Why We Kept the Failure**:
- Modifying test assertions to accommodate model drift is benchmark tampering.
- The failure exposed a critical agentic failure mode: relaxed the location constraint to Remote without asking the user to confirm. While the model had mentioned the lack of Chicago results in earlier turns, an autonomous agent should not relax a geographic requirement without explicit user confirmation.
- We documented this failure and inspection note honestly in `reports/agent_eval.json`, `README.md`, `DECISIONS.md`, and this interview guide.
