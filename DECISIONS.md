# Technical Decisions & Architecture Log

This document records all engineering decisions, model justifications, trade-offs, and design choices made across project phases.

---

## Decision 1: Dataset Selection & Provenance (Phase 1A)

- **Context**: The project requires a real-world, publicly accessible job postings dataset with complete job titles, employer/company names, locations, rich descriptions, and career seniority/experience levels. Silent synthetic data generation is strictly prohibited.
- **Evaluated Options**:
  1. *Kaggle LinkedIn Job Postings (2023–2024 by Arsh Koneru)*: Comprehensive, but requires Kaggle API credentials (`kaggle.json` or username/key), adding onboarding friction.
  2. *Zalizedata ATS Snapshot (Hugging Face)*: Openly accessible and lightweight (3.3 MB), but contains only job titles and extracted skills without full narrative job descriptions.
  3. *DataStax LinkedIn Job Listings (Hugging Face `datastax/linkedin_job_listings`)*: Direct Hugging Face mirror of the Kaggle LinkedIn Job Postings dataset.
- **Decision**: Selected `datastax/linkedin_job_listings` hosted openly on Hugging Face.
- **Justification**:
  - **Licensing**: Covered under Creative Commons Attribution-ShareAlike 4.0 International (`CC BY-SA 4.0`) and MIT terms from the upstream dataset contributors.
  - **Zero Credential Friction**: Openly downloadable without gated access tokens or proprietary API keys.
  - **Data Completeness**: Directly provides `title`, `company_name`, `location`, full multi-paragraph `description`, and `formatted_experience_level`.
  - **Scale & Realism**: Derived from authentic production LinkedIn listings, containing realistic market noise, HTML formatting artifacts, and varied description structures.

---

## Decision 2: Embedding Model Selection (Phase 1A)

- **Context**: The retrieval architecture requires a free, locally hosted sentence/passage embedding model capable of running efficiently on consumer CPU hardware without requiring external API keys (e.g. OpenAI/Cohere).
- **Evaluated Options**:
  1. *`BAAI/bge-small-en-v1.5`*: Strong MTEB benchmark performance, 384 dimensions, ~133MB footprint. Slightly slower tokenization and inference throughput on standard CPU architectures.
  2. *`sentence-transformers/all-mpnet-base-v2`*: Higher semantic fidelity (768 dimensions), but significantly heavier (~420MB), with ~3-4x higher inference latency on CPU, creating unacceptable response times for interactive retrieval experiments.
  3. *`sentence-transformers/all-MiniLM-L6-v2`*: Standard industry benchmark baseline for lightweight dense retrieval. 384 embedding dimensions, 22.7M parameters, ~80MB footprint.
- **Decision**: Selected `sentence-transformers/all-MiniLM-L6-v2`.
- **Justification**:
  - **Latency / Throughput**: Generates embeddings on CPU in <10ms per document, enabling batch indexing of 3,000 full job postings in under 2 minutes.
  - **Memory Footprint**: Extremely low RAM usage (~250MB during inference), easily deployable inside constrained containers.
  - **Downstream Compatibility**: Produces normalized 384-dimensional unit vectors directly compatible with ChromaDB's HNSW cosine distance index.

---

## Decision 3: Text Preprocessing & Cleaning Strategy (Phase 1A)

- **Context**: Raw scraped job postings contain HTML formatting tags (`<p>`, `<strong>`, `<ul>`, `<li>`), character entities (`&amp;`, `&quot;`, `&nbsp;`), irregular multi-line spacing, and duplicate entries.
- **Decision**:
  - Strip HTML tags via compiled regular expressions combined with Python's standard `html.unescape`. This avoids DOM tree parsing overhead while reliably removing formatting tags across varied ATS outputs.
  - Normalize multiple spaces, tabs, and line-breaks into single spaces.
  - Discard any records where either the title or description is missing, whitespace-only, or empties out post-cleaning.
  - Deduplicate on the normalized composite key `(title, company, description)`.
  - Fix random seed to `42` for all sampling operations (default sample size: 3,000 postings).

---

## Decision 4: Metadata Normalization & Heuristics (Phase 1A)

- **Context**: Seniority levels and locations in scraped data are frequently inconsistent or unpopulated.
- **Experience Level Mapping**:
  - Standardized into six canonical tiers: `Internship`, `Entry level`, `Mid-Senior level`, `Director`, `Executive`, and `Not Specified`.
  - Explicit raw levels mapped directly (e.g., `Associate` and `Entry level` -> `Entry level`).
  - When raw seniority is missing/empty, apply conservative regex heuristics to the job title (`\b(senior|sr|lead|staff|principal)\b` -> `Mid-Senior level`; `\b(junior|jr|entry|associate)\b` -> `Entry level`; `\b(intern|internship|trainee)\b` -> `Internship`; `\b(director|vp)\b` -> `Director`; `\b(chief|executive|ceo|cto)\b` -> `Executive`). If no heuristic matches, classify as `Not Specified`.
- **Location Mapping**:
  - Jobs explicitly mentioning remote in location or title are normalized to `Remote`.
  - Standard US city/state entries are normalized to `"City, ST"` format.

---

## Decision 5: Content Hashing & Vector Storage (Phase 1A)

- **Context**: Vector indices need traceability back to source documents, change detection, and reproducible local storage.
- **Decision**:
  - Use SHA-256 hash of `f"{normalized_title}||{normalized_description}||{normalized_location}||{normalized_level}"` stored as `content_hash` in the document metadata.
  - Persist vector index in local ChromaDB at `chroma_db/` configured with HNSW cosine distance metric.
  - Explicitly disable anonymous telemetry (`ANONYMIZED_TELEMETRY=False`) and enable offline mode (`HF_HUB_OFFLINE=1`) to prevent external network calls, ensuring fast, reproducible execution within sandboxed container environments.

---

## Decision 6: Incremental Synchronization Strategy (Phase 1B)

- **Context**: In production job platforms, hundreds of thousands of postings exist, but daily delta changes are small (<5-10%). A naive full re-index wastes immense GPU/CPU compute, creates indexing downtime, and causes query latency spikes.
- **Evaluated Options**:
  1. *Blind Periodic Re-Index*: Re-embed all documents on a cron schedule. Computationally wasteful (O(N) embedding calls every cycle).
  2. *Timestamp-Based Sync*: Rely on `updated_at` timestamps from source feeds. Vulnerable to clock skew, false-positive updates (e.g. metadata touched without text change), and silent missed updates.
  3. *Content-Hash State Comparison (Selected)*: Compare SHA-256 hash over the exact fields that affect embedding quality (`title`, `description`, `location`, `experience_level`).
- **Decision**: Implemented content-hash state synchronization (`src/sync_index.py`).
- **Justification**:
  - **Zero Unnecessary Embeddings**: Exactly 0 embedding calls for unchanged documents.
  - **Atomic Deletions**: Detects and purges documents absent from new snapshots.
  - **Empirical Validation**: Benchmarking on 3,000 postings with realistic daily turnover (5% modified, 2% added, 2% deleted) showed an **11.05x speedup** (15.09s vs 166.81s) while guaranteeing 100% state equivalence with a fresh index.

---

## Decision 7: Multi-Strategy Retrieval, Personalization, and Ranking Hierarchy (Phase 1C)

- **Context**: Candidates possess varied criteria (exact technical keywords, implicit domain context, location constraints, and seniority preferences). A single retrieval strategy is insufficient: lexical search misses synonyms; dense bi-encoders suffer from capacity limits and miss hard keywords; and strict metadata filtering risks returning zero results in specialized sub-niches.
- **Evaluated Strategies**:
  1. *BM25 Only*: Fast, exact token matching via `rank_bm25` with domain-aware regex tokenization preserving technical symbols (`C++`, `C#`, `.NET`, `Python-3`, `node.js`).
  2. *Dense Only*: Semantic vector similarity via ChromaDB HNSW cosine index over `all-MiniLM-L6-v2` embeddings. Captures latent conceptual relevance.
  3. *Hybrid RRF (Reciprocal Rank Fusion)*: Combines ranked lists from BM25 and Dense with parameter $k_{rrf} = 60$:
     $$RRF(d) = \sum_{s \in \{\text{bm25}, \text{dense}\}} \frac{1}{60 + \text{rank}_s(d)}$$
     RRF is scale-invariant and immune to disparate score distributions (BM25 raw unbounded scores vs dense cosine similarity).
  4. *Hybrid + Cross-Encoder Re-Ranking*: High-recall stage fuses top candidates (depth = 50), followed by `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring full cross-attention between candidate profile query and job text, normalized via sigmoid.
- **Three-Tier Metadata Filter Fallback Hierarchy & Tier Preservation**:
  - Hard filters can return empty result sets if no postings match both location and seniority.
  - *Tier 1 (Strict)*: Strict match on candidate's preferred location (or `"Remote"`) AND seniority level (or `"Not Specified"`). Remote detection checks location string, title keywords (`remote`, `virtual`, `wfh`), and description phrases (`100% remote`, `work from home`).
  - *Tier 2 (Relaxed Seniority)*: If Tier 1 yields $< k$ results, retain location constraint but relax seniority to avoid geographic mismatches.
  - *Tier 3 (Global Backfill)*: If Tier 2 yields $< k$ results, backfill remaining slots from top unfiltered candidates, guaranteeing exactly $\min(k, N)$ results without crashing or returning empty lists.
  - *Tier Sorting Priority*: Candidates are strictly sorted by `(filter_tier, -final_score)` ensuring Tier 1 strict matches always precede Tier 2 and Tier 3 results.
- **Explainable Skill Boosting & Score Normalization**:
  - Base scores from heterogeneous retrieval strategies (BM25 unbounded [0, 120+], Dense cosine [0, 1], RRF [0, 0.04]) are min-max normalized over candidate pools to $[0, 1]$ before boosting.
  - Formula: $\text{final\_score} = \text{norm\_base\_score} + w \times \frac{|\text{matched\_skills}|}{|\text{candidate\_skills}|}$ with default $w = 0.2$.
  - Min-max scaling ensures weight $w$ has an identical relative meaning (+20% of list range) across all 4 strategies, preventing BM25 raw scores from drowning out skill matches.
  - Regex boundary matching `(?<![\w\+\#]){skill}(?![\w\+\#])` prevents false positives (e.g., `C` inside `Cloud` or `Java` inside `JavaScript`) while supporting programming languages with symbols (`C++`, `C#`, `.NET`).
  - Every `SearchResult` explicitly carries `filter_tier: int` (1, 2, or 3), `raw_score`, `base_score`, `skill_boost`, and `matched_skills`.

---

## Decision 9: LLM Explanation Model & Groundedness Evaluation Architecture (Phase 2C)

- **Context**: In Phase 2C, candidate job matches are explained using an LLM to generate structured summaries: matched skills (which must be grounded in the posting text), missing skills, and a one-sentence rationale. The evaluation requires measuring schema validity, literal groundedness, profile support, and contradiction leakage without relying on an LLM judge.
- **Model Selection**:
  - **Date Checked**: 2026-11-20
  - **Official Docs URL**: `https://ai.google.dev/gemini-api/docs/models`
  - **Pricing Docs URL**: `https://ai.google.dev/pricing` (Checked: 2026-11-20: Free tier available for Gemini Flash-Lite with rate-controlled pacing; live project quotas visible in Google AI Studio console)
  - **Model**: `gemini-3.1-flash-lite` (fallback: `gemini-3.5-flash-lite`, `gemini-3.8-flash`)
  - **Justification**:
    - Querying the official Google AI Studio API (`/v1beta/models`) confirms that older models (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`) have been sunset/deprecated for new API users.
    - `gemini-3.1-flash-lite` delivers sub-second p50 execution (~970 ms) and native structured JSON output via `application/json` response MIME type.
    - Operates within free-tier project rate limits with deterministic exponential backoff and persistent disk caching.
- **Groundedness Verification Methodology (NO LLM Judge)**:
  - Groundedness is computed strictly programmatically against the raw job text (`title` + `description`).
  - **Boundary-Safe Matching**: Uses regex pattern `(?<![\w\+\#]){escaped}(?![\w\+\#])` with `re.IGNORECASE` to accommodate technical symbols (`C++`, `C#`, `.NET`) while preventing substring false positives (`C` in `Cloud`, `Java` in `JavaScript`).
  - **Strict Groundedness**: Exact case-sensitive substring matching with no normalization.
  - **Contradiction Leakage**: Quantifies whether skills listed under `missing_skills` actually appear in both the job description and the candidate profile.
  - **Cluster Bootstrap CIs**: Bootstrap resampling ($N=1,000$ iterations, seed 42) is performed by clustering across profiles (not independent calls) to account for intra-profile correlation.




