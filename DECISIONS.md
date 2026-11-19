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
  - Use SHA-256 hash of `f"{normalized_title}||{normalized_description}"` stored as `content_hash` in the document metadata.
  - Persist vector index in local ChromaDB at `chroma_db/` configured with HNSW cosine distance metric.
  - Explicitly disable anonymous telemetry (`ANONYMIZED_TELEMETRY=False`) and enable offline mode (`HF_HUB_OFFLINE=1`) to prevent external network calls, ensuring fast, reproducible execution within sandboxed container environments.

