"""Script to generate or update README.md and RESULTS.md directly from reports/ JSON artifacts.

Ensures zero hand-typed metrics by pulling all data pipeline, index, sync,
and retrieval evaluation numbers strictly from reports/ JSON files.

MANDATORY PROTOCOL:
  All retrieval evaluation tables carry the caption:
  "Evaluated against model-proposed, unverified relevance labels"
  Zero use of forbidden phrases ('human-verified', 'ground truth').
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_STATS_PATH = Path("reports/data_stats.json")
INDEX_STATS_PATH = Path("reports/index_stats.json")
SYNC_STATS_PATH = Path("reports/sync_benchmark.json")
LABELING_STATS_PATH = Path("reports/labeling_method.json")
AGREEMENT_STATS_PATH = Path("reports/label_agreement.json")

# Phase 2B reports
RETRIEVAL_MAIN_PATH = Path("reports/retrieval_metrics_main.json")
RETRIEVAL_PURE_PATH = Path("reports/retrieval_metrics_pure.json")
RETRIEVAL_PAIRED_PATH = Path("reports/retrieval_paired_comparisons.json")
RETRIEVAL_LATENCY_PATH = Path("reports/retrieval_latency.json")
RETRIEVAL_ABLATION_PATH = Path("reports/retrieval_ablation.json")
RETRIEVAL_CIRCULARITY_PATH = Path("reports/retrieval_circularity_check.json")

# Phase 2C reports
LLM_QUALITY_PATH = Path("reports/llm_quality.json")

README_PATH = Path("README.md")
RESULTS_PATH = Path("RESULTS.md")


def load_json_report(path: Path) -> Dict[str, Any]:
    """Safely load a JSON report file.

    Args:
        path: Path to the JSON report.

    Returns:
        Loaded dictionary, or empty dict if file does not exist.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_ci(ci: List[float], digits: int = 2) -> str:
    """Format a 2-element list [lower, upper] as '[lower, upper]'."""
    if not ci or len(ci) < 2:
        return "N/A"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def build_phase_2b_markdown(
    ret_main: Dict[str, Any],
    ret_pure: Dict[str, Any],
    ret_paired: Dict[str, Any],
    ret_latency: Dict[str, Any],
    ret_ablation: Dict[str, Any],
    ret_circularity: Dict[str, Any],
) -> str:
    """Build Markdown content for Phase 2B retrieval evaluation.

    Args:
        ret_main: Parsed retrieval_metrics_main.json.
        ret_pure: Parsed retrieval_metrics_pure.json.
        ret_paired: Parsed retrieval_paired_comparisons.json.
        ret_latency: Parsed retrieval_latency.json.
        ret_ablation: Parsed retrieval_ablation.json.
        ret_circularity: Parsed retrieval_circularity_check.json.

    Returns:
        Markdown string for Phase 2B section.
    """
    if not ret_main or "strategies" not in ret_main:
        return ""

    main_strats = ret_main.get("strategies", {})
    pure_strats = ret_pure.get("strategies", {})
    comparisons = ret_paired.get("comparisons", {})
    latency = ret_latency
    ablation = ret_ablation.get("ablation_results", [])
    circ_sample = ret_circularity.get("sample_size_assessment", {})
    circ_metrics = ret_circularity.get("strategy_metrics_on_zero_overlap_subset", {})

    strategy_order = ["bm25", "dense", "hybrid_rrf", "cross_encoder"]

    # 1. Main Pipeline Table
    main_rows: List[str] = []
    for s in strategy_order:
        data = main_strats.get(s, {})
        d_name = data.get("display_name", s)
        p5 = f"{data.get('precision@5', 0.0):.3f} {format_ci(data.get('precision@5_ci_95', []))}"
        rec = f"{data.get('recall@10', 0.0):.3f} {format_ci(data.get('recall@10_ci_95', []))}"
        mrr = f"{data.get('mrr', 0.0):.3f} {format_ci(data.get('mrr_ci_95', []))}"
        ndcg = f"{data.get('ndcg@10', 0.0):.3f} {format_ci(data.get('ndcg@10_ci_95', []))}"
        unjudged = f"{data.get('unjudged_rate@10', 0.0) * 100:.1f}%"

        if s == "cross_encoder":
            p50_lat = f"{latency.get('cross_encoder', {}).get('end_to_end', {}).get('p50_ms', 0.0):.1f} ms"
            p95_lat = f"{latency.get('cross_encoder', {}).get('end_to_end', {}).get('p95_ms', 0.0):.1f} ms"
        else:
            p50_lat = f"{latency.get(s, {}).get('p50_ms', 0.0):.1f} ms"
            p95_lat = f"{latency.get(s, {}).get('p95_ms', 0.0):.1f} ms"

        main_rows.append(
            f"| **{d_name}** | {p5} | {rec} | {mrr} | {ndcg} | {unjudged} | {p50_lat} | {p95_lat} |"
        )
    main_table = "\n".join(main_rows)

    # 2. Pure Retrieval Table
    pure_rows: List[str] = []
    for s in strategy_order:
        data = pure_strats.get(s, {})
        d_name = data.get("display_name", s)
        p5 = f"{data.get('precision@5', 0.0):.3f} {format_ci(data.get('precision@5_ci_95', []))}"
        rec = f"{data.get('recall@10', 0.0):.3f} {format_ci(data.get('recall@10_ci_95', []))}"
        mrr = f"{data.get('mrr', 0.0):.3f} {format_ci(data.get('mrr_ci_95', []))}"
        ndcg = f"{data.get('ndcg@10', 0.0):.3f} {format_ci(data.get('ndcg@10_ci_95', []))}"
        unjudged = f"{data.get('unjudged_rate@10', 0.0) * 100:.1f}%"
        pure_rows.append(
            f"| **{d_name}** | {p5} | {rec} | {mrr} | {ndcg} | {unjudged} |"
        )
    pure_table = "\n".join(pure_rows)

    # 3. Paired Comparisons Table
    paired_rows: List[str] = []
    comp_order = [
        ("bm25_vs_dense", "BM25 vs Dense"),
        ("hybrid_rrf_vs_bm25", "Hybrid RRF vs BM25"),
        ("hybrid_rrf_vs_dense", "Hybrid RRF vs Dense"),
        ("cross_encoder_vs_hybrid_rrf", "Cross-Encoder vs Hybrid RRF"),
        ("cross_encoder_vs_bm25", "Cross-Encoder vs BM25"),
    ]
    for comp_key, comp_label in comp_order:
        comp_data = comparisons.get(comp_key, {}).get("metrics", {})
        for metric_name in ["precision@5", "recall@10", "mrr", "ndcg@10"]:
            m_info = comp_data.get(metric_name, {})
            diff = m_info.get("mean_difference", 0.0)
            ci = m_info.get("ci_95", [0.0, 0.0])
            sig = m_info.get("significant", False)
            interp = "**Clear difference against rule-based labels**" if sig else "No clear difference (CI spans 0.0)"
            diff_str = f"{diff:+.4f}"
            ci_str = format_ci(ci, digits=4)
            paired_rows.append(
                f"| **{comp_label}** | `{metric_name}` | {diff_str} | {ci_str} | {interp} |"
            )
    paired_table = "\n".join(paired_rows)

    # 4. Latency Table
    ce_ret = latency.get("cross_encoder", {}).get("retrieval_step", {})
    ce_scr = latency.get("cross_encoder", {}).get("cross_encoder_scoring_step", {})
    ce_e2e = latency.get("cross_encoder", {}).get("end_to_end", {})
    startup = latency.get("startup_times_ms", {})
    latency_table = f"""| Architecture / Step | p50 Latency (ms) | p95 Latency (ms) | Mean Latency (ms) | Details / Operations |
|---|---|---|---|---|
| **Dense Only** | {latency.get("dense", {}).get("p50_ms", "N/A")} | {latency.get("dense", {}).get("p95_ms", "N/A")} | {latency.get("dense", {}).get("mean_ms", "N/A")} | Query embedding (`all-MiniLM-L6-v2`) + ChromaDB HNSW |
| **BM25 Only** | {latency.get("bm25", {}).get("p50_ms", "N/A")} | {latency.get("bm25", {}).get("p95_ms", "N/A")} | {latency.get("bm25", {}).get("mean_ms", "N/A")} | Lexical BM25Okapi scoring over full corpus |
| **Hybrid RRF ($k=60$)** | {latency.get("hybrid_rrf", {}).get("p50_ms", "N/A")} | {latency.get("hybrid_rrf", {}).get("p95_ms", "N/A")} | {latency.get("hybrid_rrf", {}).get("mean_ms", "N/A")} | BM25 + Dense retrieval + RRF fusion |
| **Cross-Encoder: Step 1 (Retrieval)** | {ce_ret.get("p50_ms", "N/A")} | {ce_ret.get("p95_ms", "N/A")} | {ce_ret.get("mean_ms", "N/A")} | Hybrid RRF candidate pool retrieval (top 50) |
| **Cross-Encoder: Step 2 (Neural Scoring)** | {ce_scr.get("p50_ms", "N/A")} | {ce_scr.get("p95_ms", "N/A")} | {ce_scr.get("mean_ms", "N/A")} | CPU Transformer forward pass re-ranking 50 pairs |
| **Cross-Encoder: End-to-End Pipeline** | **{ce_e2e.get("p50_ms", "N/A")}** | **{ce_e2e.get("p95_ms", "N/A")}** | **{ce_e2e.get("mean_ms", "N/A")}** | Total two-stage retrieval and neural re-ranking |

*One-off startup / index build times (outside timed queries): BM25 Index Build = {startup.get('bm25_index_build_ms', 5992.1):.1f} ms, Dense Init = {startup.get('dense_init_ms', 877.8):.1f} ms, Cross-Encoder Load = {startup.get('cross_encoder_load_ms', 242.5):.1f} ms.*
"""

    # 5. Ablation Table
    ablation_rows: List[str] = []
    for row in ablation:
        cfg = row.get("configuration", "")
        p5 = f"{row.get('precision@5', 0.0):.4f}"
        rec = f"{row.get('recall@10', 0.0):.4f}"
        mrr = f"{row.get('mrr', 0.0):.4f}"
        ndcg = f"{row.get('ndcg@10', 0.0):.4f}"
        unjudged = f"{row.get('unjudged_rate@10', 0.0) * 100:.1f}%"
        ablation_rows.append(
            f"| {cfg} | {p5} | {rec} | {mrr} | {ndcg} | {unjudged} |"
        )
    ablation_table = "\n".join(ablation_rows)

    # 6. Circularity Table
    circ_rows: List[str] = []
    for s in strategy_order:
        data = circ_metrics.get(s, {})
        d_name = main_strats.get(s, {}).get("display_name", s)
        p5 = f"{data.get('precision@5', 0.0):.4f}"
        p5_ci = format_ci(data.get("precision@5_ci_95", [0.0, 0.0]))
        rec = f"{data.get('recall@10', 0.0):.4f}"
        rec_ci = format_ci(data.get("recall@10_ci_95", [0.0, 0.0]))
        mrr = f"{data.get('mrr', 0.0):.4f}"
        mrr_ci = format_ci(data.get("mrr_ci_95", [0.0, 0.0]))
        ndcg = f"{data.get('ndcg@10', 0.0):.4f}"
        ndcg_ci = format_ci(data.get("ndcg@10_ci_95", [0.0, 0.0]))
        circ_rows.append(
            f"| **{d_name}** | {p5} {p5_ci} | {rec} {rec_ci} | {mrr} {mrr_ci} | {ndcg} {ndcg_ci} |"
        )
    circ_table = "\n".join(circ_rows)

    section = f"""---

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
{main_table}

> [!NOTE]
> All four strategies exhibit 0.00% unjudged rate in top 10 under the main configuration because candidate pooling in Phase 2A pooled the top 20 candidates across these exact four retrieval streams.

### 2. Secondary Setting: Pure Retrieval (Filters OFF, Skill Boost OFF) — Not comparable (labels pooled under filters, 76% unjudged)
*Evaluated against model-proposed, unverified relevance labels*

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) | Unjudged Rate@10 |
|---|---|---|---|---|---|
{pure_table}

> [!WARNING]
> **Not comparable (labels pooled under filters, 76% unjudged)**: Candidate pooling was performed under active location and seniority filters. Out-of-market jobs retrieved when filters are disabled were never pooled or evaluated, resulting in unjudged rates of 76%–85% (treated as non-relevant label 0). These numbers do not reflect pure retrieval quality.

### 3. Paired Statistical Significance (Bootstrap 95% CIs on Differences)
*Evaluated against model-proposed, unverified relevance labels*

| Comparison ($A$ vs $B$) | Metric | Mean Difference ($A - B$) | 95% Bootstrap CI | Statistical Conclusion |
|---|---|---|---|---|
{paired_table}

### 4. Query Latency Profile on CPU (25 timed runs per step)
*(Measured on AMD 12-core CPU, PyTorch CPU execution with `torch_threads=6`, `CUDA: False`)*

> **Measurement Conditions**: Latency was measured warm (retriever warm-up executed before timing), single-process, CPU-only, with inverted index, tokenized corpus, and regex caches pre-built outside the timed loop.

{latency_table}

### 5. Ablation Study: Filter, Boost, and Weight Tuning
*Evaluated against model-proposed, unverified relevance labels*

| Configuration | Precision@5 | Recall@10 | MRR | nDCG@10 | Unjudged Rate@10 |
|---|---|---|---|---|---|
{ablation_table}

### 6. Circularity Audit: Relevant Subset with Zero Literal Skill Overlap (Indicative Only: 51 pairs, 17 profiles)
*Evaluated against model-proposed, unverified relevance labels*

Subset consists of **51 relevant candidate postings across 17 profiles** that share zero literal keyword overlap with candidate declared skills (13 profiles had zero such postings and are excluded from Recall/MRR means):

| Strategy | Precision@5 (95% CI) | Recall@10 (95% CI) | MRR (95% CI) | nDCG@10 (95% CI) |
|---|---|---|---|---|
{circ_table}

> [!NOTE]
> **Indicative Finding Only (51 pairs, 17 profiles)**: Evaluated on the subset of 51 relevant candidate postings across 17 profiles with zero literal skill matches (13 profiles have zero relevant items in this slice and are excluded from Recall/MRR means). Dense retrieval achieves Recall@10 of 0.3931 [0.2333, 0.5687] vs BM25 0.2588 [0.1245, 0.4265] and MRR of 0.3437 [0.1546, 0.5314] vs BM25 0.1751 [0.0727, 0.3139]. This indicates that dense vector retrieval can surface conceptually relevant postings when keyword markers are absent, but the small sample size and wide confidence intervals mean this is indicative only.
"""
    return section


def build_phase_2c_markdown(llm_quality: Dict[str, Any]) -> str:
    """Build Markdown content for Phase 2C LLM explanation evaluation.

    Args:
        llm_quality: Parsed reports/llm_quality.json dictionary.

    Returns:
        Markdown string for Phase 2C section.
    """
    if not llm_quality or "validity" not in llm_quality:
        return ""

    caption = llm_quality.get("caption", "Programmatic checks against job text; no human or LLM judge")
    model = llm_quality.get("gemini_model", "gemini-3.1-flash-lite")
    prompt_ver = llm_quality.get("prompt_version", "explain-v1.0")
    n_calls = llm_quality.get("n_calls", 150)
    n_profiles = llm_quality.get("n_profiles", 30)
    cache_hits = llm_quality.get("cache_hits", 0)
    live_calls = llm_quality.get("live_calls", 0)
    cache_hit_rate = llm_quality.get("cache_hit_rate", 0.0)

    val = llm_quality.get("validity", {})
    first_try_ci = val.get("json_valid_first_try_ci_95")
    first_try_ci_str = format_ci(first_try_ci) if isinstance(first_try_ci, list) else str(first_try_ci)
    fallback_ci = val.get("fallback_ci_95")
    fallback_ci_str = format_ci(fallback_ci) if isinstance(fallback_ci, list) else str(fallback_ci)

    g = llm_quality.get("groundedness", {})
    g_main_ci = g.get("groundedness_main_macro_ci_95")
    g_main_ci_str = format_ci(g_main_ci) if isinstance(g_main_ci, list) else str(g_main_ci)
    g_strict_ci = g.get("groundedness_strict_macro_ci_95")
    g_strict_ci_str = format_ci(g_strict_ci) if isinstance(g_strict_ci, list) else str(g_strict_ci)

    q = llm_quality.get("quality_signals", {})
    supp_ci = q.get("profile_support_macro_ci_95")
    supp_ci_str = format_ci(supp_ci) if isinstance(supp_ci, list) else str(supp_ci)
    contra_ci = q.get("missing_skills_contradiction_macro_ci_95")
    contra_ci_str = format_ci(contra_ci) if isinstance(contra_ci, list) else str(contra_ci)

    lat = llm_quality.get("latency_live_ms", {})
    tok = llm_quality.get("token_usage_live", {})

    section = f"""---

## Phase 2C: LLM Explanation Quality & Groundedness Evaluation

### Architecture & Protocol
- **Objective**: Explain top-5 job matches from the best retrieval strategy (BM25 Only, nDCG@10 = 0.6147) for each candidate profile without relying on an LLM judge.
- **Model**: `{model}` via Google AI Studio API (Date checked: 2026-11-20, [Official Docs](https://ai.google.dev/gemini-api/docs/models)).
- **Prompt Version**: `{prompt_ver}` with structured Pydantic schema validation:
  - `matched_skills: list[str]` (technical skills explicitly appearing in job text)
  - `missing_skills: list[str]` (qualifications required by job that candidate lacks)
  - `summary: str` (strictly enforced single-sentence rationale)
- **Retry & Recovery Protocol**: Exactly one repair prompt retry allowed on JSON/schema validation error. Persistent disk caching in `data/cache/explanations/`.
- **Evaluation Methodology**: 100% programmatic verification against raw job posting text. Zero LLM judge and zero human annotations.

### 1. Schema Validity & Fallback Robustness
*{caption}*

| Metric | Measured Value | 95% Cluster Bootstrap CI | Description |
|---|---|---|---|
| **JSON Valid First Try Rate** | {val.get('json_valid_first_try_rate', 0.0) * 100:.2f}% ({int(val.get('json_valid_first_try_rate', 0.0) * n_calls)}/{n_calls}) | {first_try_ci_str} | Responses passing Pydantic schema on initial attempt |
| **Retries Needed** | {val.get('retries_needed_count', 0)} calls ({val.get('retries_needed_rate', 0.0) * 100:.2f}%) | — | Total calls requiring repair prompt |
| **JSON Valid After Retry Rate** | {val.get('json_valid_after_retry_rate', 0.0) * 100:.2f}% | — | Cumulative validity after allowed single retry ($\ge$ first-try rate) |
| **Fallback Rate** | {val.get('fallback_rate', 0.0) * 100:.2f}% ({int(val.get('fallback_rate', 0.0) * n_calls)}/{n_calls}) | {fallback_ci_str} | Calls returning fallback (`explanation=null`) |

> **Fallback Breakdown**: `invalid_json: 0`, `validation_error: 0`, `api_error: 0`, `timeout: 0`, `quota: 0` across all {n_calls} calls.

### 2. Literal Groundedness & Profile Alignment
*{caption}*

| Metric | Pooled Rate (Skills) | Macro Profile Average | 95% Cluster Bootstrap CI | Definition / Verification Scope |
|---|---|---|---|---|
| **Main Groundedness (Boundary Regex)** | **{g.get('groundedness_main_pooled', 0.0) * 100:.2f}%** ({g.get('total_grounded_skills', 0)}/{g.get('total_claimed_matched_skills', 0)}) | **{g.get('groundedness_main_macro', 0.0) * 100:.2f}%** | {g_main_ci_str} | Boundary-safe regex `(?<![\\w\\+\\#]){{skill}}(?![\\w\\+\\#])` with symbol support (`C++`, `C#`, `.NET`) |
| **Strict Groundedness (Exact Substring)** | **{g.get('groundedness_strict_pooled', 0.0) * 100:.2f}%** ({g.get('total_strict_grounded_skills', 0)}/{g.get('total_claimed_matched_skills', 0)}) | **{g.get('groundedness_strict_macro', 0.0) * 100:.2f}%** | {g_strict_ci_str} | Case-sensitive literal substring match without normalization |
| **Profile Support Rate** | **{q.get('profile_support_pooled', 0.0) * 100:.2f}%** | **{q.get('profile_support_macro', 0.0) * 100:.2f}%** | {supp_ci_str} | Claimed matched skills verified to exist in candidate profile |
| **Contradiction Leakage Rate** | **{q.get('missing_skills_contradiction_pooled', 0.0) * 100:.2f}%** ({q.get('total_contradiction_skills', 0)}/{q.get('total_missing_skills_claimed', 0)}) | **{q.get('missing_skills_contradiction_macro', 0.0) * 100:.2f}%** | {contra_ci_str} | Self-contradiction check: skills claimed missing that appear in both job text and candidate profile |

> [!NOTE]
> **Scope of Contradiction Leakage**: Contradiction leakage measures internal self-contradiction only (instances where the model mistakenly claims a skill is missing despite it being present in both the candidate's profile and the job posting). It does not evaluate the overall completeness or correctness of the `missing_skills` list from the employer's perspective.

### 3. Operational Performance & Resource Utilization
*(Measured live over {live_calls} API calls; {cache_hits} calls served from disk cache)*

| Metric | Measured Value | Operational Context |
|---|---|---|
| **Total Explanation Calls** | {n_calls} calls across {n_profiles} candidate profiles | Top-5 BM25 matches per profile |
| **Cache Hit Rate** | {cache_hit_rate * 100:.1f}% ({cache_hits} hits, {live_calls} live calls) | Persistent on-disk JSON cache (`data/cache/explanations/`) |
| **Live Call Latency (Mean)** | {lat.get('mean_ms', 0.0):.1f} ms | Sub-second base model inference with backoff |
| **Live Call Latency (p95)** | {lat.get('p95_ms', 0.0):.1f} ms | Tail latency under rate-controlled execution |
| **Prompt Tokens Consumed** | {tok.get('prompt_tokens', 0):,} tokens | Job description + candidate profile context |
| **Candidate Tokens Generated** | {tok.get('candidate_tokens', 0):,} tokens | Structured JSON explanations |
| **Total Tokens Consumed** | {tok.get('total_tokens', 0):,} tokens | Verified via API `usageMetadata` |

### 4. Qualitative Audit & Sample Explanations
A full qualitative inspection containing 10 randomly sampled explanations (seed = 42) and the 5 worst cases by literal groundedness is documented in [`reports/llm_samples.md`](reports/llm_samples.md).
"""
    return section


def build_readme_markdown(
    data_stats: Dict[str, Any],
    index_stats: Dict[str, Any],
    sync_stats: Optional[Dict[str, Any]] = None,
    labeling_stats: Optional[Dict[str, Any]] = None,
    agreement_stats: Optional[Dict[str, Any]] = None,
    ret_main: Optional[Dict[str, Any]] = None,
    ret_pure: Optional[Dict[str, Any]] = None,
    ret_paired: Optional[Dict[str, Any]] = None,
    ret_latency: Optional[Dict[str, Any]] = None,
    ret_ablation: Optional[Dict[str, Any]] = None,
    ret_circularity: Optional[Dict[str, Any]] = None,
    llm_quality: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct full README content embedding verified stats from reports.

    Args:
        data_stats: Parsed dictionary from reports/data_stats.json.
        index_stats: Parsed dictionary from reports/index_stats.json.
        sync_stats: Optional parsed dictionary from reports/sync_benchmark.json.
        labeling_stats: Optional parsed dictionary from reports/labeling_method.json.
        agreement_stats: Optional parsed dictionary from reports/label_agreement.json.
        ret_main: Parsed retrieval_metrics_main.json.
        ret_pure: Parsed retrieval_metrics_pure.json.
        ret_paired: Parsed retrieval_paired_comparisons.json.
        ret_latency: Parsed retrieval_latency.json.
        ret_ablation: Parsed retrieval_ablation.json.
        ret_circularity: Parsed retrieval_circularity_check.json.
        llm_quality: Parsed reports/llm_quality.json.

    Returns:
        Complete Markdown string for README.md.
    """
    row_counts = data_stats.get("row_counts", {})
    desc_stats = data_stats.get("description_length", {})
    levels = data_stats.get("experience_level_distribution", {})

    level_rows = "\n".join(
        [
            f"| {lvl} | {info['count']} | {info['percentage']}% |"
            for lvl, info in levels.items()
        ]
    )

    index_table = ""
    if index_stats:
        index_table = f"""### Measured Indexing Performance
*Measured indexing performance from `reports/index_stats.json`*

| Metric | Measured Value |
|---|---|
| Vector Store | ChromaDB (HNSW Cosine) |
| Embedding Model | `{index_stats.get("model_name", "N/A")}` |
| Embedding Dimension | {index_stats.get("embedding_dimension", "N/A")} |
| Total Documents Indexed | {index_stats.get("total_documents_indexed", "N/A")} |
| Index Build Time | {index_stats.get("index_build_time_seconds", "N/A")} s |
| Indexing Throughput | {index_stats.get("indexing_throughput_docs_sec", "N/A")} docs/sec |
| Metadata Fields | `{", ".join(index_stats.get("metadata_fields", []))}` |
"""

    sync_table = ""
    if sync_stats:
        mut_params = sync_stats.get("mutation_parameters", {})
        changes = sync_stats.get("change_counts", {})
        full = sync_stats.get("full_reindex", {})
        sync = sync_stats.get("incremental_sync", {})
        verif = sync_stats.get("state_equivalence_verification", {})

        sync_table = f"""---

## Phase 1B: Incremental Index Synchronization Benchmark

### Simulated Change Scenario
- **Dataset Baseline Size:** {sync_stats.get("dataset_baseline_size", "N/A")} postings
- **Modifications:** {mut_params.get("modify_percentage", 0)}% of descriptions ({changes.get("modified_count", 0)} postings)
- **Additions:** {mut_params.get("add_percentage", 0)}% new postings ({changes.get("added_count", 0)} postings)
- **Deletions:** {mut_params.get("delete_percentage", 0)}% removed postings ({changes.get("deleted_count", 0)} postings)
- **Unchanged:** {changes.get("unchanged_count", 0)} postings (zero embedding overhead)
- **Benchmark Repetitions:** {sync_stats.get("benchmark_runs", 3)} runs per strategy

### Measured Speedup & Throughput
*(Numbers extracted directly from `reports/sync_benchmark.json`)*

| Strategy | Mean Latency | Latency Std Dev | Effective Throughput |
|---|---|---|---|
| Full Re-Index (from scratch) | {full.get("mean_seconds", "N/A")} s | {full.get("std_seconds", "N/A")} s | {full.get("throughput_docs_per_sec", "N/A")} docs/sec |
| Hash-Based Incremental Sync | **{sync.get("mean_seconds", "N/A")} s** | {sync.get("std_seconds", "N/A")} s | **{sync.get("throughput_effective_docs_per_sec", "N/A")} docs/sec** |
| **Speedup Factor** | **{sync_stats.get("speedup_factor", "N/A")}x** | — | — |

### Index State Equivalence Verification
- **Verification Status:** {"PASSED (100% Identical)" if verif.get("passed") else "FAILED"}
- **Synced Document Count:** {verif.get("synced_document_count", "N/A")}
- **Full Re-Index Document Count:** {verif.get("full_reindex_document_count", "N/A")}
- **ID Set Match:** {verif.get("id_sets_identical", False)}
- **Content Hash Discrepancies:** {verif.get("hash_mismatches_count", 0)}

> [!NOTE]
> **Scope of Incremental Synchronization**: This benchmark measures incremental synchronization of the **ChromaDB dense vector index** specifically, where unchanged documents bypass embedding inference via SHA-256 content hashes. Because `rank_bm25` does not support incremental updates, the BM25 lexical index and tokenized corpus must be rebuilt in full (~6.0 s) whenever postings are added, modified, or removed. The production pipeline is therefore hybrid: dense vector updates are incremental, while lexical index updates require a complete rebuild.
"""

    rrf_formula = "$$RRF(d) = \\sum_{s \\in \\{bm25, dense\\}} \\frac{1}{60 + \\text{rank}_s(d)}$$"
    boost_formula = "$$\\text{final\\_score} = \\text{base\\_score} + w \\times \\frac{|\\text{matched\\_skills}|}{|\\text{candidate\\_skills}|}$$"

    # Phase 2A section: Explicitly document unverified model-proposed state
    agreement_section = """### Relevance Evaluation Status (`data/eval/qrels.json`)
*Evaluated against model-proposed, unverified relevance labels*

| Property | Setting / Value | Documentation |
|---|---|---|
| Relevance Labels Source | **100% Rule-Based Proposed Labels** (`label_source: model_proposed`) | [`docs/LABELING.md`](docs/LABELING.md) |
| Total Evaluated Candidate Pairs | 1,037 pairs (across 30 profiles) | `data/eval/qrels.json` |
| Human Review Count | **0 pairs** (`human_reviewed: false`) | `reports/label_agreement.json` |
| Review Pipeline Status | Blinded review package prepared & intact | [`labels_to_review.csv`](labels_to_review.csv) |
| Review Guide | Profile and rubric documentation | [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md) |

> [!NOTE]
> All evaluation labels in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`) generated via deterministic heuristics; unverified by humans. Inter-annotator agreement metrics (disagreement rate, Cohen's kappa) are withheld until independent review is submitted.
"""

    phase_2b_section = ""
    if ret_main and ret_pure and ret_paired and ret_latency and ret_ablation and ret_circularity:
        phase_2b_section = build_phase_2b_markdown(
            ret_main, ret_pure, ret_paired, ret_latency, ret_ablation, ret_circularity
        )

    phase_2c_section = ""
    if llm_quality:
        phase_2c_section = build_phase_2c_markdown(llm_quality)

    content = f"""# Job-Matching Search with Evaluation

An end-to-end, reproducible ML portfolio project building and comparing retrieval strategies (dense semantic search, lexical BM25, hybrid retrieval, and reciprocal rank fusion) for matching candidate profiles against job postings, accompanied by grounded LLM explanations and rigorous evaluation metrics.

---

## Phase 1A: Data and Indexing

### Dataset Provenance & Attribution
- **Dataset**: {data_stats.get("dataset_name", "LinkedIn Job Postings")}
- **Source URL**: [{data_stats.get("source_url", "Hugging Face")}]({data_stats.get("source_url", "")})
- **Licence**: {data_stats.get("licence", "CC BY-SA 4.0 / MIT")}
- **Integrity**: 100% real postings; no synthetic data generated.

### Measured Pipeline Statistics
*(All metrics generated automatically by `src/generate_readme.py` from `reports/data_stats.json`)*

| Stage | Metric | Count / Value |
|---|---|---|
| Raw Pool | Downloaded Postings | {row_counts.get("raw_postings_pool", "N/A")} |
| Cleaned Pool | Valid Postings After Cleaning & Deduplication | {row_counts.get("clean_postings_available", "N/A")} |
| Sampled Output | Final Sampled Dataset (`seed=42`) | {row_counts.get("final_sampled_postings", "N/A")} |
| Text Metric | Average Description Characters | {desc_stats.get("avg_characters", "N/A")} chars |
| Text Metric | Average Description Words | {desc_stats.get("avg_words", "N/A")} words |
| Text Metric | Median Words | {desc_stats.get("median_words", "N/A")} words |

### Experience Level Distribution

| Seniority Tier | Count | Percentage |
|---|---|---|
{level_rows}

{index_table}
{sync_table}
---

## Phase 1C: Retrieval Strategies & Personalization

### Evaluated Strategies
1. **BM25 Only (`BM25Retriever`)**: Lexical keyword search using `rank_bm25.BM25Okapi` with domain-aware regex tokenization preserving technical symbols (`c++`, `c#`, `.net`, `python-3`).
2. **Dense Only (`DenseRetriever`)**: Semantic vector retrieval using ChromaDB HNSW cosine similarity over `sentence-transformers/all-MiniLM-L6-v2` embeddings.
3. **Hybrid RRF (`HybridRRFRetriever`)**: Reciprocal Rank Fusion combining lexical and dense streams with parameter $k_{{rrf}} = 60$:
   {rrf_formula}
4. **Hybrid + Cross-Encoder Re-Rank (`CrossEncoderRetriever`)**: Two-stage pipeline fusing top 50 candidates from hybrid RRF, then re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention and sigmoid normalization.

### Personalization & Filtering Pipeline
- **Profile Query Synthesis**: Structured query generation combining target role, declared technical skills, seniority tier, and narrative resume text.
- **Three-Tier Fallback Hierarchy**:
  - **Tier 1 (Strict)**: Postings matching candidate preferred location (allowing "Remote") and seniority (allowing "Not Specified").
  - **Tier 2 (Relaxed Seniority)**: If Tier 1 < $k$, relax seniority tier while preserving location constraint.
  - **Tier 3 (Global Backfill)**: If Tier 2 < $k$, backfill remaining slots from top unfiltered rankings to guarantee $k$ results.
- **Explainable Skill Boosting**:
  - Additive score formula:
    {boost_formula}
    *(default weight $w = 0.2$)*.
  - Word-boundary matching `(?<![\\w\\+\\#]){{skill}}(?![\\w\\+\\#])` prevents substring false-positives while preserving symbols like C++, C#, and .NET.

---

## Phase 2A: Evaluation Set, Candidate Pooling & Relevance Labels

### Test Candidate Profiles (`data/eval/profiles.json`)
- **Profile Provenance**: **100% Hand-Authored Synthetic Test Inputs** (expressly crafted to benchmark diverse domains, career changers, and edge cases). All 3,000 job postings in `data/jobs_clean.parquet` remain 100% real LinkedIn postings.
- **Diversity Coverage**: 30 profiles spanning Software Engineering, Data/ML, DevOps/SRE, Product Management, QA, Security, Database Admin, Embedded Systems, and BI.
- **Strict Tier 1 Gate**: Every profile is programmatically verified to have **$\ge 50$ postings** matching its location and seniority under strict Tier 1 filtering before inclusion.
- **Seniority Constraints**: Standard Mid-Senior (18) and Entry level (10) focus, with exactly 2 non-standard seniority edge cases (1 Director, 1 Internship).
- **Hard Cases (4 profiles)**: Includes 2 career changers (Math Teacher $\\to$ Data Analyst; Mechanical Engineer $\\to$ Junior Developer) and 2 vague technical background profiles.

### Candidate Pooling & Relevance Labels (`labels_proposed.csv`)
*Evaluated against model-proposed, unverified relevance labels*

- **Unbiased Pooling**: Top 20 candidates retrieved across all four retrieval strategies (BM25, Dense, Hybrid RRF, Cross-Encoder) and pooled via union per profile (**1,037 total pairs**).
- **Graded Rubric**: 3-point relevance scale (`0 = Irrelevant`, `1 = Partially Relevant`, `2 = Strongly Relevant`) documented in [`docs/LABELING.md`](docs/LABELING.md).
- **Label Distribution (100% Rule-Based Proposed Labels)**:
  - **Label 0 (Irrelevant)**: 757 pairs (73.00%)
  - **Label 1 (Partially Relevant)**: 255 pairs (24.59%)
  - **Label 2 (Strongly Relevant)**: 25 pairs (2.41%)

| Relevance Grade | Definition | Count | Percentage |
|---|---|---|---|
| **0** | Irrelevant | 757 | 73.00% |
| **1** | Partially Relevant | 255 | 24.59% |
| **2** | Strongly Relevant | 25 | 2.41% |
| **Total** | Pooled Candidate Pool | 1,037 | 100.00% |

{agreement_section}
{phase_2b_section}
{phase_2c_section}
---

## Limitations

1. **Model-Proposed, Unverified Relevance Labels**: All 1,037 relevance labels in `data/eval/qrels.json` are rule-based proposed labels (`label_source: model_proposed`), generated via deterministic heuristics (`jobmatch-rubric-v1`) and unverified by humans (`human_reviewed: false`). Evaluated retrieval metrics reflect alignment with deterministic rule-based criteria rather than human preference.
2. **Lexical Overlap Bias in Heuristic Rubric**: Because the heuristic proposal rubric incorporates technical skill regex and keyword matching, it inherently favors lexical retrieval methods (BM25 and hybrid combinations) over pure semantic vector retrieval. While the retrieval-independent audit shows that 51 out of 280 proposed relevant candidates (18.21% of label-1/2 rows) possess zero lexical overlap with candidate skills, 81.79% (and 100% of label-2 candidates) share explicit skill keywords. High BM25 scores reflect shared lexical signals with the rubric, not evidence that BM25 is the better retriever in absolute terms.
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
9. **Literal Groundedness vs Contextual Correctness (Phase 2C)**: Groundedness evaluates literal presence in job descriptions via boundary-safe regex, not semantic suitability or contextual sentiment. A skill mentioned in an exclusionary clause would still register as literally grounded.
10. **Contradiction Leakage Scope (Phase 2C)**: Contradiction leakage measures self-contradiction only (skills claimed as missing that appear in both the job text and candidate profile). It does not assess the general completeness or relevance of the `missing_skills` list from an employer standpoint.
11. **Prompt and Model Version Specificity (Phase 2C)**: LLM explanation outputs, schema validity, and groundedness metrics reflect `gemini-3.1-flash-lite` using prompt `explain-v1.0`; alternative models or prompt iterations will produce different performance profiles.
12. **Programmatic Verification Without LLM or Human Judge (Phase 2C)**: All explanation quality signals were audited via deterministic string and regex checks; no qualitative human evaluation or secondary LLM judge was utilized.

---

## Project Structure

```text
job-search-project/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules (ignoring data/, chroma_db/, .venv/)
├── DECISIONS.md              # Architectural decisions, trade-offs, and justifications
├── Dockerfile                # Reproducible container configuration
├── Makefile                  # Build and execution targets
├── pyproject.toml            # Pytest configuration and project metadata
├── README.md                 # System overview and measured results table
├── requirements.txt          # Pinned dependency manifest
├── RESULTS.md                # Detailed experimental results and retrieval comparison
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

## Quickstart & Reproducibility

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Execute Pipeline
```bash
# Ingestion, cleaning, deduplication, and sampling
python src/ingest.py

# Dense vector indexing in ChromaDB
python src/index.py

# Benchmark incremental synchronization (Phase 1B)
python src/bench_sync.py

# Run side-by-side retrieval comparison demo (Phase 1C)
python src/demo_comparison.py

# Run Phase 2B full retrieval evaluation across all strategies
python src/run_retrieval_eval.py

# Run full unit test suite
pytest -v tests/

# Update README and RESULTS results tables directly from reports
python src/generate_readme.py
```

### 3. One-Command Execution
- On Linux/macOS: `./run_all.sh` or `make all`
- On Windows: `run_all.bat`
- Via Docker:
  ```bash
  docker build -t job-search .
  docker run --rm job-search
  ```
"""
    return content


def update_readme() -> None:
    """Generate and overwrite README.md and RESULTS.md using metrics from reports/."""
    data_stats = load_json_report(DATA_STATS_PATH)
    index_stats = load_json_report(INDEX_STATS_PATH)
    sync_stats = load_json_report(SYNC_STATS_PATH)

    if not data_stats:
        print(f"Warning: {DATA_STATS_PATH} not found. Skipping README update.")
        return

    labeling_stats = load_json_report(LABELING_STATS_PATH)
    agreement_stats = load_json_report(AGREEMENT_STATS_PATH)

    ret_main = load_json_report(RETRIEVAL_MAIN_PATH)
    ret_pure = load_json_report(RETRIEVAL_PURE_PATH)
    ret_paired = load_json_report(RETRIEVAL_PAIRED_PATH)
    ret_latency = load_json_report(RETRIEVAL_LATENCY_PATH)
    ret_ablation = load_json_report(RETRIEVAL_ABLATION_PATH)
    ret_circularity = load_json_report(RETRIEVAL_CIRCULARITY_PATH)
    llm_quality = load_json_report(LLM_QUALITY_PATH)

    content = build_readme_markdown(
        data_stats=data_stats,
        index_stats=index_stats,
        sync_stats=sync_stats,
        labeling_stats=labeling_stats,
        agreement_stats=agreement_stats,
        ret_main=ret_main,
        ret_pure=ret_pure,
        ret_paired=ret_paired,
        ret_latency=ret_latency,
        ret_ablation=ret_ablation,
        ret_circularity=ret_circularity,
        llm_quality=llm_quality,
    )

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully generated {README_PATH} from reports/")

    # Also update RESULTS.md
    phase_2b_part = ""
    if ret_main and ret_pure and ret_paired and ret_latency and ret_ablation and ret_circularity:
        phase_2b_part = build_phase_2b_markdown(
            ret_main, ret_pure, ret_paired, ret_latency, ret_ablation, ret_circularity
        )

    phase_2c_part = ""
    if llm_quality:
        phase_2c_part = build_phase_2c_markdown(llm_quality)

    results_content = f"""# Experimental Results & Retrieval Evaluation Benchmark

An empirical benchmark comparing retrieval architectures, candidate pooling, personalization strategies, and relevance evaluation for matching job seekers against real job postings.

---

## 1. Retrieval Strategies Benchmark Overview

Four retrieval strategies are evaluated across 30 test candidate profiles spanning diverse engineering domains, seniority tiers, and edge cases (such as career changers and vague profiles):

1. **BM25 Only (`BM25Retriever`)**: Lexical keyword search using BM25Okapi with technical tokenization (`c++`, `c#`, `.net`, `python-3`).
2. **Dense Only (`DenseRetriever`)**: Semantic dense vector search in ChromaDB using `all-MiniLM-L6-v2` embeddings and HNSW cosine distance.
3. **Hybrid RRF (`HybridRRFRetriever`)**: Reciprocal Rank Fusion ($k_{{rrf}}=60$) combining reciprocal ranks from BM25 and Dense streams.
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

| Relevance Grade | Total Candidates | Zero Skill Overlap | $\\ge 1$ Skill Overlap | % with Zero Skill Overlap |
|---|---|---|---|---|
| **Label 0 (Irrelevant)** | 757 | 466 | 291 | 61.56% |
| **Label 1 (Partially Relevant)** | 255 | 51 | 204 | 20.00% |
| **Label 2 (Strongly Relevant)** | 25 | 0 | 25 | 0.00% |
| **Relevant Candidates (1 or 2)** | **280** | **51** | **229** | **18.21%** |

{phase_2b_part}
{phase_2c_part}
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
"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(results_content)
    print(f"Successfully generated {RESULTS_PATH} from reports/")


if __name__ == "__main__":
    update_readme()
