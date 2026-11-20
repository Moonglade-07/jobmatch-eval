"""Execution runner for LLM Explanation Generation & Groundedness Evaluation (Phase 2C).

Evaluates top-5 retrieved postings for candidate profiles using the best retrieval strategy.
Supports --smoke-test (5 profiles, 25 calls) and full evaluation (30 profiles, 150 calls).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL,
    MANDATORY_CAPTION,
    PROMPT_VERSION,
    ExplainResult,
    Explanation,
    GeminiExplainer,
    evaluate_explanations,
)
from src.personalise import Profile
from src.retrieval import BM25Retriever, SearchResult

PROFILES_PATH = Path("data/eval/profiles.json")
JOBS_PATH = Path("data/jobs_clean.parquet")
MAIN_RETRIEVAL_METRICS_PATH = Path("reports/retrieval_metrics_main.json")
OUTPUTS_JSONL_PATH = Path("reports/llm_outputs.jsonl")
QUALITY_JSON_PATH = Path("reports/llm_quality.json")
SAMPLES_MD_PATH = Path("reports/llm_samples.md")


def load_eval_data() -> Tuple[List[Profile], Dict[str, Any], pd.DataFrame]:
    """Load candidate profiles, retrieval metrics, and clean job postings DataFrame."""
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Candidate profiles not found at {PROFILES_PATH}")
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles_raw = json.load(f)["profiles"]
    profiles = [Profile(**p) for p in profiles_raw]

    if not MAIN_RETRIEVAL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Retrieval metrics not found at {MAIN_RETRIEVAL_METRICS_PATH}")
    with open(MAIN_RETRIEVAL_METRICS_PATH, "r", encoding="utf-8") as f:
        main_metrics = json.load(f)

    if not JOBS_PATH.exists():
        raise FileNotFoundError(f"Job postings not found at {JOBS_PATH}")
    jobs_df = pd.read_parquet(JOBS_PATH)
    jobs_df["id"] = jobs_df["id"].astype(str)

    return profiles, main_metrics, jobs_df


def get_best_strategy_name(main_metrics: Dict[str, Any]) -> str:
    """Identify best retrieval strategy by mean nDCG@10."""
    strats = main_metrics.get("strategies", {})
    if not strats:
        return "bm25"
    best_name = max(strats.keys(), key=lambda k: strats[k].get("ndcg@10", 0.0))
    return best_name


def recompute_from_outputs() -> Dict[str, Any]:
    """Recompute evaluation metrics from reports/llm_outputs.jsonl with ZERO new API calls."""
    if not OUTPUTS_JSONL_PATH.exists():
        raise FileNotFoundError(f"{OUTPUTS_JSONL_PATH} does not exist.")

    profiles, main_metrics, jobs_df = load_eval_data()
    jobs_dict = {str(row["id"]): row.to_dict() for _, row in jobs_df.iterrows()}
    profiles_dict = {p.profile_id: p.__dict__ for p in profiles}

    results: List[ExplainResult] = []
    with open(OUTPUTS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            exp_data = item.get("explanation")
            explanation = Explanation(**exp_data) if exp_data else None
            res = ExplainResult(
                profile_id=item["profile_id"],
                job_id=item["job_id"],
                explanation=explanation,
                raw_response=None,
                cache_hit=item.get("cache_hit", False),
                retry_count=item.get("retry_count", 0),
                fallback=item.get("fallback", False),
                fallback_reason=item.get("fallback_reason"),
                latency_ms=item.get("latency_ms", 0.0),
                prompt_tokens=item.get("prompt_tokens"),
                candidate_tokens=item.get("candidate_tokens"),
                total_tokens=item.get("total_tokens"),
                model=item.get("model", DEFAULT_MODEL),
                prompt_version=item.get("prompt_version", PROMPT_VERSION),
            )
            results.append(res)

    print(f"Loaded {len(results)} explanation results from {OUTPUTS_JSONL_PATH}")
    metrics = evaluate_explanations(results, profiles_dict, jobs_dict)
    with open(QUALITY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved updated explanation quality metrics to {QUALITY_JSON_PATH}")

    print_metrics_summary(metrics)
    return metrics


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print structured metrics summary according to Phase 2C reporting requirements."""
    print("\n" + "=" * 80)
    print(f"LLM EXPLANATION QUALITY METRICS ({metrics.get('caption')})")
    print("=" * 80)
    print(f"Model: {metrics.get('gemini_model')} | Prompt Version: {metrics.get('prompt_version')}")
    print(f"Total Calls: {metrics.get('n_calls')} | Profiles Evaluated: {metrics.get('n_profiles')}")
    print(f"Cache Hits: {metrics.get('cache_hits')} | Live Calls: {metrics.get('live_calls')} ({metrics.get('cache_hit_rate', 0.0) * 100:.1f}%)")
    print("-" * 80)

    val = metrics.get("validity", {})
    print("SCHEMA VALIDITY:")
    print(f"  JSON Valid First Try Rate:    {val.get('json_valid_first_try_rate') * 100:.2f}% (95% CI: {val.get('json_valid_first_try_ci_95')})")
    print(f"  Retries Needed:               {val.get('retries_needed_count')} calls ({val.get('retries_needed_rate') * 100:.2f}%)")
    print(f"  JSON Valid After Retry Rate:  {val.get('json_valid_after_retry_rate') * 100:.2f}% (cumulative valid)")
    print(f"  Fallback Rate:                {val.get('fallback_rate') * 100:.2f}% (95% CI: {val.get('fallback_ci_95')})")
    print(f"  Fallback Breakdown:           {val.get('fallback_breakdown')}")

    g = metrics.get("groundedness", {})
    tot_claimed = g.get("total_claimed_matched_skills", 0)
    tot_grounded = g.get("total_grounded_skills", 0)
    tot_strict = g.get("total_strict_grounded_skills", 0)
    print("\nGROUNDEDNESS AGAINST JOB TEXT (No LLM Judge):")
    print(f"  Main Regex (Pooled):          {g.get('groundedness_main_pooled') * 100:.2f}% ({tot_grounded}/{tot_claimed} skills)")
    print(f"  Main Regex (Macro Profile):   {g.get('groundedness_main_macro') * 100:.2f}% (95% CI: {g.get('groundedness_main_macro_ci_95')})")
    print(f"  Strict Exact (Pooled):        {g.get('groundedness_strict_pooled') * 100:.2f}% ({tot_strict}/{tot_claimed} skills)")
    print(f"  Strict Exact (Macro Profile): {g.get('groundedness_strict_macro') * 100:.2f}% (95% CI: {g.get('groundedness_strict_macro_ci_95')})")

    q = metrics.get("quality_signals", {})
    tot_missing = q.get("total_missing_skills_claimed", 0)
    tot_contra = q.get("total_contradiction_skills", 0)
    print("\nQUALITY SIGNALS:")
    print(f"  Profile Support (Pooled):     {q.get('profile_support_pooled') * 100:.2f}%")
    print(f"  Profile Support (Macro):      {q.get('profile_support_macro') * 100:.2f}% (95% CI: {q.get('profile_support_macro_ci_95')})")
    print(f"  Contradiction Leakage (Pooled): {q.get('missing_skills_contradiction_pooled') * 100:.2f}% ({tot_contra}/{tot_missing} skills)")
    print(f"  Contradiction Leakage (Macro):  {q.get('missing_skills_contradiction_macro') * 100:.2f}% (95% CI: {q.get('missing_skills_contradiction_macro_ci_95')})")
    print(f"  Note on Contradiction Leakage: {q.get('contradiction_definition')}")

    lat = metrics.get("latency_live_ms", {})
    if lat.get("measured_live_calls", 0) > 0:
        print(f"\nOPERATIONAL LATENCY & TOKENS:")
        print(f"  Live Call Mean Latency:       {lat.get('mean_ms')} ms (p95: {lat.get('p95_ms')} ms over {lat.get('measured_live_calls')} calls)")
    tok = metrics.get("token_usage_live", {})
    print(f"  Total Live Token Usage:       {tok.get('total_tokens')} tokens (Prompt: {tok.get('prompt_tokens')}, Candidates: {tok.get('candidate_tokens')})")
    print("=" * 80 + "\n")


def run_explain_pipeline(
    smoke_test: bool = True,
    k_top: int = 5,
    request_delay: float = 1.0,
) -> Dict[str, Any]:
    """Run explanation pipeline over candidate profiles and top-k retrieved jobs."""
    profiles, main_metrics, jobs_df = load_eval_data()
    best_strat = get_best_strategy_name(main_metrics)
    print(f"\nPhase 2C LLM Explanation Pipeline")
    print(f"Top Strategy Selected: {best_strat} (based on highest nDCG@10 in reports/retrieval_metrics_main.json)")

    # Initialize retriever
    print("Initializing BM25 retriever (Filters ON, Skill Boost ON w=0.2)...")
    retriever = BM25Retriever()

    # Determine profile slice
    eval_profiles = profiles[:5] if smoke_test else profiles
    mode_str = "SMOKE TEST (5 profiles x top-5 = 25 calls)" if smoke_test else "FULL RUN (30 profiles x top-5 = 150 calls)"
    print(f"\nExecuting {mode_str}...")
    sys.stdout.flush()

    # Pre-map jobs dictionary for fast text access
    jobs_dict: Dict[str, Any] = {
        str(row["id"]): row.to_dict() for _, row in jobs_df.iterrows()
    }
    profiles_dict: Dict[str, Any] = {p.profile_id: p.__dict__ for p in profiles}

    # Initialize Explainer
    explainer = GeminiExplainer(
        request_delay=request_delay,
        cache_dir=DEFAULT_CACHE_DIR,
        prompt_version=PROMPT_VERSION,
    )
    print(f"Gemini Model: {explainer.model_name} | Prompt Version: {explainer.prompt_version}")
    print(f"Persistent Cache Directory: {DEFAULT_CACHE_DIR} (gitignored)")

    if not explainer.api_key:
        if OUTPUTS_JSONL_PATH.exists():
            print(
                f"\n[INFO] GEMINI_API_KEY is not configured in .env or environment.\n"
                f"Recomputing evaluation metrics directly from committed {OUTPUTS_JSONL_PATH} with 0 new API calls."
            )
            return recompute_from_outputs()

    results: List[ExplainResult] = []
    OUTPUTS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    jsonl_mode = "w" if smoke_test else "w"
    jsonl_file = open(OUTPUTS_JSONL_PATH, jsonl_mode, encoding="utf-8")

    total_pairs = len(eval_profiles) * k_top
    processed_count = 0

    try:
        for p_idx, profile in enumerate(eval_profiles, 1):
            top_results = retriever.search(
                profile, k=k_top, apply_filters=True, apply_boost=True, boost_weight=0.2
            )
            print(f"\n[{p_idx}/{len(eval_profiles)}] Profile {profile.profile_id} ({profile.target_role}, {profile.preferred_location})")

            for rank, search_res in enumerate(top_results, 1):
                processed_count += 1
                job_data = jobs_dict.get(search_res.job_id)
                if not job_data:
                    job_data = {
                        "id": search_res.job_id,
                        "title": search_res.title,
                        "company": search_res.company,
                        "location": search_res.location,
                        "experience_level": search_res.experience_level,
                        "description": search_res.description,
                    }

                print(f"  Call {processed_count}/{total_pairs}: Rank {rank} Job {search_res.job_id} ({search_res.title[:30]})...", end="", flush=True)

                explain_res = explainer.explain(
                    profile_data=profile.__dict__,
                    job_data=job_data,
                    use_cache=True,
                )
                results.append(explain_res)

                # Log to jsonl
                log_entry = {
                    "profile_id": explain_res.profile_id,
                    "job_id": explain_res.job_id,
                    "model": explain_res.model,
                    "prompt_version": explain_res.prompt_version,
                    "cache_hit": explain_res.cache_hit,
                    "retry_count": explain_res.retry_count,
                    "fallback": explain_res.fallback,
                    "fallback_reason": explain_res.fallback_reason,
                    "latency_ms": round(explain_res.latency_ms, 2),
                    "prompt_tokens": explain_res.prompt_tokens,
                    "candidate_tokens": explain_res.candidate_tokens,
                    "total_tokens": explain_res.total_tokens,
                    "explanation": explain_res.explanation.model_dump() if explain_res.explanation else None,
                }
                jsonl_file.write(json.dumps(log_entry) + "\n")
                jsonl_file.flush()

                status_tag = "CACHED" if explain_res.cache_hit else f"LIVE ({explain_res.latency_ms:.0f}ms)"
                if explain_res.fallback:
                    status_tag += f" [FALLBACK: {explain_res.fallback_reason}]"
                elif explain_res.retry_count > 0:
                    status_tag += " [REPAIRED ON RETRY]"
                print(f" -> {status_tag}")

                if explain_res.fallback and explain_res.fallback_reason == "quota":
                    print("\n[CRITICAL] Daily quota exceeded. Halting execution cleanly per protocol.")
                    break
            if results and results[-1].fallback and results[-1].fallback_reason == "quota":
                break

    finally:
        jsonl_file.close()

    print(f"\nCompleted {len(results)}/{total_pairs} explanation calls.")

    # Compute evaluation metrics
    metrics = evaluate_explanations(results, profiles_dict, jobs_dict)
    with open(QUALITY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved explanation quality metrics to {QUALITY_JSON_PATH}")

    print_metrics_summary(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM explanation quality evaluation.")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="Run on 5 profiles (25 calls) first.")
    parser.add_argument("--full", action="store_true", default=False, help="Run on all 30 profiles (150 calls).")
    parser.add_argument("--recompute", action="store_true", default=False, help="Recompute metrics from reports/llm_outputs.jsonl without API calls.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds.")
    args = parser.parse_args()

    if args.recompute:
        recompute_from_outputs()
    else:
        smoke = True
        if args.full:
            smoke = False
        elif args.smoke_test:
            smoke = True

        run_explain_pipeline(smoke_test=smoke, request_delay=args.delay)
