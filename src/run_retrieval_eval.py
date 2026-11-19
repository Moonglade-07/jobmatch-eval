"""Comprehensive retrieval evaluation runner across all 4 strategies.

Evaluates BM25, Dense, Hybrid RRF, and Cross-Encoder Re-Ranking over 30 candidate profiles
against rule-based proposed labels (jobmatch-rubric-v1).

MANDATORY CAPTION & PROTOCOL:
  All outputs, tables, and JSON files carry the caption:
  "Evaluated against model-proposed, unverified relevance labels"
  Zero use of forbidden phrases ('human-verified', 'ground truth').
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enforce offline mode
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch

from src.metrics import (
    compute_macro_averages,
    evaluate_query,
    get_relevance_label,
)
from src.models import Profile, SearchResult
from src.personalise import extract_matched_skills
from src.retrieval import (
    BM25Retriever,
    CrossEncoderRetriever,
    DenseRetriever,
    HybridRRFRetriever,
)

PROFILES_PATH = Path("data/eval/profiles.json")
QRELS_PATH = Path("data/eval/qrels.json")
JOBS_PATH = Path("data/jobs_clean.parquet")
REPORTS_DIR = Path("reports")

MANDATORY_CAPTION = "Evaluated against model-proposed, unverified relevance labels"
LABEL_ORIGIN_NOTE = (
    "Relevance labels are rule-based proposed labels (jobmatch-rubric-v1) "
    "generated via deterministic heuristics; unverified by humans."
)


def load_candidate_profiles(profiles_path: Path = PROFILES_PATH) -> List[Profile]:
    """Load the 30 candidate profiles from JSON."""
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles file not found at {profiles_path}")
    with open(profiles_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profiles_list = data["profiles"] if isinstance(data, dict) and "profiles" in data else data
    return [Profile(**p) for p in profiles_list]


def load_qrels(qrels_path: Path = QRELS_PATH) -> Dict[str, Dict[str, Any]]:
    """Load qrels judgments dictionary from JSON."""
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found at {qrels_path}")
    with open(qrels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["qrels"] if "qrels" in data else data


def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval for the mean of a sequence.

    Args:
        values: List of metric values across profiles.
        n_resamples: Number of bootstrap iterations.
        confidence_level: Confidence level (0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    boot_means = np.empty(n_resamples, dtype=float)
    n = len(arr)

    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    return (round(ci_lower, 4), round(ci_upper, 4))


def evaluate_strategy_over_profiles(
    retriever: Any,
    profiles: List[Profile],
    qrels: Dict[str, Dict[str, Any]],
    apply_filters: bool = True,
    apply_boost: bool = True,
    boost_weight: float = 0.2,
    k_cutoff: int = 10,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Execute search across all profiles for a given retriever strategy.

    Args:
        retriever: Retriever instance implementing search().
        profiles: List of Profile objects.
        qrels: Qrels mapping profile_id to job judgments.
        apply_filters: Whether to apply location/seniority filters with fallback.
        apply_boost: Whether to apply explainable skill boost.
        boost_weight: Skill boost weight parameter.
        k_cutoff: Max documents retrieved.

    Returns:
        Tuple of (per_profile_metrics dict, per_profile_ranked_ids dict).
    """
    per_profile_metrics: Dict[str, Dict[str, Any]] = {}
    ranked_ids_per_profile: Dict[str, List[str]] = {}

    for profile in profiles:
        prof_id = profile.profile_id
        qrels_for_prof = qrels.get(prof_id, {})

        results: List[SearchResult] = retriever.search(
            profile=profile,
            k=k_cutoff,
            apply_filters=apply_filters,
            apply_boost=apply_boost,
            boost_weight=boost_weight,
        )
        retrieved_ids = [r.job_id for r in results]
        ranked_ids_per_profile[prof_id] = retrieved_ids

        m = evaluate_query(
            retrieved_ids=retrieved_ids,
            qrels_for_profile=qrels_for_prof,
            k_p=5,
            k_recall=10,
            k_ndcg=10,
            relevance_threshold=1,
        )
        per_profile_metrics[prof_id] = m

    return per_profile_metrics, ranked_ids_per_profile


def run_paired_bootstrap(
    metrics_a: Dict[str, Dict[str, Any]],
    metrics_b: Dict[str, Dict[str, Any]],
    metric_key: str,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired bootstrap difference and 95% CI between two strategies.

    Args:
        metrics_a: Per-profile metrics for strategy A.
        metrics_b: Per-profile metrics for strategy B.
        metric_key: Metric name ('precision@5', 'recall@10', 'mrr', 'ndcg@10').
        n_resamples: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with mean_difference, ci_lower, ci_upper, and significance flag.
    """
    diffs: List[float] = []
    for prof_id in metrics_a.keys():
        val_a = metrics_a[prof_id][metric_key]
        val_b = metrics_b[prof_id][metric_key]
        if val_a is not None and val_b is not None:
            diffs.append(float(val_a) - float(val_b))

    if not diffs:
        return {"mean_difference": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False}

    mean_diff = round(float(np.mean(diffs)), 4)
    ci_lower, ci_upper = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)
    # Significant if 0 is outside [ci_lower, ci_upper]
    is_significant = (ci_lower > 0.0) or (ci_upper < 0.0)

    return {
        "mean_difference": mean_diff,
        "ci_95": [ci_lower, ci_upper],
        "significant": bool(is_significant),
        "interpretation": "Clear difference against rule-based labels sharing lexical signals; not evidence of better retrieval" if is_significant else "No clear difference (CI spans 0.0)",
    }


def benchmark_latencies(
    retrievers: Dict[str, Any],
    profiles: List[Profile],
    runs_per_query: int = 20,
    startup_times: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Measure query latency (p50 and p95) under CPU execution across multiple runs.

    Indices, tokenized corpora, and models are built once at startup outside the
    timed region. Explicitly separates retrieval latency from cross-encoder re-ranking latency.

    Args:
        retrievers: Dictionary of named retriever instances.
        profiles: List of candidate profiles.
        runs_per_query: Number of timed repetitions per query.
        startup_times: Optional dictionary of one-off index build / startup times in ms.

    Returns:
        Dictionary of measured p50 and p95 latencies and hardware metadata.
    """
    print("\nExecuting query latency benchmarks (CPU execution, warm-up enabled)...")
    hw_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_cores": os.cpu_count(),
        "pytorch_device": "cpu",
        "pytorch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "runs_per_query": runs_per_query,
    }

    latency_results: Dict[str, Any] = {
        "hardware": hw_info,
        "caption": MANDATORY_CAPTION,
        "startup_times_ms": startup_times or {},
    }

    # 1. Warm-up all retrievers
    sample_prof = profiles[0]
    for name, ret in retrievers.items():
        ret.search(sample_prof, k=10)

    # 2. Benchmark BM25, Dense, and Hybrid RRF across runs_per_query timed runs
    for name in ["bm25", "dense", "hybrid_rrf"]:
        ret = retrievers[name]
        durations_ms: List[float] = []

        for run_idx in range(runs_per_query):
            prof = profiles[run_idx % len(profiles)]
            t0 = time.perf_counter()
            ret.search(prof, k=10)
            durations_ms.append((time.perf_counter() - t0) * 1000.0)

        latency_results[name] = {
            "p50_ms": round(float(np.percentile(durations_ms, 50)), 2),
            "p95_ms": round(float(np.percentile(durations_ms, 95)), 2),
            "mean_ms": round(float(np.mean(durations_ms)), 2),
            "std_ms": round(float(np.std(durations_ms)), 2),
            "total_measurements": len(durations_ms),
        }

    # 3. Benchmark Cross-Encoder: Separating Hybrid Retrieval step from Cross-Encoder re-rank step
    ce_retriever: CrossEncoderRetriever = retrievers["cross_encoder"]
    hybrid_ret = ce_retriever.hybrid_retriever
    cross_encoder_model = ce_retriever.cross_encoder

    # Retrieval step durations
    retrieval_step_durations_ms: List[float] = []
    # Cross-encoder inference durations
    ce_inference_durations_ms: List[float] = []
    # End-to-end durations
    ce_e2e_durations_ms: List[float] = []

    # Benchmark 20 timed runs of the cross-encoder forward pass over top-50 candidate pairs
    for run_idx in range(runs_per_query):
        # Pick query
        prof = profiles[run_idx % len(profiles)]

        # Time step 1: Hybrid retrieval
        t0 = time.perf_counter()
        candidates = hybrid_ret.search(prof, k=50, apply_filters=True, apply_boost=False)
        t_ret = (time.perf_counter() - t0) * 1000.0
        retrieval_step_durations_ms.append(t_ret)

        # Build cross-encoder pairs
        query_text = f"{prof.target_role} {' '.join(prof.skills)} {prof.seniority} {prof.resume_text}"
        pairs = [
            (query_text, f"Title: {c.title} Company: {c.company} Desc: {c.description[:1000]}")
            for c in candidates
        ]

        # Time step 2: Cross-Encoder model scoring
        t0_ce = time.perf_counter()
        with torch.inference_mode():
            _ = cross_encoder_model.predict(pairs, batch_size=32, show_progress_bar=False)
        t_ce = (time.perf_counter() - t0_ce) * 1000.0
        ce_inference_durations_ms.append(t_ce)

        ce_e2e_durations_ms.append(t_ret + t_ce)

    latency_results["cross_encoder"] = {
        "retrieval_step": {
            "description": "Hybrid RRF retrieval to produce top 50 candidates",
            "p50_ms": round(float(np.percentile(retrieval_step_durations_ms, 50)), 2),
            "p95_ms": round(float(np.percentile(retrieval_step_durations_ms, 95)), 2),
            "mean_ms": round(float(np.mean(retrieval_step_durations_ms)), 2),
        },
        "cross_encoder_scoring_step": {
            "description": "Cross-Encoder Transformer forward pass re-ranking 50 pairs on CPU",
            "p50_ms": round(float(np.percentile(ce_inference_durations_ms, 50)), 2),
            "p95_ms": round(float(np.percentile(ce_inference_durations_ms, 95)), 2),
            "mean_ms": round(float(np.mean(ce_inference_durations_ms)), 2),
        },
        "end_to_end": {
            "description": "Total pipeline latency (Retrieval + Scoring)",
            "p50_ms": round(float(np.percentile(ce_e2e_durations_ms, 50)), 2),
            "p95_ms": round(float(np.percentile(ce_e2e_durations_ms, 95)), 2),
            "mean_ms": round(float(np.mean(ce_e2e_durations_ms)), 2),
            "std_ms": round(float(np.std(ce_e2e_durations_ms)), 2),
            "total_measurements": len(ce_e2e_durations_ms),
        },
    }

    return latency_results


def run_retrieval_ablation(
    retriever: Any,
    strategy_name: str,
    profiles: List[Profile],
    qrels: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute ablation grid over boost, filters, and boost weight w."""
    print(f"\nRunning ablation study on top strategy '{strategy_name}'...")
    ablation_grid = [
        {"name": "Filters ON, Boost ON (w=0.2) [Main]", "filters": True, "boost": True, "w": 0.2},
        {"name": "Filters ON, Boost OFF (w=0.0)", "filters": True, "boost": False, "w": 0.0},
        {"name": "Filters OFF, Boost ON (w=0.2)", "filters": False, "boost": True, "w": 0.2},
        {"name": "Filters OFF, Boost OFF (w=0.0) [Pure]", "filters": False, "boost": False, "w": 0.0},
        {"name": "Filters ON, Boost w=0.1", "filters": True, "boost": True, "w": 0.1},
        {"name": "Filters ON, Boost w=0.4", "filters": True, "boost": True, "w": 0.4},
    ]

    results: List[Dict[str, Any]] = []
    for cfg in ablation_grid:
        per_prof_m, _ = evaluate_strategy_over_profiles(
            retriever=retriever,
            profiles=profiles,
            qrels=qrels,
            apply_filters=cfg["filters"],
            apply_boost=cfg["boost"],
            boost_weight=cfg["w"],
            k_cutoff=10,
        )
        macro = compute_macro_averages(per_prof_m)
        results.append({
            "configuration": cfg["name"],
            "apply_filters": cfg["filters"],
            "apply_boost": cfg["boost"],
            "boost_weight": cfg["w"],
            "precision@5": macro["mean_precision@5"],
            "recall@10": macro["mean_recall@10"],
            "mrr": macro["mean_mrr"],
            "ndcg@10": macro["mean_ndcg@10"],
            "unjudged_rate@10": macro["mean_unjudged_rate@10"],
        })

    return {
        "caption": MANDATORY_CAPTION,
        "note": LABEL_ORIGIN_NOTE,
        "strategy": strategy_name,
        "ablation_results": results,
    }


def run_circularity_audit(
    ranked_ids_by_strategy: Dict[str, Dict[str, List[str]]],
    profiles: List[Profile],
    qrels: Dict[str, Dict[str, Any]],
    jobs_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Execute circularity check restricted to relevant items with ZERO skill overlap."""
    print("\nRunning circularity audit (evaluating on subset with zero literal skill matches)...")

    # 1. Identify all (profile_id, job_id) relevant pairs with ZERO skill overlap
    zero_overlap_qrels: Dict[str, Dict[str, Any]] = {}
    total_zero_overlap_pairs = 0
    profiles_with_zero_overlap: List[str] = []

    prof_map = {p.profile_id: p for p in profiles}

    for prof_id, jobs in qrels.items():
        prof = prof_map.get(prof_id)
        if not prof:
            continue
        zero_overlap_qrels[prof_id] = {}

        for job_id, d in jobs.items():
            rel = d["relevance"] if isinstance(d, dict) else int(d)
            if rel >= 1:
                # Check lexical skill match in posting
                if job_id in jobs_df.index:
                    j_row = jobs_df.loc[job_id]
                    txt = f"{j_row['title']} {j_row['description']}"
                    matched = extract_matched_skills(prof.skills, txt)
                    if len(matched) == 0:
                        zero_overlap_qrels[prof_id][job_id] = d
                        total_zero_overlap_pairs += 1
            else:
                # Irrelevant rows can remain irrelevant
                zero_overlap_qrels[prof_id][job_id] = d

        if any(
            (d["relevance"] if isinstance(d, dict) else int(d)) >= 1
            for d in zero_overlap_qrels[prof_id].values()
        ):
            profiles_with_zero_overlap.append(prof_id)

    # 2. Evaluate each strategy on this restricted qrels set using main setting retrieved rankings
    strat_results: Dict[str, Any] = {}
    for name in ["bm25", "dense", "hybrid_rrf", "cross_encoder"]:
        per_prof_m: Dict[str, Dict[str, Any]] = {}
        for prof_id, retrieved_ids in ranked_ids_by_strategy[name].items():
            per_prof_m[prof_id] = evaluate_query(
                retrieved_ids=retrieved_ids,
                qrels_for_profile=zero_overlap_qrels.get(prof_id, {}),
                k_p=5,
                k_recall=10,
                k_ndcg=10,
                relevance_threshold=1,
            )
        macro = compute_macro_averages(per_prof_m)

        # Compute bootstrap CIs over the valid profiles
        p5_vals = [float(v["precision@5"]) for v in per_prof_m.values()]
        ndcg_vals = [float(v["ndcg@10"]) for v in per_prof_m.values()]
        rec_vals = [float(v["recall@10"]) for v in per_prof_m.values() if v["recall@10"] is not None]
        mrr_vals = [float(v["mrr"]) for v in per_prof_m.values() if v["mrr"] is not None]

        strat_results[name] = {
            "precision@5": macro["mean_precision@5"],
            "precision@5_ci_95": bootstrap_ci(p5_vals, seed=42) if p5_vals else [0.0, 0.0],
            "recall@10": macro["mean_recall@10"],
            "recall@10_ci_95": bootstrap_ci(rec_vals, seed=42) if rec_vals else [0.0, 0.0],
            "mrr": macro["mean_mrr"],
            "mrr_ci_95": bootstrap_ci(mrr_vals, seed=42) if mrr_vals else [0.0, 0.0],
            "ndcg@10": macro["mean_ndcg@10"],
            "ndcg@10_ci_95": bootstrap_ci(ndcg_vals, seed=42) if ndcg_vals else [0.0, 0.0],
        }

    return {
        "caption": MANDATORY_CAPTION,
        "note": LABEL_ORIGIN_NOTE,
        "audit_scope": "Relevant postings with ZERO literal match of candidate declared skills",
        "sample_size": "51 pairs, 17 profiles",
        "is_indicative_only": True,
        "sample_size_assessment": {
            "total_zero_overlap_relevant_pairs": total_zero_overlap_pairs,
            "profiles_with_zero_overlap_relevant_docs": len(profiles_with_zero_overlap),
            "total_profiles": len(profiles),
            "excluded_zero_relevant_profiles": len(profiles) - len(profiles_with_zero_overlap),
            "is_sample_too_small": True,
            "statement": (
                f"Sample size is {total_zero_overlap_pairs} pairs across {len(profiles_with_zero_overlap)} "
                f"profiles. This check is indicative only due to the small sample size and high variance."
            ),
        },
        "strategy_metrics_on_zero_overlap_subset": strat_results,
    }


def execute_full_evaluation() -> None:
    """Run end-to-end retrieval evaluation pipeline and save all reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_candidate_profiles()
    qrels = load_qrels()
    jobs_df = pd.read_parquet(JOBS_PATH)
    jobs_df["id"] = jobs_df["id"].astype(str)
    jobs_df = jobs_df.set_index("id")

    print(f"Loaded {len(profiles)} profiles and {len(qrels)} qrels profile entries.")

    # Instantiate retrievers outside the timed benchmarking region
    print("Initializing retrievers (BM25, Dense, Hybrid RRF, Cross-Encoder)...")
    t0_bm25 = time.perf_counter()
    bm25 = BM25Retriever()
    t_bm25_build = (time.perf_counter() - t0_bm25) * 1000.0

    t0_dense = time.perf_counter()
    dense = DenseRetriever()
    t_dense_build = (time.perf_counter() - t0_dense) * 1000.0

    t0_hybrid = time.perf_counter()
    hybrid = HybridRRFRetriever(bm25, dense)
    t_hybrid_build = (time.perf_counter() - t0_hybrid) * 1000.0

    t0_ce = time.perf_counter()
    cross_encoder = CrossEncoderRetriever(hybrid)
    t_ce_build = (time.perf_counter() - t0_ce) * 1000.0

    startup_times = {
        "bm25_index_build_ms": round(t_bm25_build, 2),
        "dense_init_ms": round(t_dense_build, 2),
        "hybrid_init_ms": round(t_hybrid_build, 2),
        "cross_encoder_load_ms": round(t_ce_build, 2),
    }
    print(f"One-off startup build times: BM25={t_bm25_build:.1f}ms, Dense={t_dense_build:.1f}ms, Cross-Encoder={t_ce_build:.1f}ms")

    retrievers = {
        "bm25": bm25,
        "dense": dense,
        "hybrid_rrf": hybrid,
        "cross_encoder": cross_encoder,
    }

    strategy_display_names = {
        "bm25": "BM25 Only",
        "dense": "Dense Only (ChromaDB)",
        "hybrid_rrf": "Hybrid RRF (k=60)",
        "cross_encoder": "Hybrid + Cross-Encoder",
    }

    # ---------------------------------------------------------
    # 1. Main Setting Evaluation (Filters ON, Skill Boost ON w=0.2)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("EVALUATING MAIN SETTING (Filters ON, Skill Boost ON w=0.2)")
    print("=" * 75)

    main_per_profile: Dict[str, Dict[str, Dict[str, Any]]] = {}
    main_ranked_ids: Dict[str, Dict[str, List[str]]] = {}
    main_summary: Dict[str, Any] = {
        "caption": MANDATORY_CAPTION,
        "note": LABEL_ORIGIN_NOTE,
        "setting": "Main Product Pipeline (Location & Seniority Filter Fallback ON, Skill Boost ON w=0.2)",
        "strategies": {},
    }
    all_per_profile_rows_main: List[Dict[str, Any]] = []

    for name, ret in retrievers.items():
        print(f"Evaluating {strategy_display_names[name]}...")
        sys.stdout.flush()
        prof_m, ranked_ids = evaluate_strategy_over_profiles(
            retriever=ret,
            profiles=profiles,
            qrels=qrels,
            apply_filters=True,
            apply_boost=True,
            boost_weight=0.2,
            k_cutoff=10,
        )
        main_per_profile[name] = prof_m
        main_ranked_ids[name] = ranked_ids

        # Compute macro-averages
        macro = compute_macro_averages(prof_m)

        # Compute 95% bootstrap CIs
        p5_vals = [float(v["precision@5"]) for v in prof_m.values()]
        ndcg_vals = [float(v["ndcg@10"]) for v in prof_m.values()]
        rec_vals = [float(v["recall@10"]) for v in prof_m.values() if v["recall@10"] is not None]
        mrr_vals = [float(v["mrr"]) for v in prof_m.values() if v["mrr"] is not None]

        p5_ci = bootstrap_ci(p5_vals, seed=42)
        ndcg_ci = bootstrap_ci(ndcg_vals, seed=42)
        rec_ci = bootstrap_ci(rec_vals, seed=42)
        mrr_ci = bootstrap_ci(mrr_vals, seed=42)

        main_summary["strategies"][name] = {
            "display_name": strategy_display_names[name],
            "precision@5": macro["mean_precision@5"],
            "precision@5_ci_95": p5_ci,
            "recall@10": macro["mean_recall@10"],
            "recall@10_ci_95": rec_ci,
            "mrr": macro["mean_mrr"],
            "mrr_ci_95": mrr_ci,
            "ndcg@10": macro["mean_ndcg@10"],
            "ndcg@10_ci_95": ndcg_ci,
            "unjudged_rate@10": macro["mean_unjudged_rate@10"],
            "excluded_zero_relevant_profiles": macro["excluded_zero_relevant_profiles"],
        }

        for prof_id, vals in prof_m.items():
            all_per_profile_rows_main.append({
                "strategy": name,
                "profile_id": prof_id,
                **vals,
            })

    # Save Main reports
    main_summary_path = REPORTS_DIR / "retrieval_metrics_main.json"
    with open(main_summary_path, "w", encoding="utf-8") as f:
        json.dump(main_summary, f, indent=2)

    pd.DataFrame(all_per_profile_rows_main).to_csv(
        REPORTS_DIR / "retrieval_per_profile_main.csv", index=False
    )
    print(f"Saved main metrics to {main_summary_path}")

    # ---------------------------------------------------------
    # 2. Secondary Setting Evaluation (Filters OFF, Boost OFF)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("EVALUATING SECONDARY SETTING (Pure Retrieval: Filters OFF, Boost OFF)")
    print("=" * 75)

    pure_per_profile: Dict[str, Dict[str, Dict[str, Any]]] = {}
    pure_summary: Dict[str, Any] = {
        "caption": MANDATORY_CAPTION,
        "note": LABEL_ORIGIN_NOTE,
        "setting": "Pure Retrieval (Filters OFF, Skill Boost OFF) - Not comparable (labels pooled under filters, 76% unjudged)",
        "comparability_warning": (
            "Not comparable (labels pooled under filters, 76% unjudged). Because candidate pooling was performed "
            "under active location and seniority filters, out-of-market jobs retrieved when filters are off were never "
            "pooled or evaluated, causing unjudged rates of 76%–85%. These numbers do not reflect pure retrieval quality."
        ),
        "strategies": {},
    }
    all_per_profile_rows_pure: List[Dict[str, Any]] = []

    for name, ret in retrievers.items():
        print(f"Evaluating Pure {strategy_display_names[name]}...")
        sys.stdout.flush()
        prof_m, _ = evaluate_strategy_over_profiles(
            retriever=ret,
            profiles=profiles,
            qrels=qrels,
            apply_filters=False,
            apply_boost=False,
            boost_weight=0.0,
            k_cutoff=10,
        )
        pure_per_profile[name] = prof_m
        macro = compute_macro_averages(prof_m)

        p5_vals = [float(v["precision@5"]) for v in prof_m.values()]
        ndcg_vals = [float(v["ndcg@10"]) for v in prof_m.values()]
        rec_vals = [float(v["recall@10"]) for v in prof_m.values() if v["recall@10"] is not None]
        mrr_vals = [float(v["mrr"]) for v in prof_m.values() if v["mrr"] is not None]

        pure_summary["strategies"][name] = {
            "display_name": strategy_display_names[name],
            "precision@5": macro["mean_precision@5"],
            "precision@5_ci_95": bootstrap_ci(p5_vals, seed=42),
            "recall@10": macro["mean_recall@10"],
            "recall@10_ci_95": bootstrap_ci(rec_vals, seed=42),
            "mrr": macro["mean_mrr"],
            "mrr_ci_95": bootstrap_ci(mrr_vals, seed=42),
            "ndcg@10": macro["mean_ndcg@10"],
            "ndcg@10_ci_95": bootstrap_ci(ndcg_vals, seed=42),
            "unjudged_rate@10": macro["mean_unjudged_rate@10"],
        }

        for prof_id, vals in prof_m.items():
            all_per_profile_rows_pure.append({
                "strategy": name,
                "profile_id": prof_id,
                **vals,
            })

    pure_summary_path = REPORTS_DIR / "retrieval_metrics_pure.json"
    with open(pure_summary_path, "w", encoding="utf-8") as f:
        json.dump(pure_summary, f, indent=2)

    pd.DataFrame(all_per_profile_rows_pure).to_csv(
        REPORTS_DIR / "retrieval_per_profile_pure.csv", index=False
    )
    print(f"Saved pure metrics to {pure_summary_path}")

    # ---------------------------------------------------------
    # 3. Paired Comparisons (Paired Bootstrap on Differences)
    # ---------------------------------------------------------
    print("\nComputing paired strategy comparisons with 95% CIs...")
    pairs_to_compare = [
        ("bm25", "dense"),
        ("hybrid_rrf", "bm25"),
        ("hybrid_rrf", "dense"),
        ("cross_encoder", "hybrid_rrf"),
        ("cross_encoder", "bm25"),
    ]

    paired_results: Dict[str, Any] = {
        "caption": MANDATORY_CAPTION,
        "note": LABEL_ORIGIN_NOTE,
        "comparisons": {},
    }

    for a, b in pairs_to_compare:
        comp_key = f"{a}_vs_{b}"
        paired_results["comparisons"][comp_key] = {
            "strategy_a": strategy_display_names[a],
            "strategy_b": strategy_display_names[b],
            "metrics": {
                "precision@5": run_paired_bootstrap(main_per_profile[a], main_per_profile[b], "precision@5"),
                "recall@10": run_paired_bootstrap(main_per_profile[a], main_per_profile[b], "recall@10"),
                "mrr": run_paired_bootstrap(main_per_profile[a], main_per_profile[b], "mrr"),
                "ndcg@10": run_paired_bootstrap(main_per_profile[a], main_per_profile[b], "ndcg@10"),
            },
        }

    paired_path = REPORTS_DIR / "retrieval_paired_comparisons.json"
    with open(paired_path, "w", encoding="utf-8") as f:
        json.dump(paired_results, f, indent=2)
    print(f"Saved paired comparisons to {paired_path}")

    # ---------------------------------------------------------
    # 4. Latency Benchmarks
    # ---------------------------------------------------------
    latency_report = benchmark_latencies(retrievers, profiles, runs_per_query=25, startup_times=startup_times)
    latency_path = REPORTS_DIR / "retrieval_latency.json"
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_report, f, indent=2)
    print(f"Saved latency benchmark to {latency_path}")
    sys.stdout.flush()

    # ---------------------------------------------------------
    # 5. Ablation for Top Strategy by nDCG@10
    # ---------------------------------------------------------
    # Find best strategy by nDCG@10 in main setting
    best_strat = max(
        main_summary["strategies"].keys(),
        key=lambda k: main_summary["strategies"][k]["ndcg@10"],
    )
    print(f"\nTop strategy by nDCG@10 is '{best_strat}' ({strategy_display_names[best_strat]}).")
    ablation_report = run_retrieval_ablation(
        retriever=retrievers[best_strat],
        strategy_name=best_strat,
        profiles=profiles,
        qrels=qrels,
    )
    ablation_path = REPORTS_DIR / "retrieval_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_report, f, indent=2)
    print(f"Saved ablation study to {ablation_path}")
    sys.stdout.flush()

    # ---------------------------------------------------------
    # 6. Circularity Check
    # ---------------------------------------------------------
    circ_report = run_circularity_audit(main_ranked_ids, profiles, qrels, jobs_df)
    circ_path = REPORTS_DIR / "retrieval_circularity_check.json"
    with open(circ_path, "w", encoding="utf-8") as f:
        json.dump(circ_report, f, indent=2)
    print(f"Saved circularity check to {circ_path}")
    sys.stdout.flush()

    # ---------------------------------------------------------
    # Summary Output Table
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"RETRIEVAL EVALUATION RESULTS ({MANDATORY_CAPTION})")
    print("=" * 90)
    print(f"{'Strategy':<24} | {'P@5 (95% CI)':<18} | {'Recall@10 (95% CI)':<20} | {'MRR (95% CI)':<18} | {'nDCG@10 (95% CI)':<18} | {'p50 Latency':<10}")
    print("-" * 90)
    for name in ["bm25", "dense", "hybrid_rrf", "cross_encoder"]:
        info = main_summary["strategies"][name]
        p5_str = f"{info['precision@5']:.3f} [{info['precision@5_ci_95'][0]:.2f},{info['precision@5_ci_95'][1]:.2f}]"
        rec_str = f"{info['recall@10']:.3f} [{info['recall@10_ci_95'][0]:.2f},{info['recall@10_ci_95'][1]:.2f}]"
        mrr_str = f"{info['mrr']:.3f} [{info['mrr_ci_95'][0]:.2f},{info['mrr_ci_95'][1]:.2f}]"
        ndcg_str = f"{info['ndcg@10']:.3f} [{info['ndcg@10_ci_95'][0]:.2f},{info['ndcg@10_ci_95'][1]:.2f}]"
        
        if name == "cross_encoder":
            p50_str = f"{latency_report['cross_encoder']['end_to_end']['p50_ms']:.1f}ms"
        else:
            p50_str = f"{latency_report[name]['p50_ms']:.1f}ms"

        print(f"{strategy_display_names[name]:<24} | {p5_str:<18} | {rec_str:<20} | {mrr_str:<18} | {ndcg_str:<18} | {p50_str:<10}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    execute_full_evaluation()
