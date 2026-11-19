"""Side-by-side comparison demo across all four retrieval strategies for a sample candidate profile.

Strategies evaluated:
  1. BM25 Only: Lexical keyword search using rank_bm25.
  2. Dense Only: Semantic vector search using ChromaDB and all-MiniLM-L6-v2.
  3. Hybrid (BM25 + Dense RRF): Reciprocal Rank Fusion with k=60.
  4. Hybrid + Cross-Encoder Re-rank: ms-marco-MiniLM-L-6-v2 re-ranking top 50 fused candidates.

Usage:
  python src/demo_comparison.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep offline mode for Hugging Face and disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.models import Profile, SearchResult
from src.retrieval import (
    BM25Retriever,
    CrossEncoderRetriever,
    DenseRetriever,
    HybridRRFRetriever,
)


def create_sample_profile() -> Profile:
    """Create a representative candidate profile for job search demonstration.

    Returns:
        Populated candidate Profile.
    """
    return Profile(
        target_role="Machine Learning Engineer",
        skills=[
            "Python",
            "PyTorch",
            "NLP",
            "Transformers",
            "ChromaDB",
            "FastAPI",
            "Docker",
        ],
        preferred_location="Remote",
        seniority="Mid-Senior level",
        resume_text=(
            "Machine Learning Engineer with 4+ years of industry experience developing "
            "production search, semantic retrieval, and natural language processing pipelines. "
            "Proficient in PyTorch, Hugging Face Transformers, vector indexing with ChromaDB, "
            "and deploying microservices with FastAPI and Docker."
        ),
    )


def run_comparison(
    profile: Profile,
    k: int = 5,
    apply_filters: bool = True,
    apply_boost: bool = True,
) -> Dict[str, List[SearchResult]]:
    """Execute search across all four retrieval strategies for the provided profile.

    Args:
        profile: Candidate profile to search with.
        k: Number of ranked postings to return per strategy.
        apply_filters: Whether to apply metadata filtering with fallback.
        apply_boost: Whether to apply explainable skill boost.

    Returns:
        Mapping from strategy name to ranked list of SearchResult objects.
    """
    print("=" * 80)
    print("INITIALIZING RETRIEVAL STRATEGIES")
    print("=" * 80)
    print("  [1/4] Initializing BM25 retriever...")
    bm25 = BM25Retriever()

    print("  [2/4] Initializing Dense vector retriever (ChromaDB + all-MiniLM-L6-v2)...")
    dense = DenseRetriever()

    print("  [3/4] Initializing Hybrid RRF retriever (k=60)...")
    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_k=60)

    print("  [4/4] Initializing Cross-Encoder re-ranker (ms-marco-MiniLM-L-6-v2)...")
    cross = CrossEncoderRetriever(hybrid_retriever=hybrid, top_candidates_depth=50)

    strategies = {
        "BM25 Only": bm25,
        "Dense Only": dense,
        "Hybrid RRF (k=60)": hybrid,
        "Hybrid + Cross-Encoder": cross,
    }

    results: Dict[str, List[SearchResult]] = {}
    print("\n" + "=" * 80)
    print(f"EXECUTING SEARCH FOR TOP {k} POSTINGS")
    print("=" * 80)

    for name, retriever in strategies.items():
        print(f"  Searching with '{name}'...")
        res = retriever.search(
            profile=profile,
            k=k,
            apply_filters=apply_filters,
            apply_boost=apply_boost,
            boost_weight=0.2,
        )
        results[name] = res

    return results


def print_comparison_table(
    profile: Profile,
    results: Dict[str, List[SearchResult]],
    k: int = 5,
) -> None:
    """Print candidate profile and side-by-side strategy comparison.

    Args:
        profile: Candidate profile.
        results: Dictionary mapping strategy name to search results.
        k: Number of rankings displayed.
    """
    print("\n" + "=" * 100)
    print("CANDIDATE PROFILE")
    print("=" * 100)
    print(f"  Target Role        : {profile.target_role}")
    print(f"  Preferred Location : {profile.preferred_location}")
    print(f"  Seniority Level    : {profile.seniority}")
    print(f"  Key Skills         : {', '.join(profile.skills)}")
    print(f"  Resume Narrative   : {profile.resume_text}")
    print("=" * 100)

    # Strategy individual breakdown
    for strat_name, res_list in results.items():
        tier_counts = {1: 0, 2: 0, 3: 0}
        for res in res_list:
            tier_counts[res.filter_tier] = tier_counts.get(res.filter_tier, 0) + 1

        print(f"\n--- Strategy: {strat_name} ---")
        print(
            f"  Filter Tier Distribution: "
            f"Tier 1 (Strict): {tier_counts[1]} | "
            f"Tier 2 (Relaxed Seniority): {tier_counts[2]} | "
            f"Tier 3 (Global Backfill): {tier_counts[3]}"
        )
        for rank, res in enumerate(res_list, 1):
            matched_str = ", ".join(res.matched_skills) if res.matched_skills else "None"
            tier_label = {
                1: "Tier 1 (Strict Match)",
                2: "Tier 2 (Relaxed Seniority)",
                3: "Tier 3 (Global Backfill)",
            }.get(res.filter_tier, f"Tier {res.filter_tier}")

            print(
                f"  Rank {rank} | Score: {res.score:.4f} "
                f"(Base: {res.base_score:.4f}, Raw: {res.raw_score:.2f}, Boost: +{res.skill_boost:.4f}) | {tier_label}\n"
                f"    ID: {res.job_id} | Title: {res.title} | Company: {res.company}\n"
                f"    Location: {res.location} | Level: {res.experience_level}\n"
                f"    Matched Skills ({len(res.matched_skills)}): {matched_str}"
            )

    # Side-by-side summary table
    print("\n" + "=" * 130)
    print(f"{'SIDE-BY-SIDE SUMMARY (TOP ' + str(k) + ')':^130}")
    print("=" * 130)

    header = f"{'Rank':<5} | {'Strategy':<24} | {'Job ID':<10} | {'Score':<7} | {'Tier':<6} | {'Location':<14} | {'Title'}"
    print(header)
    print("-" * 130)

    for rank_idx in range(k):
        for strat_name in results.keys():
            res = results[strat_name][rank_idx] if rank_idx < len(results[strat_name]) else None
            if res:
                title_truncated = (res.title[:42] + "..") if len(res.title) > 44 else res.title
                print(
                    f"{rank_idx + 1:<5} | {strat_name:<24} | {res.job_id:<10} | "
                    f"{res.score:<7.4f} | Tier {res.filter_tier:<1} | {res.location:<14} | {title_truncated}"
                )
        if rank_idx < k - 1:
            print("-" * 130)
    print("=" * 130)


def main() -> None:
    """Run full side-by-side comparison pipeline for sample profile."""
    profile = create_sample_profile()
    results = run_comparison(profile=profile, k=5, apply_filters=True, apply_boost=True)
    print_comparison_table(profile=profile, results=results, k=5)


if __name__ == "__main__":
    main()
