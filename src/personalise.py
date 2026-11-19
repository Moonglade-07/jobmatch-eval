"""Personalization utilities: query synthesis, metadata filtering with fallback, and explainable skill boosting."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Profile, SearchResult


def build_query_text(profile: Profile) -> str:
    """Synthesize candidate profile attributes into dense/lexical search query text.

    Combines the target role, primary technical skills, and narrative resume text
    with structured prefixes to ensure balanced matching across both keyword (BM25)
    and semantic vector retrieval models.

    Args:
        profile: Candidate Profile dataclass instance.

    Returns:
        Structured query string for retrieval.
    """
    sections: List[str] = []

    if profile.target_role and profile.target_role.strip():
        sections.append(f"Target Role: {profile.target_role.strip()}")

    if profile.skills:
        clean_skills = [s.strip() for s in profile.skills if s.strip()]
        if clean_skills:
            sections.append(f"Key Skills: {', '.join(clean_skills)}")

    if profile.seniority and profile.seniority.strip():
        sections.append(f"Desired Seniority: {profile.seniority.strip()}")

    if profile.resume_text and profile.resume_text.strip():
        sections.append(f"Candidate Experience: {profile.resume_text.strip()}")

    return "\n".join(sections)


def extract_matched_skills(
    skills: List[str], text: str
) -> List[str]:
    """Identify which candidate skills are explicitly referenced in job posting text.

    Uses case-insensitive word-boundary matching with regex escaping to prevent
    partial substring false positives (e.g., 'C' inside 'Cloud' or 'Java' inside 'JavaScript').

    Args:
        skills: List of candidate skills to search for.
        text: Combined job title and description text.

    Returns:
        Sorted list of matched skills found in text.
    """
    if not skills or not text:
        return []

    matched: Set[str] = set()
    for skill in skills:
        cleaned_skill = skill.strip()
        if not cleaned_skill:
            continue

        # Adjust word boundaries for symbols like C++, C#, .NET, C
        escaped = re.escape(cleaned_skill)
        pattern = rf"(?<![\w\+\#]){escaped}(?![\w\+\#])"
        if re.search(pattern, text, re.IGNORECASE):
            matched.add(cleaned_skill)

    return sorted(matched)


def apply_skill_boost(
    results: List[SearchResult],
    skills: List[str],
    weight: float = 0.2,
) -> List[SearchResult]:
    """Apply an explainable additive score boost based on matched candidate skills.

    Formula:
        final_score = base_score + (weight * matched_skills_fraction)
        matched_skills_fraction = len(matched_skills) / len(candidate_skills)

    Each SearchResult is annotated with matched_skills, base_score, and skill_boost.
    Results are re-sorted descending by the new final_score.

    Args:
        results: List of SearchResult objects from preliminary retrieval.
        skills: Candidate's skill list.
        weight: Importance weight parameter `w` for skill alignment (default: 0.2).

    Returns:
        Re-ranked and annotated SearchResult list.
    """
    if not skills:
        for r in results:
            r.base_score = r.score
            r.skill_boost = 0.0
            r.matched_skills = []
        return sorted(results, key=lambda x: x.score, reverse=True)

    total_skills = len(skills)
    boosted_results: List[SearchResult] = []

    for r in results:
        combined_text = f"{r.title} {r.description}"
        matched = extract_matched_skills(skills, combined_text)
        fraction = len(matched) / total_skills if total_skills > 0 else 0.0
        boost = round(weight * fraction, 4)

        r.base_score = r.score
        r.skill_boost = boost
        r.score = round(r.base_score + boost, 4)
        r.matched_skills = matched
        boosted_results.append(r)

    boosted_results.sort(key=lambda x: x.score, reverse=True)
    return boosted_results


def matches_location(job_location: str, preferred_location: str) -> bool:
    """Check if job location satisfies preferred location constraint.

    Treats 'Remote' jobs as compatible with any location preference.

    Args:
        job_location: Normalized job location.
        preferred_location: Candidate preferred location.

    Returns:
        Boolean indicating location compatibility.
    """
    if not preferred_location or not preferred_location.strip():
        return True
    pref = preferred_location.strip().lower()
    job_loc = (job_location or "").strip().lower()

    if "remote" in pref or "remote" in job_loc:
        return True

    # Exact or substring match (e.g. "Austin, TX" in "Austin, TX" or "TX")
    return pref in job_loc or job_loc in pref


def matches_seniority(job_level: str, preferred_level: str) -> bool:
    """Check if job seniority level satisfies preferred seniority constraint.

    Postings with 'Not Specified' seniority are accepted under all preferences.

    Args:
        job_level: Normalized job experience level.
        preferred_level: Candidate preferred seniority.

    Returns:
        Boolean indicating seniority compatibility.
    """
    if not preferred_level or not preferred_level.strip():
        return True
    pref = preferred_level.strip().lower()
    job_lvl = (job_level or "").strip().lower()

    if job_lvl == "not specified":
        return True

    return pref == job_lvl or pref in job_lvl


def filter_with_fallback(
    results: List[SearchResult],
    preferred_location: str,
    seniority: str,
    k: int = 10,
) -> Tuple[List[SearchResult], str]:
    """Filter search results with a strict multi-tier fallback hierarchy.

    Fallback Hierarchy:
      1. Tier 1 (Strict): Both `preferred_location` and `seniority` must match.
         If len(matches) >= k, returns top k strictly matching results.
      2. Tier 2 (Relax Seniority): If Tier 1 yields < k results, relax the
         seniority constraint while strictly preserving the location constraint.
         If len(matches) >= k, returns top k location-compatible results.
      3. Tier 3 (Global Backfill): If Tier 2 still yields < k results, backfill
         remaining slots with top-scoring unfiltered results to guarantee
         that exactly min(k, len(results)) items are returned.

    Args:
        results: Candidate SearchResult pool in rank order.
        preferred_location: Candidate location preference.
        seniority: Candidate seniority preference.
        k: Minimum requested results.

    Returns:
        Tuple of (filtered SearchResult list of length min(k, len(results)), fallback_tier_applied).
    """
    if not results:
        return [], "tier_0_empty"

    has_loc = bool(preferred_location and preferred_location.strip())
    has_lvl = bool(seniority and seniority.strip())

    if not has_loc and not has_lvl:
        return results[:k], "tier_0_no_filters"

    # Tier 1: Strict match on both location and seniority
    tier_1: List[SearchResult] = [
        r
        for r in results
        if matches_location(r.location, preferred_location)
        and matches_seniority(r.experience_level, seniority)
    ]

    if len(tier_1) >= k:
        return tier_1[:k], "tier_1_strict"

    # Tier 2: Relax seniority, preserve location
    if has_loc:
        tier_2: List[SearchResult] = [
            r for r in results if matches_location(r.location, preferred_location)
        ]
        if len(tier_2) >= k:
            return tier_2[:k], "tier_2_relaxed_seniority"
    else:
        # If no location was given, relax location, preserve seniority
        tier_2 = [
            r for r in results if matches_seniority(r.experience_level, seniority)
        ]
        if len(tier_2) >= k:
            return tier_2[:k], "tier_2_relaxed_location"

    # Tier 3: Global backfill from remaining unfiltered results
    selected_ids = {r.job_id for r in tier_2}
    backfill: List[SearchResult] = [
        r for r in results if r.job_id not in selected_ids
    ]
    combined = tier_2 + backfill
    return combined[:k], "tier_3_global_backfill"
