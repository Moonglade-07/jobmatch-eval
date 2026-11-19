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
    normalize_base_scores: bool = True,
) -> List[SearchResult]:
    """Apply an explainable additive score boost based on matched candidate skills.

    Normalizes base scores to [0, 1] per candidate pool using min-max scaling so that
    boost weight `w` operates on an identical scale across all retrieval strategies
    (BM25, Dense, Hybrid RRF, and Cross-Encoder).

    Formula:
        norm_base_score = (raw_score - min_score) / (max_score - min_score)  [if max > min]
        skill_boost = weight * (matched_skills_count / total_skills_count)
        final_score = norm_base_score + skill_boost

    Each SearchResult is annotated with:
        raw_score: original retriever score before normalization.
        base_score: min-max normalized base score in [0, 1].
        skill_boost: additive boost earned.
        matched_skills: list of matching skills found in posting text.
        score: final combined ranking score.

    Args:
        results: List of SearchResult objects from preliminary retrieval.
        skills: Candidate's skill list.
        weight: Importance weight parameter `w` for skill alignment (default: 0.2).
        normalize_base_scores: Whether to min-max scale base scores to [0, 1].

    Returns:
        Re-ranked and annotated SearchResult list sorted descending by final score.
    """
    if not results:
        return []

    # Preserve raw scores
    for r in results:
        if not hasattr(r, "raw_score") or r.raw_score == 0.0:
            r.raw_score = r.score

    # Min-max normalize base scores over the candidate list
    if normalize_base_scores and len(results) > 0:
        raw_scores = [float(r.raw_score) for r in results]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max_s - min_s
        for r in results:
            if score_range > 1e-6:
                r.base_score = round(float((r.raw_score - min_s) / score_range), 4)
            else:
                r.base_score = 1.0
    else:
        for r in results:
            r.base_score = round(float(r.score), 4)

    if not skills:
        for r in results:
            r.skill_boost = 0.0
            r.score = r.base_score
            r.matched_skills = []
        return sorted(results, key=lambda x: (x.filter_tier, -x.score))

    total_skills = len(skills)
    boosted_results: List[SearchResult] = []

    for r in results:
        combined_text = f"{r.title} {r.description}"
        matched = extract_matched_skills(skills, combined_text)
        fraction = len(matched) / total_skills if total_skills > 0 else 0.0
        boost = round(weight * fraction, 4)

        r.skill_boost = boost
        r.score = round(r.base_score + boost, 4)
        r.matched_skills = matched
        boosted_results.append(r)

    boosted_results.sort(key=lambda x: (x.filter_tier, -x.score))
    return boosted_results


def is_remote_posting(title: str = "", description: str = "", location: str = "") -> bool:
    """Detect whether a posting offers remote work based on location, title, or description.

    Applies positive remote signal patterns while filtering out negations (e.g. 'unavailable
    for remote work', 'remote work is not available'), technical remote access verbs (e.g.
    'remote into users computers'), and generic boilerplate perks.

    Args:
        title: Job posting title.
        description: Job posting description text.
        location: Job posting location string.

    Returns:
        True if the posting offers remote work, False otherwise.
    """
    loc = (location or "").lower()
    if "remote" in loc:
        return True

    t = (title or "").lower()
    if re.search(r"\b(remote|virtual|telecommute|work from home|wfh)\b", t):
        return True

    d = (description or "").lower()
    if not d:
        return False

    # 1. Filter out explicit negations and disqualifying phrases
    if re.search(r"\b(is\s+)?unavailable\s+for\s+remote\b", d):
        return False
    if re.search(r"\bremote\s+work\s+is\s+not\s+available\b", d):
        return False
    if re.search(r"\b(no|not|neither|non-?)\s+(a\s+)?remote\b", d):
        return False
    if re.search(r"\b(not\s+eligible\s+for\s+remote|on-?site\s+only)\b", d):
        return False
    if re.search(r"\bperks\s+like\s+flexible\/remote\b", d):
        return False
    if re.search(r"\bremote\s+into\b", d):
        return False
    if re.search(r"\ba\s+remote\s+career\s+transition\b", d):
        return False

    # 2. Match targeted positive remote signals
    remote_signals = [
        r"\b100%\s*remote\b",
        r"\bfully\s*remote\b",
        r"\bremote-?first\b",
        r"\bwork\s*(from|at)\s*home\b",
        r"\bwfh\b",
        r"\bwork\s*remotely\b",
        r"\bremote\s*(position|role|job|opportunity|environment|option|friendly|status)\b",
        r"\bremote\s*work\b",
        r"\b(flexibility\s+for\s+remote|option\s+(for\s+.*|to\s+be)\s+remote|can\s+be\s+remote|open\s+to\s+remote)\b",
        r"\b(location|workplace):\s*remote\b",
        r"\(\s*remote\b",
        r"\b(part-time|full-time)\s+remote\b",
    ]

    for pattern in remote_signals:
        if re.search(pattern, d):
            return True

    return False


def matches_location(
    job_location: str,
    preferred_location: str,
    title: str = "",
    description: str = "",
) -> bool:
    """Check if job location satisfies preferred location constraint.

    If candidate prefers 'Remote', the posting must offer remote work.
    If the posting itself is remote, it satisfies any candidate location preference.
    Otherwise, standard geographic matching (exact or substring) is performed.

    Args:
        job_location: Normalized job location.
        preferred_location: Candidate preferred location.
        title: Optional job title for remote keyword detection.
        description: Optional job description for remote keyword detection.

    Returns:
        Boolean indicating location compatibility.
    """
    if not preferred_location or not preferred_location.strip():
        return True
    pref = preferred_location.strip().lower()
    job_loc = (job_location or "").strip().lower()
    remote_job = is_remote_posting(title=title, description=description, location=job_location)

    # If candidate requests Remote, the job MUST offer remote work
    if "remote" in pref:
        return remote_job

    # If job offers remote work, candidate anywhere is compatible
    if remote_job:
        return True

    # Otherwise, check geographic match (e.g. "Austin, TX" in "Austin, TX" or state match)
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
         Annotates matched results with `filter_tier = 1`.
      2. Tier 2 (Relax Seniority): If Tier 1 yields < k results, relax the
         seniority constraint while strictly preserving the location constraint.
         Annotates newly added results with `filter_tier = 2`.
      3. Tier 3 (Global Backfill): If Tier 2 still yields < k results, backfill
         remaining slots with top-scoring unfiltered results to guarantee
         that min(k, len(results)) items are returned.
         Annotates newly added results with `filter_tier = 3`.

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
        for r in results:
            r.filter_tier = 1
        return results[:k], "tier_0_no_filters"

    # Tier 1: Strict match on both location and seniority
    tier_1: List[SearchResult] = []
    tier_1_ids = set()
    for r in results:
        loc_ok = matches_location(
            r.location, preferred_location, title=r.title, description=r.description
        )
        lvl_ok = matches_seniority(r.experience_level, seniority)
        if loc_ok and lvl_ok:
            r.filter_tier = 1
            tier_1.append(r)
            tier_1_ids.add(r.job_id)

    if len(tier_1) >= k:
        return tier_1[:k], "tier_1_strict"

    # Tier 2: Relax seniority, preserve location (or relax location if no location preference)
    tier_2: List[SearchResult] = []
    tier_2_ids = set()
    for r in results:
        if r.job_id in tier_1_ids:
            continue
        if has_loc:
            loc_ok = matches_location(
                r.location, preferred_location, title=r.title, description=r.description
            )
            if loc_ok:
                r.filter_tier = 2
                tier_2.append(r)
                tier_2_ids.add(r.job_id)
        else:
            lvl_ok = matches_seniority(r.experience_level, seniority)
            if lvl_ok:
                r.filter_tier = 2
                tier_2.append(r)
                tier_2_ids.add(r.job_id)

    combined_1_2 = tier_1 + tier_2
    if len(combined_1_2) >= k:
        label = "tier_2_relaxed_seniority" if has_loc else "tier_2_relaxed_location"
        return combined_1_2[:k], label

    # Tier 3: Global backfill from remaining unfiltered results
    existing_ids = tier_1_ids | tier_2_ids
    tier_3: List[SearchResult] = []
    for r in results:
        if r.job_id not in existing_ids:
            r.filter_tier = 3
            tier_3.append(r)

    combined_all = combined_1_2 + tier_3
    return combined_all[:k], "tier_3_global_backfill"
