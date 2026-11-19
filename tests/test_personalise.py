"""Unit tests for personalization: query synthesis, skill matching, score boost, and fallback filtering."""

from __future__ import annotations

import pytest

from src.models import Profile, SearchResult
from src.personalise import (
    apply_skill_boost,
    build_query_text,
    extract_matched_skills,
    filter_with_fallback,
    matches_location,
    matches_seniority,
)


def test_build_query_text() -> None:
    """Test candidate query synthesis includes all relevant profile sections."""
    profile = Profile(
        target_role="Senior ML Engineer",
        skills=["Python", "PyTorch", "Transformers"],
        seniority="Mid-Senior level",
        resume_text="5 years building search engines and neural retrieval pipelines.",
    )
    query = build_query_text(profile)

    assert "Target Role: Senior ML Engineer" in query
    assert "Key Skills: Python, PyTorch, Transformers" in query
    assert "Desired Seniority: Mid-Senior level" in query
    assert "Candidate Experience: 5 years building search engines" in query


def test_extract_matched_skills_exact_boundaries() -> None:
    """Test skill extraction prevents substring false-positives and handles symbols."""
    skills = ["Python", "C", "C++", "Java", "SQL", "Go"]
    # "JavaScript" contains "Java" substring; "Cloud" contains "C" substring.
    text = "We use Python, modern C++, and SQL on GCP cloud. No Java or JavaScript."

    matched = extract_matched_skills(skills, text)
    # Must match: Python, C++, SQL, Java (Java appears as standalone word).
    # Must NOT match: C (because of 'C' in cloud), Go (not present).
    assert "Python" in matched
    assert "C++" in matched
    assert "SQL" in matched
    assert "Java" in matched
    assert "C" not in matched
    assert "Go" not in matched


def test_apply_skill_boost_reorders_correctly() -> None:
    """Test that skill boost changes ordering only when skill match fraction warrants it."""
    candidate_skills = ["Python", "PyTorch", "Docker", "Kubernetes"]

    # Candidate 1: Higher base score (0.80), but 0 matched skills.
    # Candidate 2: Slightly lower base score (0.75), but matches all 4 skills.
    r1 = SearchResult(
        job_id="job_high_base",
        score=0.80,
        title="Software Engineer",
        description="Work with Ruby and Rails on AWS.",
    )
    r2 = SearchResult(
        job_id="job_high_skills",
        score=0.75,
        title="MLOps Engineer",
        description="Requires Python, PyTorch, Docker, and Kubernetes deployment.",
    )

    # With weight w=0.2:
    # r1 boost = 0.2 * (0/4) = 0.0 -> final = 0.80
    # r2 boost = 0.2 * (4/4) = 0.2 -> final = 0.75 + 0.20 = 0.95
    # Result: r2 must overtake r1!
    boosted = apply_skill_boost([r1, r2], skills=candidate_skills, weight=0.2)

    assert boosted[0].job_id == "job_high_skills"
    assert boosted[0].score == 0.95
    assert boosted[0].skill_boost == 0.2
    assert len(boosted[0].matched_skills) == 4

    assert boosted[1].job_id == "job_high_base"
    assert boosted[1].score == 0.80
    assert boosted[1].skill_boost == 0.0
    assert len(boosted[1].matched_skills) == 0


def test_filter_fallback_tier_1_strict() -> None:
    """Test Tier 1 fallback: when sufficient strict matches exist, returns only strict matches."""
    results = [
        SearchResult(
            job_id="1", score=0.9, location="Austin, TX", experience_level="Entry level"
        ),
        SearchResult(
            job_id="2", score=0.8, location="Austin, TX", experience_level="Entry level"
        ),
        SearchResult(
            job_id="3", score=0.7, location="New York, NY", experience_level="Entry level"
        ),
    ]

    filtered, tier = filter_with_fallback(
        results, preferred_location="Austin, TX", seniority="Entry level", k=2
    )

    assert tier == "tier_1_strict"
    assert len(filtered) == 2
    assert {r.job_id for r in filtered} == {"1", "2"}


def test_filter_fallback_tier_2_relaxed_seniority() -> None:
    """Test Tier 2 fallback: relaxes seniority when strict match has < k candidates."""
    results = [
        # Only 1 strictly matching entry level in Austin
        SearchResult(
            job_id="1", score=0.9, location="Austin, TX", experience_level="Entry level"
        ),
        # 1 mid-senior in Austin (location compatible)
        SearchResult(
            job_id="2", score=0.85, location="Austin, TX", experience_level="Mid-Senior level"
        ),
        # 1 in New York
        SearchResult(
            job_id="3", score=0.7, location="New York, NY", experience_level="Entry level"
        ),
    ]

    # Asking for k=2 with Austin, TX and Entry level
    filtered, tier = filter_with_fallback(
        results, preferred_location="Austin, TX", seniority="Entry level", k=2
    )

    assert tier == "tier_2_relaxed_seniority"
    assert len(filtered) == 2
    assert [r.job_id for r in filtered] == ["1", "2"]


def test_filter_fallback_tier_3_global_backfill() -> None:
    """Test Tier 3 fallback: backfills from global pool when location constraint yields < k."""
    results = [
        # Only 1 posting in Seattle
        SearchResult(
            job_id="1", score=0.9, location="Seattle, WA", experience_level="Mid-Senior level"
        ),
        # Other postings elsewhere
        SearchResult(
            job_id="2", score=0.85, location="San Francisco, CA", experience_level="Mid-Senior level"
        ),
        SearchResult(
            job_id="3", score=0.80, location="Boston, MA", experience_level="Mid-Senior level"
        ),
    ]

    # Asking for k=3 with Seattle, WA
    filtered, tier = filter_with_fallback(
        results, preferred_location="Seattle, WA", seniority="Mid-Senior level", k=3
    )

    assert tier == "tier_3_global_backfill"
    assert len(filtered) == 3
    assert filtered[0].job_id == "1"  # Seattle job preserved at top
    assert {r.job_id for r in filtered} == {"1", "2", "3"}
