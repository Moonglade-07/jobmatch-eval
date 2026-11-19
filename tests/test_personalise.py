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
    """Test that min-max normalized skill boost changes ordering when candidate matches skills."""
    candidate_skills = ["Python", "PyTorch", "Docker", "Kubernetes"]

    # Candidate 1: Higher base score (80.0), but 0 matched skills.
    # Candidate 2: Slightly lower base score (75.0), but matches all 4 skills.
    # Candidate 3: Baseline bottom candidate (50.0).
    r1 = SearchResult(
        job_id="job_high_base",
        score=80.0,
        title="Software Engineer",
        description="Work with Ruby and Rails on AWS.",
    )
    r2 = SearchResult(
        job_id="job_high_skills",
        score=75.0,
        title="MLOps Engineer",
        description="Requires Python, PyTorch, Docker, and Kubernetes deployment.",
    )
    r3 = SearchResult(
        job_id="job_low_base",
        score=50.0,
        title="Database Admin",
        description="SQL and database maintenance.",
    )

    # Min-max normalization over [50.0, 80.0] (range = 30.0):
    # r1 norm = (80 - 50) / 30 = 1.0000; boost = 0.2 * (0/4) = 0.0 -> final = 1.0000
    # r2 norm = (75 - 50) / 30 = 0.8333; boost = 0.2 * (4/4) = 0.2 -> final = 1.0333
    # r3 norm = (50 - 50) / 30 = 0.0000; boost = 0.0 -> final = 0.0000
    # Result: r2 must overtake r1!
    boosted = apply_skill_boost([r1, r2, r3], skills=candidate_skills, weight=0.2)

    assert boosted[0].job_id == "job_high_skills"
    assert boosted[0].score > boosted[1].score
    assert boosted[0].skill_boost == 0.2
    assert len(boosted[0].matched_skills) == 4

    assert boosted[1].job_id == "job_high_base"
    assert boosted[1].score == 1.0
    assert boosted[1].skill_boost == 0.0


def test_matches_location_remote_strictly_requires_remote_job() -> None:
    """Test that preferred_location='Remote' strictly excludes onsite postings and accepts remote postings."""
    onsite = SearchResult(
        job_id="onsite",
        score=100.0,
        title="Data Scientist",
        location="Dallas, TX",
        description="Must work onsite at our Plano, TX facility 5 days a week.",
    )
    remote_desc = SearchResult(
        job_id="remote_desc",
        score=90.0,
        title="ML Engineer",
        location="United States",
        description="100% remote work opportunity with flexible hours.",
    )
    remote_title = SearchResult(
        job_id="remote_title",
        score=80.0,
        title="Remote Software Engineer",
        location="Chicago, IL",
        description="Build Python web services.",
    )

    filtered, tier = filter_with_fallback(
        [onsite, remote_desc, remote_title],
        preferred_location="Remote",
        seniority="",
        k=2,
    )

    assert tier == "tier_1_strict"
    assert len(filtered) == 2
    assert "onsite" not in [r.job_id for r in filtered]
    assert {r.job_id for r in filtered} == {"remote_desc", "remote_title"}
    assert all(r.filter_tier == 1 for r in filtered)


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
    assert all(r.filter_tier == 1 for r in filtered)


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
    assert filtered[0].filter_tier == 1
    assert filtered[1].filter_tier == 2


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
    assert filtered[0].filter_tier == 1
    assert filtered[1].filter_tier == 3
    assert filtered[2].filter_tier == 3
    assert {r.job_id for r in filtered} == {"1", "2", "3"}

