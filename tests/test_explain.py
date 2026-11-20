"""Unit tests for LLM explanation generation, groundedness metrics, caching, and fallback policies."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.explain import (
    Explanation,
    ExplainResult,
    GeminiExplainer,
    evaluate_claim_groundedness,
    evaluate_missing_skills_leakage,
    evaluate_profile_support,
    is_skill_grounded_regex,
    is_skill_grounded_strict,
)


# ---------------------------------------------------------------------------
# 1. Schema Validation Tests
# ---------------------------------------------------------------------------
def test_valid_json_passes_validation() -> None:
    """Verify that a compliant payload instantiates valid Explanation."""
    payload = {
        "matched_skills": ["Python", "Docker"],
        "missing_skills": ["Kubernetes"],
        "summary": "Candidate matches primary backend scripting and containerization requirements.",
    }
    explanation = Explanation(**payload)
    assert explanation.matched_skills == ["Python", "Docker"]
    assert explanation.missing_skills == ["Kubernetes"]
    assert "Candidate matches" in explanation.summary


def test_summary_multiple_sentences_rejected() -> None:
    """Verify that a summary with more than one sentence raises ValidationError."""
    payload = {
        "matched_skills": ["Python"],
        "missing_skills": [],
        "summary": "Candidate matches the Python requirement. However, candidate lacks Kubernetes experience.",
    }
    with pytest.raises(ValidationError) as exc:
        Explanation(**payload)
    assert "Summary must be exactly one sentence" in str(exc.value)


def test_summary_empty_rejected() -> None:
    """Verify that empty summary raises ValidationError."""
    with pytest.raises(ValidationError):
        Explanation(matched_skills=[], missing_skills=[], summary="   ")


# ---------------------------------------------------------------------------
# 2. Groundedness & Substring Boundary Tests
# ---------------------------------------------------------------------------
def test_groundedness_symbol_languages_and_case_insensitivity() -> None:
    """Verify boundary-safe matching for C++, C#, .NET, and casing."""
    text = "We are seeking a senior engineer with deep C++ and C# expertise building on the .NET framework."

    # Direct matches with symbols
    assert is_skill_grounded_regex("C++", text) is True
    assert is_skill_grounded_regex("c++", text) is True
    assert is_skill_grounded_regex("C#", text) is True
    assert is_skill_grounded_regex("c#", text) is True
    assert is_skill_grounded_regex(".NET", text) is True
    assert is_skill_grounded_regex(".net", text) is True

    # Strict match is case-sensitive
    assert is_skill_grounded_strict("C++", text) is True
    assert is_skill_grounded_strict("c++", text) is False
    assert is_skill_grounded_strict(".NET", text) is True
    assert is_skill_grounded_strict(".net", text) is False


def test_groundedness_substring_false_positives_prevented() -> None:
    """Ensure partial substrings inside longer words are NOT counted as matches."""
    text = "Deploying enterprise Cloud solutions using JavaScript and Rust-based toolchains."

    # 'C' inside 'Cloud' must NOT match
    assert is_skill_grounded_regex("C", text) is False

    # 'Java' inside 'JavaScript' must NOT match
    assert is_skill_grounded_regex("Java", text) is False

    # 'Script' inside 'JavaScript' must NOT match
    assert is_skill_grounded_regex("Script", text) is False

    # Full word matches
    assert is_skill_grounded_regex("Cloud", text) is True
    assert is_skill_grounded_regex("JavaScript", text) is True
    assert is_skill_grounded_regex("Rust", text) is True


def test_evaluate_claim_groundedness_hand_computed() -> None:
    """Hand-verify groundedness ratios on controlled text."""
    job_text = "Role requires Python, Docker, and PostgreSQL. Nice to have: AWS."
    claimed = ["Python", "docker", "Kubernetes", "AWS"]

    ratio, strict_ratio, details = evaluate_claim_groundedness(claimed, job_text)

    # Grounded regex: Python (yes), docker (yes - case insensitive), Kubernetes (no), AWS (yes) -> 3/4 = 0.75
    assert ratio == 0.75
    # Strict grounded: Python (yes), docker (no - text has Docker), Kubernetes (no), AWS (yes) -> 2/4 = 0.50
    assert strict_ratio == 0.50
    assert len(details) == 4


def test_missing_skills_leakage_contradiction() -> None:
    """Verify that a skill present in BOTH job and profile flags as contradiction leakage."""
    job_text = "Seeking Python and Docker engineer."
    profile_text = "Experienced in Python, Docker, and Linux."

    # Model claims Docker is missing, but candidate HAS it and job REQUIRES it
    missing_claimed = ["Docker", "Kubernetes"]
    leakage = evaluate_missing_skills_leakage(missing_claimed, job_text, profile_text)

    # Docker is in both (leakage = 1), Kubernetes is not in profile/job (no leakage) -> 1/2 = 0.5
    assert leakage == 0.5


# ---------------------------------------------------------------------------
# 3. Retry and Fallback Logic (Mocked API)
# ---------------------------------------------------------------------------
def test_invalid_json_triggers_exactly_one_retry_and_recovers(tmp_path: Path) -> None:
    """Verify that malformed initial JSON prompts a repair call and succeeds if repaired."""
    explainer = GeminiExplainer(
        api_key="mock_key",
        model_name="mock-model",
        cache_dir=tmp_path,
        request_delay=0.0,
    )

    invalid_resp = {
        "candidates": [{"content": {"parts": [{"text": "Not valid JSON at all"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }
    repaired_resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "matched_skills": ["Python"],
                                "missing_skills": ["Go"],
                                "summary": "Candidate matches required Python skills for this role.",
                            })
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 20, "totalTokenCount": 35},
    }

    with patch.object(explainer, "_send_generate_content") as mock_send:
        # 1st call invalid, 2nd call (repair) valid
        mock_send.side_effect = [
            (200, invalid_resp, 100.0, None),
            (200, repaired_resp, 120.0, None),
        ]

        res = explainer.explain(
            profile_data={"profile_id": "p1", "target_role": "Developer", "skills": ["Python"]},
            job_data={"job_id": "j1", "title": "Python Dev", "description": "Need Python"},
            use_cache=False,
        )

        assert mock_send.call_count == 2
        assert res.fallback is False
        assert res.retry_count == 1
        assert res.explanation is not None
        assert res.explanation.matched_skills == ["Python"]


def test_two_failures_trigger_fallback_with_explanation_null(tmp_path: Path) -> None:
    """Verify that failure on initial AND repair prompt returns fallback with explanation=None."""
    explainer = GeminiExplainer(
        api_key="mock_key",
        model_name="mock-model",
        cache_dir=tmp_path,
        request_delay=0.0,
    )

    invalid_resp = {
        "candidates": [{"content": {"parts": [{"text": "{bad_json: True"}]}}],
    }

    with patch.object(explainer, "_send_generate_content") as mock_send:
        mock_send.side_effect = [
            (200, invalid_resp, 100.0, None),
            (200, invalid_resp, 100.0, None),
        ]

        res = explainer.explain(
            profile_data={"profile_id": "p1", "target_role": "Developer", "skills": ["Python"]},
            job_data={"job_id": "j1", "title": "Dev", "description": "Desc"},
            use_cache=False,
        )

        assert mock_send.call_count == 2
        assert res.fallback is True
        assert res.explanation is None
        assert res.fallback_reason == "invalid_json"


def test_api_error_and_timeout_fallback_reasons(tmp_path: Path) -> None:
    """Verify that HTTP errors and timeouts produce fallbacks with precise reasons."""
    explainer = GeminiExplainer(
        api_key="mock_key",
        model_name="mock-model",
        cache_dir=tmp_path,
        request_delay=0.0,
    )

    # Case 1: Timeout
    with patch.object(explainer, "_send_generate_content", return_value=(408, None, 50.0, "timeout")):
        res_timeout = explainer.explain(
            profile_data={"profile_id": "p1"},
            job_data={"job_id": "j1"},
            use_cache=False,
        )
        assert res_timeout.fallback is True
        assert res_timeout.fallback_reason == "timeout"

    # Case 2: Quota
    with patch.object(
        explainer,
        "_send_generate_content",
        return_value=(429, None, 30.0, "quota_daily_exhausted"),
    ):
        res_quota = explainer.explain(
            profile_data={"profile_id": "p1"},
            job_data={"job_id": "j1"},
            use_cache=False,
        )
        assert res_quota.fallback is True
        assert res_quota.fallback_reason == "quota"


# ---------------------------------------------------------------------------
# 4. Cache Tests
# ---------------------------------------------------------------------------
def test_cache_hit_makes_no_api_calls(tmp_path: Path) -> None:
    """Verify that cached result is read from disk with zero external API calls."""
    explainer = GeminiExplainer(
        api_key="mock_key",
        model_name="mock-model",
        cache_dir=tmp_path,
        request_delay=0.0,
    )

    valid_payload = {
        "matched_skills": ["SQL"],
        "missing_skills": [],
        "summary": "Candidate is a great fit for the SQL analyst role.",
    }
    resp = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(valid_payload)}]}}],
        "usageMetadata": {"totalTokenCount": 50},
    }

    with patch.object(explainer, "_send_generate_content", return_value=(200, resp, 150.0, None)) as mock_send:
        # Call 1: Miss -> Makes API call and writes cache
        res1 = explainer.explain(
            profile_data={"profile_id": "prof_cached", "skills": ["SQL"]},
            job_data={"job_id": "job_cached", "title": "SQL Analyst"},
            use_cache=True,
        )
        assert mock_send.call_count == 1
        assert res1.cache_hit is False

        # Call 2: Hit -> Reads from disk, 0 API calls
        res2 = explainer.explain(
            profile_data={"profile_id": "prof_cached", "skills": ["SQL"]},
            job_data={"job_id": "job_cached", "title": "SQL Analyst"},
            use_cache=True,
        )
        assert mock_send.call_count == 1  # count has not increased
        assert res2.cache_hit is True
        assert res2.explanation is not None
        assert res2.explanation.matched_skills == ["SQL"]


def test_bootstrap_ci_zero_variance_returns_none() -> None:
    """Verify that constant arrays (zero variance) or small n return None and format as 'n too small'."""
    from src.explain import compute_bootstrap_profile_ci, format_ci

    assert compute_bootstrap_profile_ci([1.0, 1.0, 1.0]) is None
    assert format_ci(compute_bootstrap_profile_ci([1.0, 1.0, 1.0])) == "n too small"
    assert compute_bootstrap_profile_ci([0.5]) is None
    assert format_ci(None) == "n too small"

    # Variance exists
    ci = compute_bootstrap_profile_ci([0.1, 0.9, 0.5, 0.8])
    assert ci is not None
    assert len(ci) == 2
    assert ci[0] <= ci[1]
    assert "n too small" not in format_ci(ci)


def test_validity_after_retry_and_retries_needed() -> None:
    """Verify that json_valid_after_retry_rate is >= first_try_rate and retries_needed is tracked."""
    from src.explain import ExplainResult, Explanation, evaluate_explanations

    results = [
        # Call 1: valid first try
        ExplainResult(
            profile_id="p1",
            job_id="j1",
            explanation=Explanation(matched_skills=["Python"], missing_skills=[], summary="Match."),
            retry_count=0,
            fallback=False,
        ),
        # Call 2: repaired on retry
        ExplainResult(
            profile_id="p1",
            job_id="j2",
            explanation=Explanation(matched_skills=["SQL"], missing_skills=[], summary="Match."),
            retry_count=1,
            fallback=False,
        ),
        # Call 3: fallback
        ExplainResult(
            profile_id="p2",
            job_id="j3",
            explanation=None,
            retry_count=1,
            fallback=True,
            fallback_reason="api_error",
        ),
    ]

    metrics = evaluate_explanations(results, {"p1": {}, "p2": {}}, {"j1": {}, "j2": {}, "j3": {}})
    val = metrics["validity"]

    assert val["json_valid_first_try_rate"] == round(1 / 3, 4)
    assert val["retries_needed_count"] == 2  # Call 2 and Call 3 both needed retries
    assert val["json_valid_after_retry_rate"] == round(2 / 3, 4)
    assert val["json_valid_after_retry_rate"] >= val["json_valid_first_try_rate"]
    assert val["fallback_rate"] == round(1 / 3, 4)

