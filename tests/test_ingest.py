"""Unit tests for data ingestion, cleaning, deduplication, and normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest import (
    clean_dataframe,
    clean_text,
    normalize_experience_level,
    normalize_location,
    normalize_whitespace,
    sample_dataframe,
    strip_html,
)


def test_strip_html_basic_and_entities() -> None:
    """Test HTML tag removal and HTML entity unescaping."""
    raw_html = "<p>We are seeking a <strong>Machine Learning Engineer</strong> &amp; Data Scientist.</p>"
    expected = "We are seeking a Machine Learning Engineer & Data Scientist."
    cleaned = strip_html(raw_html)
    assert clean_text(cleaned) == expected


def test_strip_html_block_tags_boundary_spacing() -> None:
    """Test that block-level HTML tags insert space separation to prevent joined words."""
    # Adjacent list items
    html_bullets = "<ul><li>JavaScript</li><li>Understanding of software</li></ul>"
    assert clean_text(strip_html(html_bullets)) == "JavaScript Understanding of software"

    # Adjacent paragraphs without whitespace
    html_paragraphs = "<p>First paragraph</p><p>Second paragraph</p>"
    assert clean_text(strip_html(html_paragraphs)) == "First paragraph Second paragraph"

    # Line break without whitespace
    html_br = "LineOne<br>LineTwo<br/>LineThree"
    assert clean_text(strip_html(html_br)) == "LineOne LineTwo LineThree"

    # Table cells
    html_table = "<tr><td>Tableau</td><td>Experience</td></tr>"
    assert clean_text(strip_html(html_table)) == "Tableau Experience"



def test_strip_html_none_and_empty() -> None:
    """Test HTML stripping handles None, empty string, and whitespace gracefully."""
    assert strip_html(None) == ""
    assert strip_html("") == ""
    assert strip_html("   ") == "   "


def test_normalize_whitespace() -> None:
    """Test collapsing tabs, newlines, and multiple spaces."""
    raw = "  Senior   Data \t\t Engineer \n\n with  Python   "
    expected = "Senior Data Engineer with Python"
    assert normalize_whitespace(raw) == expected
    assert normalize_whitespace(None) == ""
    assert normalize_whitespace("") == ""


def test_clean_text_combined() -> None:
    """Test end-to-end text cleaning combining HTML stripping and whitespace normalization."""
    raw = "<div> <h3>Role: ML Engineer</h3> \n<p>Experience with &quot;PyTorch&quot; &amp;   NLP.</p></div>"
    expected = 'Role: ML Engineer Experience with "PyTorch" & NLP.'
    assert clean_text(raw) == expected


def test_normalize_experience_level_direct_mappings() -> None:
    """Test canonical mapping of explicit raw experience levels."""
    assert normalize_experience_level("Internship") == "Internship"
    assert normalize_experience_level("Entry level") == "Entry level"
    assert normalize_experience_level("Associate") == "Entry level"
    assert normalize_experience_level("Mid-Senior level") == "Mid-Senior level"
    assert normalize_experience_level("Director") == "Director"
    assert normalize_experience_level("Executive") == "Executive"


def test_normalize_experience_level_inference_from_title() -> None:
    """Test fallback inference of seniority when raw level is missing or NaN."""
    assert (
        normalize_experience_level(None, title="Senior Backend Engineer")
        == "Mid-Senior level"
    )
    assert (
        normalize_experience_level("", title="Principal AI Researcher")
        == "Mid-Senior level"
    )
    assert (
        normalize_experience_level(None, title="Software Engineering Intern")
        == "Internship"
    )
    assert (
        normalize_experience_level("", title="Junior Frontend Developer")
        == "Entry level"
    )
    assert (
        normalize_experience_level(None, title="VP of Engineering")
        == "Director"
    )
    assert (
        normalize_experience_level("", title="Chief Technology Officer (CTO)")
        == "Executive"
    )
    assert (
        normalize_experience_level(None, title="Accountant")
        == "Not Specified"
    )


def test_normalize_location() -> None:
    """Test location normalization for cities, states, and remote positions."""
    assert normalize_location("Austin, TX, United States") == "Austin, TX"
    assert normalize_location("San Francisco, CA") == "San Francisco, CA"
    assert normalize_location("Remote - US Only") == "Remote"
    assert normalize_location("Fully Remote") == "Remote"
    assert normalize_location(None) == "Remote / Not Specified"
    assert normalize_location("") == "Remote / Not Specified"


def test_clean_dataframe_drops_empty_and_null_rows() -> None:
    """Test that rows with missing or whitespace-only title or description are dropped."""
    data = [
        {
            "job_id": "1",
            "company_name": "Acme",
            "title": "Data Scientist",
            "location": "Boston, MA",
            "description": "Build ML models.",
            "formatted_experience_level": "Entry level",
        },
        {
            "job_id": "2",
            "company_name": "Acme",
            "title": None,  # null title
            "location": "Boston, MA",
            "description": "Build ML models.",
            "formatted_experience_level": "Entry level",
        },
        {
            "job_id": "3",
            "company_name": "Acme",
            "title": "   ",  # whitespace only title
            "location": "Boston, MA",
            "description": "Build ML models.",
            "formatted_experience_level": "Entry level",
        },
        {
            "job_id": "4",
            "company_name": "Acme",
            "title": "Data Scientist",
            "location": "Boston, MA",
            "description": None,  # null description
            "formatted_experience_level": "Entry level",
        },
        {
            "job_id": "5",
            "company_name": "Acme",
            "title": "Data Scientist",
            "location": "Boston, MA",
            "description": "  \t \n ",  # whitespace only description
            "formatted_experience_level": "Entry level",
        },
    ]
    df_raw = pd.DataFrame(data)
    cleaned_df, counts = clean_dataframe(df_raw)

    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["id"] == "1"
    assert counts["raw_total"] == 5
    assert counts["after_dedupe"] == 1


def test_clean_dataframe_deduplication() -> None:
    """Test that identical postings are properly deduplicated."""
    data = [
        {
            "job_id": "101",
            "company_name": "TechCorp",
            "title": "ML Engineer",
            "location": "Remote",
            "description": "Work on LLM retrieval.",
            "formatted_experience_level": "Mid-Senior level",
        },
        {
            "job_id": "102",  # Duplicate posting under different ID
            "company_name": "TechCorp",
            "title": "ML Engineer",
            "location": "Remote",
            "description": "Work on LLM retrieval.",
            "formatted_experience_level": "Mid-Senior level",
        },
        {
            "job_id": "103",  # Different title
            "company_name": "TechCorp",
            "title": "Data Scientist",
            "location": "Remote",
            "description": "Work on computer vision.",
            "formatted_experience_level": "Mid-Senior level",
        },
        {
            "job_id": "104",  # Duplicate company + title with different location
            "company_name": "TechCorp",
            "title": "Data Scientist",
            "location": "New York, NY",
            "description": "Work on computer vision in NY office.",
            "formatted_experience_level": "Mid-Senior level",
        },
    ]
    df_raw = pd.DataFrame(data)
    cleaned_df, counts = clean_dataframe(df_raw)

    assert len(cleaned_df) == 2
    assert set(cleaned_df["id"].tolist()) == {"101", "103"}


def test_sample_dataframe_reproducibility() -> None:
    """Test deterministic sampling with fixed seed."""
    data = [
        {
            "id": str(i),
            "title": f"Job {i}",
            "company": "Company",
            "location": "Remote",
            "description": f"Description {i}",
            "experience_level": "Entry level",
        }
        for i in range(100)
    ]
    df = pd.DataFrame(data)

    sample_1 = sample_dataframe(df, n=10, seed=42)
    sample_2 = sample_dataframe(df, n=10, seed=42)
    sample_diff_seed = sample_dataframe(df, n=10, seed=999)

    assert len(sample_1) == 10
    pd.testing.assert_frame_equal(sample_1, sample_2)
    assert not sample_1["id"].equals(sample_diff_seed["id"])

