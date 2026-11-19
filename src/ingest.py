"""Data ingestion and preprocessing pipeline for job postings.

Loads raw job postings, cleans text (stripping HTML, normalising whitespace),
drops empty or invalid rows, removes duplicate postings, normalises metadata
(experience level and location), samples a reproducible subset, and records
comprehensive dataset statistics.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 3000
RAW_DATA_PATH = Path("data/raw_postings.csv")
CLEAN_DATA_PATH = Path("data/jobs_clean.parquet")
STATS_REPORT_PATH = Path("reports/data_stats.json")

DATASET_SOURCE_URL = (
    "https://huggingface.co/datasets/datastax/linkedin_job_listings"
)
DATASET_LICENCE = (
    "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) / MIT"
)
DATASET_NAME = "LinkedIn Job Postings (2023-2024)"

HTML_TAG_REGEX = re.compile(r"<[^>]+>")
WHITESPACE_REGEX = re.compile(r"\s+")

TITLE_INTERN_REGEX = re.compile(
    r"\b(intern|internship|trainee|apprentice|co-op)\b", re.IGNORECASE
)
TITLE_ENTRY_REGEX = re.compile(
    r"\b(junior|jr\.?|entry\s*level|associate|graduate)\b", re.IGNORECASE
)
TITLE_SENIOR_REGEX = re.compile(
    r"\b(senior|sr\.?|lead|staff|principal|manager|architect)\b", re.IGNORECASE
)
TITLE_DIRECTOR_REGEX = re.compile(
    r"\b(director|head of|vp|vice\s*president)\b", re.IGNORECASE
)
TITLE_EXECUTIVE_REGEX = re.compile(
    r"\b(chief|cxo|ceo|cto|cfo|coo|cmo|cro|executive)\b", re.IGNORECASE
)


def strip_html(text: Optional[str]) -> str:
    """Strip HTML markup and unescape HTML entities.

    Args:
        text: Raw text string containing optional HTML markup.

    Returns:
        Clean plain text without HTML tags or unescaped entities.
    """
    if not text or not isinstance(text, str):
        return ""
    unescaped = html.unescape(text)
    clean_text = HTML_TAG_REGEX.sub(" ", unescaped)
    return clean_text


def normalize_whitespace(text: Optional[str]) -> str:
    """Normalize irregular whitespace, line breaks, and tabs into single spaces.

    Args:
        text: Input text string.

    Returns:
        Whitespace-normalized and trimmed string.
    """
    if not text or not isinstance(text, str):
        return ""
    return WHITESPACE_REGEX.sub(" ", text).strip()


def clean_text(text: Optional[str]) -> str:
    """Clean text by stripping HTML tags and normalizing all whitespace.

    Args:
        text: Input string possibly with HTML tags or uneven whitespace.

    Returns:
        Fully cleaned plain text string.
    """
    return normalize_whitespace(strip_html(text))


def normalize_experience_level(
    raw_level: Optional[str], title: Optional[str] = None
) -> str:
    """Normalize raw experience level into standardized seniority categories.

    Maps explicit LinkedIn levels to canonical categories. If the raw level
    is missing or empty, attempts to infer seniority from the job title.

    Args:
        raw_level: Raw experience level string from the dataset.
        title: Optional job title for heuristic fallback classification.

    Returns:
        Standardized seniority category string.
    """
    if raw_level and isinstance(raw_level, str):
        cleaned_lvl = raw_level.strip().lower()
        if "intern" in cleaned_lvl:
            return "Internship"
        if "entry" in cleaned_lvl or "associate" in cleaned_lvl:
            return "Entry level"
        if "mid" in cleaned_lvl or "senior" in cleaned_lvl:
            return "Mid-Senior level"
        if "director" in cleaned_lvl:
            return "Director"
        if "executive" in cleaned_lvl:
            return "Executive"

    if title and isinstance(title, str):
        if TITLE_INTERN_REGEX.search(title):
            return "Internship"
        if TITLE_ENTRY_REGEX.search(title):
            return "Entry level"
        if TITLE_EXECUTIVE_REGEX.search(title):
            return "Executive"
        if TITLE_DIRECTOR_REGEX.search(title):
            return "Director"
        if TITLE_SENIOR_REGEX.search(title):
            return "Mid-Senior level"

    return "Not Specified"


def normalize_location(location_raw: Optional[str]) -> str:
    """Normalize job location string into standardized city/state or remote labels.

    Args:
        location_raw: Raw location string from dataset.

    Returns:
        Standardized location string.
    """
    if not location_raw or not isinstance(location_raw, str):
        return "Remote / Not Specified"
    cleaned = normalize_whitespace(location_raw)
    if not cleaned:
        return "Remote / Not Specified"

    if re.search(r"\bremote\b", cleaned, re.IGNORECASE):
        return "Remote"

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) >= 2:
        return f"{parts[0]}, {parts[1]}"
    return cleaned


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean and standardize the raw job postings dataframe.

    Strips HTML, normalizes text, drops empty titles/descriptions, deduplicates,
    and constructs standardized metadata fields.

    Args:
        df: Input raw pandas DataFrame.

    Returns:
        A tuple of (cleaned DataFrame, dictionary of intermediate row counts).
    """
    counts: Dict[str, int] = {
        "raw_total": len(df),
    }

    # Ensure required columns exist
    for col in ["title", "description", "company_name", "location", "job_id"]:
        if col not in df.columns:
            df[col] = ""

    # Drop explicit nulls in critical fields
    df = df.dropna(subset=["title", "description"]).copy()
    counts["after_dropna"] = len(df)

    # Clean text columns
    df["title_clean"] = df["title"].astype(str).apply(clean_text)
    df["description_clean"] = df["description"].astype(str).apply(clean_text)

    # Filter out records where cleaned text is empty
    valid_mask = (df["title_clean"] != "") & (df["description_clean"] != "")
    df = df[valid_mask].copy()
    counts["after_non_empty_text"] = len(df)

    # Clean company and location
    df["company_clean"] = (
        df["company_name"]
        .fillna("Unknown Company")
        .astype(str)
        .apply(normalize_whitespace)
    )
    df["company_clean"] = df["company_clean"].replace("", "Unknown Company")

    raw_level_col = "formatted_experience_level"
    if raw_level_col not in df.columns:
        df[raw_level_col] = ""

    # Normalize metadata fields
    df["experience_level_norm"] = [
        normalize_experience_level(raw_lvl, t)
        for raw_lvl, t in zip(df[raw_level_col], df["title_clean"])
    ]
    df["location_norm"] = df["location"].astype(str).apply(normalize_location)

    # Deduplicate on (company_clean, title_clean) and (company_clean, description_clean)
    # to eliminate duplicate postings and identical job templates from the same company
    df = df.drop_duplicates(
        subset=["company_clean", "title_clean"],
        keep="first",
    ).copy()
    df = df.drop_duplicates(
        subset=["company_clean", "description_clean"],
        keep="first",
    ).copy()
    counts["after_dedupe"] = len(df)

    # Rename and select final schema
    df["id"] = df["job_id"].astype(str)
    df["title"] = df["title_clean"]
    df["company"] = df["company_clean"]
    df["location"] = df["location_norm"]
    df["description"] = df["description_clean"]
    df["experience_level"] = df["experience_level_norm"]

    final_columns = [
        "id",
        "title",
        "company",
        "location",
        "description",
        "experience_level",
    ]
    cleaned_df = df[final_columns].reset_index(drop=True)
    return cleaned_df, counts


def sample_dataframe(
    df: pd.DataFrame, n: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Sample a reproducible subset of rows from the cleaned dataframe.

    Args:
        df: Input cleaned dataframe.
        n: Number of rows to sample. If larger than len(df), returns all rows.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Sampled DataFrame with reset index.
    """
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def compute_statistics(
    raw_count: int,
    cleaned_count: int,
    df_sampled: pd.DataFrame,
    source_url: str = DATASET_SOURCE_URL,
    licence_string: str = DATASET_LICENCE,
) -> Dict[str, Any]:
    """Compute summary statistics for the dataset report.

    Args:
        raw_count: Total raw row count before filtering.
        cleaned_count: Total valid row count after deduplication and cleaning.
        df_sampled: Final sampled DataFrame.
        source_url: Upstream dataset repository URL.
        licence_string: Dataset license description.

    Returns:
        Dictionary containing all metrics for data_stats.json.
    """
    level_counts = df_sampled["experience_level"].value_counts().to_dict()
    total_sampled = len(df_sampled)
    level_dist = {
        lvl: {
            "count": int(count),
            "percentage": round((count / total_sampled) * 100, 2),
        }
        for lvl, count in level_counts.items()
    }

    location_counts = df_sampled["location"].value_counts().head(20).to_dict()
    location_dist = {
        loc: {
            "count": int(count),
            "percentage": round((count / total_sampled) * 100, 2),
        }
        for loc, count in location_counts.items()
    }

    desc_lengths_chars = df_sampled["description"].str.len()
    desc_lengths_words = df_sampled["description"].apply(
        lambda x: len(x.split())
    )

    stats: Dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "source_url": source_url,
        "licence": licence_string,
        "row_counts": {
            "raw_postings_pool": int(raw_count),
            "clean_postings_available": int(cleaned_count),
            "final_sampled_postings": int(total_sampled),
        },
        "description_length": {
            "avg_characters": round(float(desc_lengths_chars.mean()), 2),
            "median_characters": int(desc_lengths_chars.median()),
            "avg_words": round(float(desc_lengths_words.mean()), 2),
            "median_words": int(desc_lengths_words.median()),
            "min_words": int(desc_lengths_words.min()),
            "max_words": int(desc_lengths_words.max()),
        },
        "experience_level_distribution": level_dist,
        "location_distribution_top20": location_dist,
    }
    return stats


def run_ingest(
    raw_path: Path = RAW_DATA_PATH,
    clean_path: Path = CLEAN_DATA_PATH,
    stats_path: Path = STATS_REPORT_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Execute the full data ingestion, cleaning, sampling, and reporting pipeline.

    Args:
        raw_path: Path to the raw CSV dataset.
        clean_path: Destination path for the clean Parquet output.
        stats_path: Destination path for the JSON summary report.
        sample_size: Target sample size (between 2000 and 5000).
        seed: Random seed for sampling.

    Returns:
        Final cleaned and sampled DataFrame.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at {raw_path}. "
            "Ensure raw postings are downloaded before running ingestion."
        )

    print(f"Loading raw data from {raw_path}...")
    df_raw = pd.read_csv(raw_path, low_memory=False)
    raw_count = len(df_raw)

    print(f"Cleaning {raw_count} postings...")
    df_clean, intermediate_counts = clean_dataframe(df_raw)
    cleaned_count = len(df_clean)
    print(
        f"Postings after cleaning & deduplication: {cleaned_count} "
        f"(dropped {raw_count - cleaned_count})"
    )

    print(f"Sampling {sample_size} postings with seed={seed}...")
    df_sampled = sample_dataframe(df_clean, n=sample_size, seed=seed)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving clean parquet to {clean_path}...")
    df_sampled.to_parquet(clean_path, index=False)

    print(f"Computing statistics and writing to {stats_path}...")
    stats = compute_statistics(raw_count, cleaned_count, df_sampled)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Ingestion complete.")
    return df_sampled


if __name__ == "__main__":
    run_ingest()

