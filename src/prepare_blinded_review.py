"""Prepare blinded human evaluation dataset and decoupled answer key.

Generates:
  1. labels_to_review.csv: Blinded review file containing the union of the 260
     random must_review sample and all proposed_label==2 rows (de-duplicated),
     shuffled within each profile, with ~600 character description snippets and empty my_label.
     Contains NO proposed_label and NO must_review columns.
  2. data/eval/review_key.csv (gitignored): Ground-truth answer key recording
     proposed_label, must_review, and is_label2_row matching row-for-row with the review file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DEFAULT_LABELS_PROPOSED_PATH = Path("labels_proposed.csv")
DEFAULT_JOBS_CLEAN_PATH = Path("data/jobs_clean.parquet")
DEFAULT_REVIEW_CSV_PATH = Path("labels_to_review.csv")
DEFAULT_REVIEW_KEY_PATH = Path("data/eval/review_key.csv")
SHUFFLE_SEED = 42
TARGET_SNIPPET_LENGTH = 600


def extract_rich_snippet(text: str, max_chars: int = TARGET_SNIPPET_LENGTH) -> str:
    """Extract a clean, informative description snippet of approximately 600 characters.

    Args:
        text: Raw cleaned description text.
        max_chars: Target character limit.

    Returns:
        Formatted snippet collapsing excessive whitespace.
    """
    if not text or not isinstance(text, str):
        return ""

    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean

    # Truncate at nearest sentence or word boundary around max_chars
    boundary = clean[:max_chars].rfind(". ")
    if boundary > max_chars * 0.70:
        return f"{clean[:boundary + 1].strip()}..."

    return f"{clean[:max_chars].rstrip()}..."


def build_blinded_review_files(
    proposed_csv_path: Path = DEFAULT_LABELS_PROPOSED_PATH,
    clean_data_path: Path = DEFAULT_JOBS_CLEAN_PATH,
    review_output_path: Path = DEFAULT_REVIEW_CSV_PATH,
    key_output_path: Path = DEFAULT_REVIEW_KEY_PATH,
    seed: int = SHUFFLE_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate blinded review CSV and decoupled answer key.

    Args:
        proposed_csv_path: Source proposed labels CSV.
        clean_data_path: Source clean job postings parquet for full text snippets.
        review_output_path: Path to write labels_to_review.csv.
        key_output_path: Path to write review_key.csv.
        seed: Random seed for intra-profile row shuffling.

    Returns:
        Tuple of (review DataFrame, answer key DataFrame).
    """
    if not proposed_csv_path.exists():
        raise FileNotFoundError(f"Proposed labels file not found at {proposed_csv_path}")
    if not clean_data_path.exists():
        raise FileNotFoundError(f"Clean postings dataset not found at {clean_data_path}")

    print(f"Loading proposed candidate pool from {proposed_csv_path}...")
    df_prop = pd.read_csv(proposed_csv_path)

    print(f"Loading job descriptions from {clean_data_path}...")
    df_jobs = pd.read_parquet(clean_data_path)
    job_desc_map: Dict[str, str] = {
        str(row["id"]): str(row["description"]) for _, row in df_jobs.iterrows()
    }

    # Filter condition: union of 260 must_review rows AND all proposed_label == 2 rows
    must_review_mask = df_prop["must_review"].astype(str).str.lower() == "true"
    label2_mask = df_prop["proposed_label"].astype(int) == 2
    target_mask = must_review_mask | label2_mask

    selected_df = df_prop[target_mask].copy()
    print(
        f"Filtered target candidates: {len(selected_df)} rows "
        f"({must_review_mask.sum()} must_review + {label2_mask.sum()} label-2 rows, union de-duplicated)."
    )

    # Attach rich ~600-character description snippets from jobs dataset
    snippets: List[str] = []
    for _, row in selected_df.iterrows():
        jid = str(row["job_id"])
        full_desc = job_desc_map.get(jid, str(row.get("key snippet of description", "")))
        snippets.append(extract_rich_snippet(full_desc, max_chars=TARGET_SNIPPET_LENGTH))

    selected_df["description snippet"] = snippets
    selected_df["is_label2_row"] = (selected_df["proposed_label"].astype(int) == 2)
    selected_df["must_review_bool"] = (selected_df["must_review"].astype(str).str.lower() == "true")

    # Shuffle rows within each profile to blind label proposals
    rng = np.random.RandomState(seed)
    shuffled_groups: List[pd.DataFrame] = []

    profile_ids = sorted(selected_df["profile_id"].unique(), key=lambda x: int(x.split("_")[1]))
    for prof_id in profile_ids:
        group = selected_df[selected_df["profile_id"] == prof_id].copy()
        # Randomly permute the rows within this profile
        permuted_indices = rng.permutation(len(group))
        shuffled_group = group.iloc[permuted_indices].reset_index(drop=True)
        shuffled_groups.append(shuffled_group)

    final_merged = pd.concat(shuffled_groups, ignore_index=True)
    final_merged["my_label"] = ""  # Strictly empty for human evaluation

    # 1. Blinded review file (NO proposed_label and NO must_review columns)
    review_cols = [
        "profile_id",
        "job_id",
        "title",
        "company",
        "description snippet",
        "my_label",
    ]
    df_review = final_merged[review_cols].copy()

    # 2. Answer key file (matching row-for-row)
    key_cols = [
        "profile_id",
        "job_id",
        "title",
        "company",
        "proposed_label",
        "must_review_bool",
        "is_label2_row",
    ]
    df_key = final_merged[key_cols].copy()
    df_key.rename(columns={"must_review_bool": "must_review"}, inplace=True)

    # Save review file to workspace root (and optional eval copy)
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_review.to_csv(review_output_path, index=False, encoding="utf-8")
    print(f"Saved blinded review file to {review_output_path} ({len(df_review)} rows).")

    # Save answer key to gitignored location
    key_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_key.to_csv(key_output_path, index=False, encoding="utf-8")
    print(f"Saved decoupled answer key to {key_output_path} ({len(df_key)} rows).")

    return df_review, df_key


if __name__ == "__main__":
    build_blinded_review_files()
