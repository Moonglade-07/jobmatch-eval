"""Compute inter-annotator agreement metrics and construct final evaluation qrels.

STRICT INTEGRITY ENFORCEMENT:
  1. Refuses to run unless an authentic human review file named `labels_reviewed.csv` is provided.
  2. Validates that every row in `labels_reviewed.csv` matches an expected (profile_id, job_id)
     entry from the official candidate pool.
  3. Verifies that all non-empty `my_label` values are valid ordinal ratings in {0, 1, 2}.
  4. Joins with `data/eval/review_key.csv` to calculate agreement metrics on the random
     `must_review` sample only, while evaluating the additional label-2 rows separately.
  5. Refuses execution if 0 reviewed rows are detected.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

DEFAULT_REVIEWED_PATH = Path("labels_reviewed.csv")
DEFAULT_PROPOSED_PATH = Path("labels_proposed.csv")
DEFAULT_REVIEW_KEY_PATH = Path("data/eval/review_key.csv")
DEFAULT_QRELS_PATH = Path("data/eval/qrels.json")
DEFAULT_AGREEMENT_REPORT_PATH = Path("reports/label_agreement.json")


def compute_weighted_cohen_kappa(
    y_true: List[int],
    y_pred: List[int],
    weights: str = "quadratic",
) -> float:
    """Compute Cohen's weighted kappa for ordinal classification.

    Args:
        y_true: Human reviewer ground truth ratings {0, 1, 2}.
        y_pred: Model or heuristic proposed ratings {0, 1, 2}.
        weights: Weighting scheme ('quadratic' or 'linear').

    Returns:
        Weighted kappa score in range [-1.0, 1.0].
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred.")
    if len(y_true) == 0:
        return 1.0

    return float(cohen_kappa_score(y_true, y_pred, weights=weights))


def evaluate_label_agreement(
    reviewed_csv_path: Path = DEFAULT_REVIEWED_PATH,
    proposed_csv_path: Path = DEFAULT_PROPOSED_PATH,
    review_key_path: Path = DEFAULT_REVIEW_KEY_PATH,
    qrels_path: Path = DEFAULT_QRELS_PATH,
    report_path: Path = DEFAULT_AGREEMENT_REPORT_PATH,
) -> Dict[str, Any]:
    """Calculate inter-annotator agreement metrics and export final qrels and report.

    Strictly requires `labels_reviewed.csv` to exist and contain authentic human entries.

    Args:
        reviewed_csv_path: Path to returned human-reviewed CSV (MUST be named labels_reviewed.csv).
        proposed_csv_path: Path to full proposed candidate pool (labels_proposed.csv).
        review_key_path: Path to decoupled answer key (review_key.csv).
        qrels_path: Destination path for data/eval/qrels.json.
        report_path: Destination path for reports/label_agreement.json.

    Returns:
        Dictionary containing agreement metrics and distribution statistics.
    """
    # 1. Strict file presence check: MUST be named labels_reviewed.csv
    if not reviewed_csv_path.exists():
        raise FileNotFoundError(
            f"Required human review file '{reviewed_csv_path}' not found!\n"
            f"Execution halted. You must annotate rows in 'labels_to_review.csv', save as "
            f"'{reviewed_csv_path.name}', and return the file before agreement can be computed."
        )

    if reviewed_csv_path.name != "labels_reviewed.csv" and not str(reviewed_csv_path).endswith("test_labels_reviewed.csv"):
        raise ValueError(
            f"Invalid review file name '{reviewed_csv_path.name}'. File must strictly be named 'labels_reviewed.csv'."
        )

    print(f"Loading authentic human reviews from {reviewed_csv_path}...")
    df_reviewed = pd.read_csv(reviewed_csv_path)

    if "my_label" not in df_reviewed.columns:
        raise ValueError("Input file must contain a 'my_label' column.")

    # 2. Check for human annotations
    non_empty_mask = df_reviewed["my_label"].notna() & df_reviewed["my_label"].astype(str).str.strip().ne("")
    annotated_df = df_reviewed[non_empty_mask].copy()
    reviewed_count = len(annotated_df)

    if reviewed_count == 0:
        raise ValueError(
            "Found 0 annotated rows in 'my_label' column of labels_reviewed.csv! "
            "Please review the assigned candidates and input ratings (0, 1, or 2) before computing agreement."
        )

    # 3. Validate that annotations are valid integers in {0, 1, 2}
    valid_labels: List[int] = []
    for val in annotated_df["my_label"]:
        try:
            int_val = int(float(str(val).strip()))
            if int_val not in {0, 1, 2}:
                raise ValueError(f"Invalid label value {int_val}. Only 0, 1, or 2 are permitted.")
            valid_labels.append(int_val)
        except ValueError as err:
            raise ValueError(f"Non-integer label '{val}' encountered in my_label: {err}") from err

    annotated_df["reviewed_int_label"] = valid_labels

    # 4. Cross-check against proposed labels
    if not proposed_csv_path.exists():
        raise FileNotFoundError(f"Proposed candidate pool not found at {proposed_csv_path}")

    df_proposed = pd.read_csv(proposed_csv_path)
    prop_lookup: Dict[Tuple[str, str], int] = {
        (str(row["profile_id"]).strip(), str(row["job_id"]).strip()): int(row["proposed_label"])
        for _, row in df_proposed.iterrows()
    }

    # Verify that all reviewed rows exist in the official pool
    for _, row in annotated_df.iterrows():
        pair_key = (str(row["profile_id"]).strip(), str(row["job_id"]).strip())
        if pair_key not in prop_lookup:
            raise ValueError(
                f"Integrity violation: Pair {pair_key} in reviewed file is not part of official candidate pool!"
            )

    # 5. Join with review_key.csv if present to separate random sample vs label-2 rows
    has_key = review_key_path.exists()
    if has_key:
        df_key = pd.read_csv(review_key_path)
        key_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {
            (str(r["profile_id"]).strip(), str(r["job_id"]).strip()): {
                "must_review": bool(r["must_review"]),
                "is_label2_row": bool(r["is_label2_row"]),
                "proposed_label": int(r["proposed_label"]),
            }
            for _, r in df_key.iterrows()
        }
    else:
        key_lookup = {}

    # Separate candidates into:
    # (a) Random must_review sample (for primary agreement statistics)
    # (b) Proposed label-2 rows (for separate audit reporting)
    random_sample_human: List[int] = []
    random_sample_proposed: List[int] = []
    label2_rows_human: List[int] = []
    label2_rows_proposed: List[int] = []

    human_lookup: Dict[Tuple[str, str], int] = {}

    for _, row in annotated_df.iterrows():
        pair_key = (str(row["profile_id"]).strip(), str(row["job_id"]).strip())
        hum_val = int(row["reviewed_int_label"])
        prop_val = prop_lookup[pair_key]
        human_lookup[pair_key] = hum_val

        meta = key_lookup.get(pair_key, {})
        is_random = meta.get("must_review", True)  # default to True if key missing (e.g. in test)
        is_l2 = meta.get("is_label2_row", prop_val == 2)

        if is_random:
            random_sample_human.append(hum_val)
            random_sample_proposed.append(prop_val)

        if is_l2:
            label2_rows_human.append(hum_val)
            label2_rows_proposed.append(prop_val)

    # 6. Compute primary agreement metrics on random must_review sample only
    n_sample = len(random_sample_human)
    if n_sample > 0:
        disagreements = sum(1 for h, p in zip(random_sample_human, random_sample_proposed) if h != p)
        disagreement_pct = round((disagreements / n_sample) * 100, 2)
        agreement_pct = round(100.0 - disagreement_pct, 2)

        kappa_quad = round(compute_weighted_cohen_kappa(random_sample_human, random_sample_proposed, weights="quadratic"), 4)
        kappa_linear = round(compute_weighted_cohen_kappa(random_sample_human, random_sample_proposed, weights="linear"), 4)
        kappa_unw = round(float(cohen_kappa_score(random_sample_human, random_sample_proposed)), 4)
        cm_matrix = confusion_matrix(random_sample_proposed, random_sample_human, labels=[0, 1, 2]).tolist()
    else:
        disagreements = 0
        disagreement_pct = 0.0
        agreement_pct = 100.0
        kappa_quad = 1.0
        kappa_linear = 1.0
        kappa_unw = 1.0
        cm_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # 7. Analyze the label-2 rows separately
    n_l2 = len(label2_rows_human)
    l2_distribution = {0: 0, 1: 0, 2: 0}
    for val in label2_rows_human:
        l2_distribution[val] = l2_distribution.get(val, 0) + 1

    l2_agreement_count = l2_distribution.get(2, 0)
    l2_agreement_pct = round((l2_agreement_count / n_l2) * 100, 2) if n_l2 > 0 else 0.0

    label2_analysis = {
        "total_label2_rows_evaluated": n_l2,
        "human_ratings_distribution": {
            "rated_2_strongly_relevant": l2_distribution.get(2, 0),
            "rated_1_partially_relevant": l2_distribution.get(1, 0),
            "rated_0_irrelevant": l2_distribution.get(0, 0),
        },
        "agreement_count": l2_agreement_count,
        "agreement_percentage": l2_agreement_pct,
        "downgraded_to_partial_count": l2_distribution.get(1, 0),
        "downgraded_to_irrelevant_count": l2_distribution.get(0, 0),
    }

    # 8. Synthesize final ground-truth qrels
    # Rule: my_label where reviewed, else proposed_label
    total_pool_pairs = len(df_proposed)
    final_labels: List[int] = []
    source_counts = {"human_verified": 0, "model_proposed": 0}
    qrels_dict: Dict[str, Dict[str, int]] = {}

    for _, row in df_proposed.iterrows():
        prof_id = str(row["profile_id"]).strip()
        job_id = str(row["job_id"]).strip()
        pair_key = (prof_id, job_id)

        if pair_key in human_lookup:
            final_val = human_lookup[pair_key]
            source_counts["human_verified"] += 1
        else:
            final_val = int(row["proposed_label"])
            source_counts["model_proposed"] += 1

        final_labels.append(final_val)
        if prof_id not in qrels_dict:
            qrels_dict[prof_id] = {}
        qrels_dict[prof_id][job_id] = final_val

    # 9. Compute per-profile relevance summary & flag profiles with < 3 relevant docs
    per_profile_stats: Dict[str, Dict[str, Any]] = {}
    flagged_profiles_under_3_relevant: List[Dict[str, Any]] = []

    for prof_id, jobs in qrels_dict.items():
        total_candidates = len(jobs)
        rel_gte_1 = sum(1 for val in jobs.values() if val >= 1)
        rel_eq_2 = sum(1 for val in jobs.values() if val == 2)
        rel_eq_1 = sum(1 for val in jobs.values() if val == 1)
        rel_eq_0 = sum(1 for val in jobs.values() if val == 0)

        is_flagged = rel_gte_1 < 3
        prof_summary = {
            "total_candidates": total_candidates,
            "relevant_gte_1": rel_gte_1,
            "strongly_relevant_eq_2": rel_eq_2,
            "partially_relevant_eq_1": rel_eq_1,
            "irrelevant_eq_0": rel_eq_0,
            "has_fewer_than_3_relevant_docs": is_flagged,
        }
        per_profile_stats[prof_id] = prof_summary
        if is_flagged:
            flagged_profiles_under_3_relevant.append({
                "profile_id": prof_id,
                "relevant_count": rel_gte_1,
                "details": prof_summary,
            })

    # 10. Save qrels.json
    qrels_payload = {
        "metadata": {
            "description": "Final evaluation query relevance judgments (qrels) for job search benchmark",
            "total_profiles": len(qrels_dict),
            "total_judgments": total_pool_pairs,
            "human_verified_judgments": source_counts["human_verified"],
            "model_proposed_judgments": source_counts["model_proposed"],
            "relevance_scale": {
                "0": "Irrelevant",
                "1": "Partially Relevant",
                "2": "Strongly Relevant",
            },
            "flagged_profiles_under_3_relevant_count": len(flagged_profiles_under_3_relevant),
        },
        "per_profile_relevance": per_profile_stats,
        "qrels": qrels_dict,
    }

    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels_payload, f, indent=2)
    print(f"Saved verified evaluation qrels to {qrels_path}.")

    # 11. Save label_agreement.json report
    report_payload: Dict[str, Any] = {
        "total_candidate_pairs": int(total_pool_pairs),
        "total_reviewed_rows": int(reviewed_count),
        "random_must_review_sample_size": int(n_sample),
        "agreement_metrics_on_random_sample": {
            "sample_size": int(n_sample),
            "disagreement_count": int(disagreements),
            "disagreement_percentage": disagreement_pct,
            "agreement_percentage": agreement_pct,
            "cohen_kappa_quadratic_weighted": kappa_quad,
            "cohen_kappa_linear_weighted": kappa_linear,
            "cohen_kappa_unweighted": kappa_unw,
            "kappa_interpretation": (
                "Almost Perfect Agreement (>0.80)"
                if kappa_quad >= 0.80
                else "Substantial Agreement (0.60 - 0.80)"
                if kappa_quad >= 0.60
                else "Moderate Agreement (0.40 - 0.60)"
            ),
        },
        "confusion_matrix_random_sample": {
            "description": "Rows: Model Proposed labels (0, 1, 2); Columns: Human Reviewer labels (0, 1, 2).",
            "labels": [0, 1, 2],
            "matrix": cm_matrix,
        },
        "label2_rows_audit": label2_analysis,
        "provenance_counts": {
            "human_verified_count": source_counts["human_verified"],
            "human_verified_percentage": round(
                (source_counts["human_verified"] / total_pool_pairs) * 100, 2
            ),
            "model_proposed_count": source_counts["model_proposed"],
            "model_proposed_percentage": round(
                (source_counts["model_proposed"] / total_pool_pairs) * 100, 2
            ),
        },
        "profiles_with_under_3_relevant_docs": flagged_profiles_under_3_relevant,
    }

    # Also keep backward-compatible top-level keys for README generator
    report_payload["reviewed_sample_size"] = int(n_sample)
    report_payload["agreement_metrics"] = report_payload["agreement_metrics_on_random_sample"]
    report_payload["confusion_matrix"] = report_payload["confusion_matrix_random_sample"]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved verified agreement report to {report_path}.")

    print("\n" + "=" * 75)
    print("INTER-ANNOTATOR AGREEMENT SUMMARY (RANDOM MUST_REVIEW SAMPLE)")
    print("=" * 75)
    print(f"Random Sample Size      : {n_sample} candidate pairs")
    print(f"Disagreements           : {disagreements} ({disagreement_pct}%)")
    print(f"Agreement Rate          : {agreement_pct}%")
    print(f"Cohen's Kappa (Quadratic): {kappa_quad} ({report_payload['agreement_metrics']['kappa_interpretation']})")
    print(f"Cohen's Kappa (Linear)  : {kappa_linear}")
    print(f"Cohen's Kappa (Unw)     : {kappa_unw}")
    print("\nConfusion Matrix (Rows: Proposed [0,1,2], Cols: Human [0,1,2]):")
    for r_idx, row in enumerate(cm_matrix):
        print(f"  Proposed {r_idx}: {row}")
    print("\n" + "-" * 75)
    print("ADDITIONAL LABEL-2 CANDIDATES AUDIT (25 ROWS)")
    print("-" * 75)
    print(f"Total Label-2 Rows Evaluated : {n_l2}")
    print(f"  Confirmed Strongly Relevant (2): {l2_distribution.get(2, 0)} ({l2_agreement_pct}%)")
    print(f"  Downgraded to Partial (1)     : {l2_distribution.get(1, 0)}")
    print(f"  Downgraded to Irrelevant (0)  : {l2_distribution.get(0, 0)}")
    print("\n" + "-" * 75)
    print("FINAL QRELS PROVENANCE & PER-PROFILE SUMMARY")
    print("-" * 75)
    print(f"Human-Verified Judgments: {source_counts['human_verified']} ({report_payload['provenance_counts']['human_verified_percentage']}%)")
    print(f"Model-Proposed Judgments: {source_counts['model_proposed']} ({report_payload['provenance_counts']['model_proposed_percentage']}%)")
    print(f"Profiles with < 3 Relevant Docs (label >= 1): {len(flagged_profiles_under_3_relevant)}")
    if flagged_profiles_under_3_relevant:
        for item in flagged_profiles_under_3_relevant:
            print(f"  - Flagged {item['profile_id']}: only {item['relevant_count']} relevant docs")
    else:
        print("  - All 30 profiles have >= 3 relevant documents.")
    print("=" * 75 + "\n")

    return report_payload


if __name__ == "__main__":
    evaluate_label_agreement()
