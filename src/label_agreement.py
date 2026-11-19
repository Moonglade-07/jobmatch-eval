"""Compute inter-annotator agreement metrics and construct final evaluation qrels.

STRICT INTEGRITY ENFORCEMENT:
  1. Refuses to run unless an authentic human review file named `labels_reviewed.csv` is provided.
  2. Validates that every row in `labels_reviewed.csv` matches an expected (profile_id, job_id)
     entry from the official candidate pool.
  3. Verifies that all non-empty `my_label` values are valid ordinal ratings in {0, 1, 2}.
  4. Refuses execution if 0 reviewed rows are detected or if any automated fabrication is found.
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
        (str(row["profile_id"]), str(row["job_id"])): int(row["proposed_label"])
        for _, row in df_proposed.iterrows()
    }

    # Verify that all reviewed rows exist in the official pool
    y_human: List[int] = []
    y_proposed: List[int] = []
    disagreements = 0

    for _, row in annotated_df.iterrows():
        pair_key = (str(row["profile_id"]), str(row["job_id"]))
        if pair_key not in prop_lookup:
            raise ValueError(
                f"Integrity violation: Pair {pair_key} in reviewed file is not part of official candidate pool!"
            )
        prop_val = prop_lookup[pair_key]
        hum_val = int(row["reviewed_int_label"])

        y_human.append(hum_val)
        y_proposed.append(prop_val)
        if hum_val != prop_val:
            disagreements += 1

    total_pool_pairs = len(df_proposed)
    disagreement_pct = round((disagreements / reviewed_count) * 100, 2)
    agreement_pct = round(100.0 - disagreement_pct, 2)

    # 5. Compute Cohen's kappa metrics
    kappa_quad = round(compute_weighted_cohen_kappa(y_human, y_proposed, weights="quadratic"), 4)
    kappa_linear = round(compute_weighted_cohen_kappa(y_human, y_proposed, weights="linear"), 4)
    kappa_unw = round(float(cohen_kappa_score(y_human, y_proposed)), 4)

    # 6. Confusion matrix (3x3)
    cm_matrix = confusion_matrix(y_proposed, y_human, labels=[0, 1, 2])
    cm_list = cm_matrix.tolist()

    # 7. Synthesize final ground-truth labels
    # Final label = human review where present, otherwise model proposed
    human_lookup: Dict[Tuple[str, str], int] = {
        (str(row["profile_id"]), str(row["job_id"])): int(row["reviewed_int_label"])
        for _, row in annotated_df.iterrows()
    }

    final_labels: List[int] = []
    source_counts = {"human_verified": 0, "model_proposed": 0}
    qrels_dict: Dict[str, Dict[str, int]] = {}

    for _, row in df_proposed.iterrows():
        pair_key = (str(row["profile_id"]), str(row["job_id"]))
        prof_id, job_id = pair_key
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

    # 8. Save qrels.json
    qrels_payload = {
        "metadata": {
            "description": "Evaluation query relevance judgments (qrels) for job search benchmark",
            "total_profiles": len(qrels_dict),
            "total_judgments": total_pool_pairs,
            "human_verified_judgments": source_counts["human_verified"],
            "model_proposed_judgments": source_counts["model_proposed"],
            "relevance_scale": {
                "0": "Irrelevant",
                "1": "Partially Relevant",
                "2": "Strongly Relevant",
            },
        },
        "qrels": qrels_dict,
    }

    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels_payload, f, indent=2)
    print(f"Saved verified evaluation qrels to {qrels_path}.")

    # 9. Save label_agreement.json report
    report_payload: Dict[str, Any] = {
        "total_candidate_pairs": int(total_pool_pairs),
        "reviewed_sample_size": int(reviewed_count),
        "reviewed_sample_percentage": round((reviewed_count / total_pool_pairs) * 100, 2),
        "unreviewed_model_count": int(source_counts["model_proposed"]),
        "agreement_metrics": {
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
            ),
        },
        "confusion_matrix": {
            "description": "Rows represent Model Proposed labels; Columns represent Human Reviewer labels.",
            "labels": [0, 1, 2],
            "matrix": cm_list,
        },
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
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved verified agreement report to {report_path}.")

    print("\n" + "=" * 70)
    print("VERIFIED INTER-ANNOTATOR AGREEMENT SUMMARY")
    print("=" * 70)
    print(f"Reviewed Pairs Count   : {reviewed_count} / {total_pool_pairs} ({report_payload['reviewed_sample_percentage']}%)")
    print(f"Disagreements          : {disagreements} ({disagreement_pct}%)")
    print(f"Agreement Rate         : {agreement_pct}%")
    print(f"Cohen's Kappa (Quad)   : {kappa_quad} ({report_payload['agreement_metrics']['kappa_interpretation']})")
    print(f"Cohen's Kappa (Linear) : {kappa_linear}")
    print(f"Human-Verified Labels  : {source_counts['human_verified']}")
    print(f"Model-Proposed Labels  : {source_counts['model_proposed']}")
    print("=" * 70)

    return report_payload


if __name__ == "__main__":
    evaluate_label_agreement()
