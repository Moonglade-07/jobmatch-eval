"""Compute inter-annotator agreement metrics and construct final evaluation qrels.

Calculates:
  1. Percentage of reviewed rows with disagreement (my_label != proposed_label).
  2. Cohen's weighted kappa (both quadratic and linear weights).
  3. Complete 3x3 confusion matrix across relevance categories (0, 1, 2).
  4. Final ground truth labels (my_label when human-reviewed, proposed_label otherwise).
  5. Exports data/eval/qrels.json and reports/label_agreement.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

DEFAULT_LABELS_PATH = Path("labels_proposed.csv")
DEFAULT_QRELS_PATH = Path("data/eval/qrels.json")
DEFAULT_AGREEMENT_REPORT_PATH = Path("reports/label_agreement.json")


def compute_weighted_cohen_kappa(
    y_true: List[int],
    y_pred: List[int],
    weights: str = "quadratic",
    num_classes: int = 3,
) -> float:
    """Compute Cohen's weighted kappa for ordinal classification.

    Supports both 'quadratic' and 'linear' weighting schemes.

    Args:
        y_true: Ground truth or human reviewer ratings {0, 1, ..., num_classes-1}.
        y_pred: Model or heuristic proposed ratings.
        weights: Weighting scheme ('quadratic' or 'linear').
        num_classes: Number of distinct ordinal rating categories (default: 3).

    Returns:
        Weighted kappa score in range [-1.0, 1.0].
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred.")
    if len(y_true) == 0:
        return 1.0

    return float(cohen_kappa_score(y_true, y_pred, weights=weights))


def evaluate_label_agreement(
    labels_csv_path: Path = DEFAULT_LABELS_PATH,
    qrels_path: Path = DEFAULT_QRELS_PATH,
    report_path: Path = DEFAULT_AGREEMENT_REPORT_PATH,
) -> Dict[str, Any]:
    """Calculate inter-annotator agreement metrics and export final qrels and report.

    Args:
        labels_csv_path: Path to labels_proposed.csv with human reviewed labels.
        qrels_path: Destination path for data/eval/qrels.json.
        report_path: Destination path for reports/label_agreement.json.

    Returns:
        Dictionary containing agreement metrics and distribution statistics.
    """
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"Labels CSV file not found at {labels_csv_path}")

    print(f"Loading labeled candidates from {labels_csv_path}...")
    df = pd.read_csv(labels_csv_path)

    total_pairs = len(df)
    # Identify rows reviewed by human (non-empty my_label)
    reviewed_mask = df["my_label"].notna() & df["my_label"].astype(str).str.strip().ne("")
    df_reviewed = df[reviewed_mask].copy()

    reviewed_count = len(df_reviewed)
    unreviewed_count = total_pairs - reviewed_count

    if reviewed_count == 0:
        raise ValueError("No human-reviewed rows found in my_label column. Please annotate sample rows first.")

    y_proposed = df_reviewed["proposed_label"].astype(int).tolist()
    y_human = df_reviewed["my_label"].astype(int).tolist()

    # 1. Compute disagreement count & percentage
    disagreements = sum(1 for p, h in zip(y_proposed, y_human) if p != h)
    disagreement_pct = round((disagreements / reviewed_count) * 100, 2)
    agreement_pct = round(100.0 - disagreement_pct, 2)

    # 2. Compute Cohen's kappa (quadratic and linear weighted, plus unweighted)
    kappa_quad = round(compute_weighted_cohen_kappa(y_human, y_proposed, weights="quadratic"), 4)
    kappa_linear = round(compute_weighted_cohen_kappa(y_human, y_proposed, weights="linear"), 4)
    kappa_unweighted = round(float(cohen_kappa_score(y_human, y_proposed)), 4)

    # 3. Compute 3x3 confusion matrix
    cm_matrix = confusion_matrix(y_proposed, y_human, labels=[0, 1, 2])
    cm_list = cm_matrix.tolist()

    # 4. Construct final ground truth labels:
    # final_label = my_label if reviewed, else proposed_label
    final_labels: List[int] = []
    source_distribution = {"human_verified": 0, "model_proposed": 0}

    for _, row in df.iterrows():
        val = row["my_label"]
        if pd.notna(val) and str(val).strip() != "":
            final_labels.append(int(val))
            source_distribution["human_verified"] += 1
        else:
            final_labels.append(int(row["proposed_label"]))
            source_distribution["model_proposed"] += 1

    df["final_label"] = final_labels

    # 5. Build qrels dictionary mapping: profile_id -> {job_id: final_label}
    qrels_dict: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        prof_id = str(row["profile_id"])
        job_id = str(row["job_id"])
        label = int(row["final_label"])

        if prof_id not in qrels_dict:
            qrels_dict[prof_id] = {}
        qrels_dict[prof_id][job_id] = label

    # 6. Final label distribution
    final_counts = df["final_label"].value_counts().to_dict()
    final_distribution = {
        f"label_{k}": {
            "count": int(v),
            "percentage": round((v / total_pairs) * 100, 2),
        }
        for k, v in sorted(final_counts.items())
    }

    # 7. Serialize qrels.json
    qrels_payload = {
        "metadata": {
            "description": "Evaluation query relevance judgments (qrels) for job search benchmark",
            "total_profiles": len(qrels_dict),
            "total_judgments": total_pairs,
            "human_verified_judgments": source_distribution["human_verified"],
            "model_proposed_judgments": source_distribution["model_proposed"],
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
    print(f"Saved final qrels to {qrels_path}.")

    # 8. Serialize report
    report_payload: Dict[str, Any] = {
        "total_candidate_pairs": int(total_pairs),
        "reviewed_sample_size": int(reviewed_count),
        "reviewed_sample_percentage": round((reviewed_count / total_pairs) * 100, 2),
        "unreviewed_model_count": int(unreviewed_count),
        "agreement_metrics": {
            "disagreement_count": int(disagreements),
            "disagreement_percentage": disagreement_pct,
            "agreement_percentage": agreement_pct,
            "cohen_kappa_quadratic_weighted": kappa_quad,
            "cohen_kappa_linear_weighted": kappa_linear,
            "cohen_kappa_unweighted": kappa_unweighted,
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
            "human_verified_count": source_distribution["human_verified"],
            "human_verified_percentage": round(
                (source_distribution["human_verified"] / total_pairs) * 100, 2
            ),
            "model_proposed_count": source_distribution["model_proposed"],
            "model_proposed_percentage": round(
                (source_distribution["model_proposed"] / total_pairs) * 100, 2
            ),
        },
        "final_label_distribution": final_distribution,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved agreement report to {report_path}.")

    print("\n" + "=" * 70)
    print("INTER-ANNOTATOR AGREEMENT SUMMARY")
    print("=" * 70)
    print(f"Total Reviewed Pairs   : {reviewed_count} / {total_pairs} ({report_payload['reviewed_sample_percentage']}%)")
    print(f"Disagreements          : {disagreements} ({disagreement_pct}%)")
    print(f"Agreement Rate         : {agreement_pct}%")
    print(f"Cohen's Kappa (Quadratic): {kappa_quad} ({report_payload['agreement_metrics']['kappa_interpretation']})")
    print(f"Cohen's Kappa (Linear)   : {kappa_linear}")
    print(f"Human-Verified Labels  : {source_distribution['human_verified']}")
    print(f"Model-Proposed Labels  : {source_distribution['model_proposed']}")
    print("-" * 70)
    print("Confusion Matrix [Rows: Proposed, Columns: Human Reviewer]:")
    print("            Human_0  Human_1  Human_2")
    for r_idx, label_name in enumerate(["Proposed_0", "Proposed_1", "Proposed_2"]):
        print(f"{label_name:<11} {cm_list[r_idx][0]:>8} {cm_list[r_idx][1]:>8} {cm_list[r_idx][2]:>8}")
    print("=" * 70)

    return report_payload


if __name__ == "__main__":
    evaluate_label_agreement()

