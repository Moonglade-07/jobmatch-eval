"""Build unverified evaluation qrels and report from model-proposed labels.

Explicitly documents that all 1,037 relevance judgments are model-proposed
and unverified, setting human_verified_count to 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROPOSED_CSV_PATH = Path("labels_proposed.csv")
QRELS_PATH = Path("data/eval/qrels.json")
REPORT_PATH = Path("reports/label_agreement.json")


def build_unverified_qrels(
    proposed_csv_path: Path = PROPOSED_CSV_PATH,
    qrels_path: Path = QRELS_PATH,
    report_path: Path = REPORT_PATH,
) -> Dict[str, Any]:
    """Construct data/eval/qrels.json and reports/label_agreement.json from proposed labels only.

    Args:
        proposed_csv_path: Path to labels_proposed.csv.
        qrels_path: Destination path for data/eval/qrels.json.
        report_path: Destination path for reports/label_agreement.json.

    Returns:
        Summary dict of created assets.
    """
    if not proposed_csv_path.exists():
        raise FileNotFoundError(f"Proposed labels file not found at {proposed_csv_path}")

    df = pd.read_csv(proposed_csv_path)
    total_pairs = len(df)

    # 1. Build qrels dictionary
    qrels_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}
    label_counts = {0: 0, 1: 0, 2: 0}

    for _, row in df.iterrows():
        prof_id = str(row["profile_id"]).strip()
        job_id = str(row["job_id"]).strip()
        prop_label = int(row["proposed_label"])

        label_counts[prop_label] = label_counts.get(prop_label, 0) + 1

        if prof_id not in qrels_dict:
            qrels_dict[prof_id] = {}

        qrels_dict[prof_id][job_id] = {
            "relevance": prop_label,
            "label_source": "model_proposed",
        }

    qrels_payload = {
        "metadata": {
            "description": "Evaluation relevance judgments (qrels) for job search benchmark",
            "note": "qrels are model-proposed and unverified",
            "human_verified_count": 0,
            "model_proposed_count": total_pairs,
            "total_judgments": total_pairs,
            "total_profiles": len(qrels_dict),
            "relevance_scale": {
                "0": "Irrelevant",
                "1": "Partially Relevant",
                "2": "Strongly Relevant",
            },
            "label_distribution": {
                "0": label_counts.get(0, 0),
                "1": label_counts.get(1, 0),
                "2": label_counts.get(2, 0),
            },
        },
        "qrels": qrels_dict,
    }

    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels_payload, f, indent=2)
    print(f"Successfully generated {qrels_path} (100% model-proposed, 0 human-reviewed).")

    # 2. Write reports/label_agreement.json containing ONLY required fields
    report_payload = {
        "human_reviewed": False,
        "human_verified_count": 0,
        "note": "qrels are model-proposed and unverified",
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Successfully generated {report_path}: {report_payload}")

    return {
        "total_judgments": total_pairs,
        "human_verified_count": 0,
        "model_proposed_count": total_pairs,
        "qrels_path": str(qrels_path),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    build_unverified_qrels()
