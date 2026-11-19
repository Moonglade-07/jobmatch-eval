"""Unit tests for label agreement evaluation, Cohen's kappa calculations, and qrels generation."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.build_unverified_qrels import build_unverified_qrels
from src.label_agreement import compute_weighted_cohen_kappa, evaluate_label_agreement


def test_compute_weighted_cohen_kappa_perfect_agreement() -> None:
    """Test that identical rating vectors yield a kappa of exactly 1.0."""
    y1 = [0, 1, 2, 0, 1, 2, 1, 0]
    y2 = [0, 1, 2, 0, 1, 2, 1, 0]

    kappa_quad = compute_weighted_cohen_kappa(y1, y2, weights="quadratic")
    kappa_lin = compute_weighted_cohen_kappa(y1, y2, weights="linear")

    assert pytest.approx(kappa_quad, abs=1e-5) == 1.0
    assert pytest.approx(kappa_lin, abs=1e-5) == 1.0


def test_compute_weighted_cohen_kappa_hand_computed_example() -> None:
    """Verify weighted kappa arithmetic on a simple 3-class hand-computed example.

    Ratings:
      y_true = [0, 0, 1, 1, 2, 2]
      y_pred = [0, 1, 1, 2, 2, 2]

    Observations:
      Pairs: (0,0), (0,1), (1,1), (1,2), (2,2), (2,2)
      Agreements: 4 out of 6 (66.7%).
      Disagreements: (0,1) and (1,2), both distance = 1.
    """
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 2]

    kappa_quad = compute_weighted_cohen_kappa(y_true, y_pred, weights="quadratic")
    # Weighted kappa should reflect substantial positive agreement
    assert 0.60 < kappa_quad < 0.90


def test_evaluate_label_agreement_end_to_end(tmp_path: Path) -> None:
    """Test full pipeline on temporary CSV: agreement metrics, qrels, and report."""
    proposed_data = [
        {
            "profile_id": "p1",
            "job_id": "j1",
            "title": "ML Engineer",
            "company": "Tech Corp",
            "key snippet of description": "Build ML models",
            "proposed_label": 2,
            "my_label": "",
            "must_review": True,
        },
        {
            "profile_id": "p1",
            "job_id": "j2",
            "title": "Software Dev",
            "company": "App Co",
            "key snippet of description": "Write backend APIs",
            "proposed_label": 1,
            "my_label": "",
            "must_review": True,
        },
        {
            "profile_id": "p2",
            "job_id": "j3",
            "title": "Accountant",
            "company": "Finance Inc",
            "key snippet of description": "Balance sheet audits",
            "proposed_label": 0,
            "my_label": "",
            "must_review": False,
        },
    ]
    df_prop = pd.DataFrame(proposed_data)
    test_prop_csv = tmp_path / "labels_proposed.csv"
    df_prop.to_csv(test_prop_csv, index=False)

    # Simulated authentic returned review file with 2 annotations
    reviewed_data = [
        {
            "profile_id": "p1",
            "job_id": "j1",
            "title": "ML Engineer",
            "company": "Tech Corp",
            "key snippet of description": "Build ML models",
            "my_label": 2,  # Agreement (2 vs 2)
        },
        {
            "profile_id": "p1",
            "job_id": "j2",
            "title": "Software Dev",
            "company": "App Co",
            "key snippet of description": "Write backend APIs",
            "my_label": 0,  # Disagreement (human 0 vs proposed 1)
        },
    ]
    df_rev = pd.DataFrame(reviewed_data)
    test_reviewed_csv = tmp_path / "labels_reviewed.csv"
    test_qrels = tmp_path / "test_qrels.json"
    test_report = tmp_path / "test_report.json"
    df_rev.to_csv(test_reviewed_csv, index=False)

    report = evaluate_label_agreement(
        reviewed_csv_path=test_reviewed_csv,
        proposed_csv_path=test_prop_csv,
        qrels_path=test_qrels,
        report_path=test_report,
    )

    # Check metrics on the 2 reviewed rows: 1 agreement, 1 disagreement
    assert report["total_candidate_pairs"] == 3
    assert report["reviewed_sample_size"] == 2
    assert report["agreement_metrics"]["disagreement_count"] == 1
    assert report["agreement_metrics"]["disagreement_percentage"] == 50.0

    # Verify qrels output
    with open(test_qrels, "r", encoding="utf-8") as f:
        qrels_data = json.load(f)

    assert qrels_data["metadata"]["total_judgments"] == 3
    assert qrels_data["metadata"]["human_verified_judgments"] == 2
    assert qrels_data["metadata"]["model_proposed_judgments"] == 1

    # Check that final labels followed the rule: my_label if reviewed, else proposed
    assert qrels_data["qrels"]["p1"]["j1"] == 2
    assert qrels_data["qrels"]["p1"]["j2"] == 0  # took human label (0), not proposed (1)
    assert qrels_data["qrels"]["p2"]["j3"] == 0  # unreviewed, took proposed (0)


def test_evaluate_label_agreement_refuses_missing_or_invalid_filename(tmp_path: Path) -> None:
    """Verify that evaluate_label_agreement refuses execution if file is missing or wrongly named."""
    missing_csv = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError, match="Required human review file .* not found"):
        evaluate_label_agreement(reviewed_csv_path=missing_csv)

    wrong_name_csv = tmp_path / "wrong_name.csv"
    wrong_name_csv.write_text("my_label\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid review file name"):
        evaluate_label_agreement(reviewed_csv_path=wrong_name_csv)


def test_evaluate_label_agreement_refuses_zero_annotations(tmp_path: Path) -> None:
    """Verify that evaluate_label_agreement refuses execution if my_label is completely blank."""
    empty_csv = tmp_path / "labels_reviewed.csv"
    empty_csv.write_text("profile_id,job_id,my_label\np1,j1,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Found 0 annotated rows in 'my_label' column"):
        evaluate_label_agreement(reviewed_csv_path=empty_csv)


def test_build_unverified_qrels_produces_valid_payload(tmp_path: Path) -> None:
    """Verify that build_unverified_qrels creates expected qrels.json and label_agreement.json."""
    sample_df = pd.DataFrame([
        {"profile_id": "p1", "job_id": 101, "proposed_label": 2},
        {"profile_id": "p1", "job_id": 102, "proposed_label": 1},
        {"profile_id": "p2", "job_id": 201, "proposed_label": 0},
    ])
    test_csv = tmp_path / "labels_proposed.csv"
    test_qrels = tmp_path / "qrels.json"
    test_report = tmp_path / "label_agreement.json"
    sample_df.to_csv(test_csv, index=False)

    res = build_unverified_qrels(
        proposed_csv_path=test_csv,
        qrels_path=test_qrels,
        report_path=test_report,
    )

    assert res["total_judgments"] == 3
    assert res["human_verified_count"] == 0
    assert res["model_proposed_count"] == 3

    with open(test_qrels, "r", encoding="utf-8") as f:
        qrels_data = json.load(f)

    assert qrels_data["metadata"]["human_verified_count"] == 0
    assert qrels_data["metadata"]["model_proposed_count"] == 3
    assert qrels_data["metadata"]["note"] == "qrels are model-proposed and unverified"
    assert qrels_data["qrels"]["p1"]["101"] == {"relevance": 2, "label_source": "model_proposed"}
    assert qrels_data["qrels"]["p1"]["102"] == {"relevance": 1, "label_source": "model_proposed"}
    assert qrels_data["qrels"]["p2"]["201"] == {"relevance": 0, "label_source": "model_proposed"}

    with open(test_report, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert report_data == {
        "human_reviewed": False,
        "human_verified_count": 0,
        "note": "qrels are model-proposed and unverified",
    }


def test_export_relevant_docs_per_profile_matches_qrels() -> None:
    """Verify that export_relevant_docs_per_profile correctly tallies judged pairs and relevant docs."""
    from src.export_relevant_docs_per_profile import export_relevant_docs_per_profile

    payload = export_relevant_docs_per_profile()
    assert payload["total_profiles"] == 30
    assert payload["caption"] == "Evaluated against model-proposed, unverified relevance labels"
    assert len(payload["profiles"]) == 30

    # Verify that exactly 6 profiles have < 3 relevant docs
    under_3_ids = {p["profile_id"] for p in payload["profiles_with_fewer_than_3_relevant_docs"]}
    assert under_3_ids == {"prof_01", "prof_07", "prof_09", "prof_11", "prof_19", "prof_23"}
    assert payload["profiles_with_fewer_than_3_relevant_docs_count"] == 6

    # Verify that prof_07 and prof_23 have exactly 0 relevant docs
    p07 = next(p for p in payload["profiles"] if p["profile_id"] == "prof_07")
    p23 = next(p for p in payload["profiles"] if p["profile_id"] == "prof_23")
    assert p07["relevant_count"] == 0
    assert p23["relevant_count"] == 0
    assert p07["target_role"] == "Mobile Developer"
    assert p23["target_role"] == "Computer Vision Engineer"


