"""Unit tests for information retrieval metrics: P@5, Recall@10, MRR, nDCG@10, and unjudged handling."""

import math
from typing import Dict, List
import pytest

from src.metrics import (
    compute_macro_averages,
    dcg_at_k,
    evaluate_query,
    get_relevance_label,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unjudged_rate_at_k,
)


def test_hand_computed_precision_and_recall_and_mrr() -> None:
    """Verify hand-computed P@5, Recall@10, and MRR on an exact small example."""
    # Qrels: d1 (2), d2 (1), d3 (0), d4 (1), d5 (0), d6 (2) -> 4 relevant (d1, d2, d4, d6)
    qrels = {
        "d1": {"relevance": 2, "label_source": "model_proposed"},
        "d2": 1,
        "d3": 0,
        "d4": {"relevance": 1},
        "d5": 0,
        "d6": 2,
    }
    # Retrieved 6 documents (d7 is unjudged)
    retrieved = ["d1", "d3", "d2", "d7", "d4", "d5"]

    # 1. Precision@5: top 5 are [d1 (rel), d3 (not), d2 (rel), d7 (unjudged->not), d4 (rel)]
    # Relevant count = 3 -> 3 / 5 = 0.60
    p5 = precision_at_k(retrieved, qrels, k=5, relevance_threshold=1)
    assert pytest.approx(p5, abs=1e-5) == 0.60

    # 2. Recall@10: 3 relevant retrieved out of 4 total relevant in qrels -> 3 / 4 = 0.75
    rec10 = recall_at_k(retrieved, qrels, k=10, relevance_threshold=1)
    assert rec10 is not None
    assert pytest.approx(rec10, abs=1e-5) == 0.75

    # 3. Reciprocal Rank: first relevant doc is d1 at rank 1 -> 1.0 / 1 = 1.0
    rr = reciprocal_rank(retrieved, qrels, relevance_threshold=1)
    assert rr is not None
    assert pytest.approx(rr, abs=1e-5) == 1.0

    # Test Reciprocal Rank when first relevant doc is at rank 3
    retrieved_sub = ["d3", "d7", "d2", "d4"]
    rr_3 = reciprocal_rank(retrieved_sub, qrels, relevance_threshold=1)
    assert rr_3 is not None
    assert pytest.approx(rr_3, abs=1e-5) == 1.0 / 3.0


def test_hand_computed_ndcg_at_k() -> None:
    """Verify hand-computed graded nDCG@5 against manual pen-and-paper calculation."""
    qrels = {
        "d1": 2,
        "d2": 1,
        "d3": 0,
        "d4": 1,
        "d5": 0,
        "d6": 2,
    }
    # Retrieved: ["d1", "d3", "d2", "d7", "d4"] (d7 is unjudged -> gain 0)
    retrieved = ["d1", "d3", "d2", "d7", "d4"]

    # Actual gains:
    # rank 1: d1 (r=2) -> gain (2^2-1) = 3 / log2(2) = 3.0 / 1.0 = 3.0
    # rank 2: d3 (r=0) -> gain 0
    # rank 3: d2 (r=1) -> gain (2^1-1) = 1 / log2(4) = 1.0 / 2.0 = 0.5
    # rank 4: d7 (unjudged->r=0) -> gain 0
    # rank 5: d4 (r=1) -> gain (2^1-1) = 1 / log2(6) = 1.0 / 2.584962500721156 = 0.3868528
    # DCG@5 = 3.0 + 0 + 0.5 + 0 + 0.3868528 = 3.8868528
    expected_dcg = 3.0 + 0.5 + (1.0 / math.log2(6))

    # Ideal gains sorted: [2, 2, 1, 1, 0, 0]
    # rank 1: 2 -> 3 / log2(2) = 3.0
    # rank 2: 2 -> 3 / log2(3) = 3 / 1.584962500721156 = 1.89278926
    # rank 3: 1 -> 1 / log2(4) = 0.5
    # rank 4: 1 -> 1 / log2(5) = 1 / 2.321928094887362 = 0.43067656
    # rank 5: 0 -> 0
    expected_idcg = 3.0 + (3.0 / math.log2(3)) + 0.5 + (1.0 / math.log2(5))
    expected_ndcg = expected_dcg / expected_idcg

    actual_ndcg = ndcg_at_k(retrieved, qrels, k=5)
    assert pytest.approx(actual_ndcg, abs=1e-4) == round(expected_ndcg, 4)


def test_zero_relevant_docs_handling() -> None:
    """Verify that profiles with 0 relevant documents return None for recall and MRR."""
    # All documents judged as 0 (irrelevant)
    qrels = {"d1": 0, "d2": 0, "d3": 0}
    retrieved = ["d1", "d2", "d4"]

    p5 = precision_at_k(retrieved, qrels, k=5)
    assert p5 == 0.0

    # Recall must be None (not 0.0, excluded from mean)
    rec10 = recall_at_k(retrieved, qrels, k=10)
    assert rec10 is None

    # MRR must be None (excluded from mean)
    rr = reciprocal_rank(retrieved, qrels)
    assert rr is None

    # nDCG when IDCG is 0 must be 0.0
    ndcg10 = ndcg_at_k(retrieved, qrels, k=10)
    assert ndcg10 == 0.0


def test_fewer_than_k_results_returned() -> None:
    """Verify behavior when retriever returns fewer than k results."""
    qrels = {"d1": 1, "d2": 2}
    retrieved = ["d1", "d2"]  # only 2 returned, k=5

    # P@5 denominator is k (5), so 2/5 = 0.40
    p5 = precision_at_k(retrieved, qrels, k=5)
    assert pytest.approx(p5, abs=1e-5) == 0.40

    # Recall@10 denominator is total relevant (2), so 2/2 = 1.0
    rec10 = recall_at_k(retrieved, qrels, k=10)
    assert pytest.approx(rec10, abs=1e-5) == 1.0

    # MRR first relevant is rank 1 -> 1.0
    rr = reciprocal_rank(retrieved, qrels)
    assert pytest.approx(rr, abs=1e-5) == 1.0


def test_unjudged_documents_handling() -> None:
    """Verify that unjudged documents are treated as non-relevant and unjudged rate is accurate."""
    qrels = {"d1": 2, "d2": 1}
    # 2 judged, 3 unjudged
    retrieved = ["u1", "d1", "u2", "u3", "d2"]

    assert get_relevance_label(qrels, "u1") == 0
    assert get_relevance_label(qrels, "u2") == 0

    # Top 5 contains 3 unjudged -> rate = 3 / 5 = 0.60
    rate5 = unjudged_rate_at_k(retrieved, qrels, k=5)
    assert pytest.approx(rate5, abs=1e-5) == 0.60

    # Top 5 precision: 2 relevant / 5 = 0.40
    p5 = precision_at_k(retrieved, qrels, k=5)
    assert pytest.approx(p5, abs=1e-5) == 0.40


def test_ranking_ties_and_idcg_stability() -> None:
    """Verify that queries with tied relevance scores compute stable, deterministic nDCG."""
    qrels = {"d1": 1, "d2": 1, "d3": 1, "d4": 1}
    # If all items have identical relevance 1, any permutation of top 4 should give nDCG 1.0
    retrieved1 = ["d1", "d2", "d3", "d4"]
    retrieved2 = ["d4", "d3", "d2", "d1"]

    ndcg1 = ndcg_at_k(retrieved1, qrels, k=4)
    ndcg2 = ndcg_at_k(retrieved2, qrels, k=4)
    assert pytest.approx(ndcg1, abs=1e-5) == 1.0
    assert pytest.approx(ndcg2, abs=1e-5) == 1.0


def test_compute_macro_averages_excludes_zero_relevant_profiles() -> None:
    """Verify macro-average computation properly excludes zero-relevant profiles from recall and MRR."""
    per_profile = {
        "prof_valid": {
            "precision@5": 0.8,
            "recall@10": 0.6,
            "mrr": 1.0,
            "ndcg@10": 0.85,
            "unjudged_rate@10": 0.0,
            "total_relevant_docs": 5,
            "is_zero_relevant_profile": False,
        },
        "prof_zero": {
            "precision@5": 0.0,
            "recall@10": None,
            "mrr": None,
            "ndcg@10": 0.0,
            "unjudged_rate@10": 0.1,
            "total_relevant_docs": 0,
            "is_zero_relevant_profile": True,
        },
    }

    summary = compute_macro_averages(per_profile)

    assert summary["total_profiles_evaluated"] == 2
    assert summary["profiles_with_relevant_docs"] == 1
    assert summary["excluded_zero_relevant_count"] == 1
    assert summary["excluded_zero_relevant_profiles"] == ["prof_zero"]

    # Precision and nDCG include both profiles
    assert pytest.approx(summary["mean_precision@5"], abs=1e-4) == 0.40
    assert pytest.approx(summary["mean_ndcg@10"], abs=1e-4) == 0.4250

    # Recall and MRR exclude prof_zero
    assert pytest.approx(summary["mean_recall@10"], abs=1e-4) == 0.60
    assert pytest.approx(summary["mean_mrr"], abs=1e-4) == 1.0

