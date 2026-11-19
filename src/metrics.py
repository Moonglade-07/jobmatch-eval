"""Evaluation metrics for retrieval benchmarking: P@5, Recall@10, MRR, and nDCG@10.

OPERATIONAL RULES:
  1. Graded Relevance:
     - Relevance threshold: A document is considered binary-relevant if its qrel label >= threshold (default: 1).
     - Graded nDCG: Uses full graded relevance gain (default: 2^r - 1) for r in {0, 1, 2}.
     - No label == 2 binary variant is reported (per protocol: only ~25 unverified label-2 rows).
  2. Unjudged Documents:
     - Documents retrieved but absent from qrels are treated strictly as non-relevant (gain = 0).
     - The unjudged rate in top-10 is tracked and reported per strategy.
  3. Zero-Relevant Profiles:
     - Profiles with 0 relevant documents in the pool (total_relevant == 0) cannot produce
       a valid recall denominator or meaningful reciprocal rank.
     - They are EXCLUDED from the macro-average means for Recall@10 and MRR.
     - The number and IDs of excluded profiles are explicitly reported in all outputs.
     - For nDCG@10, when IDCG == 0 (no relevant docs), nDCG is defined as 0.0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


def get_relevance_label(
    qrels_for_profile: Dict[str, Union[int, Dict[str, Any]]],
    doc_id: str,
) -> int:
    """Retrieve integer relevance label for a document, treating unjudged as 0.

    Args:
        qrels_for_profile: Dictionary mapping doc_id to either int label or entry dict.
        doc_id: Document / Job identifier string.

    Returns:
        Integer relevance grade {0, 1, 2}. Unjudged documents return 0.
    """
    if doc_id not in qrels_for_profile:
        return 0
    entry = qrels_for_profile[doc_id]
    if isinstance(entry, dict):
        return int(entry.get("relevance", 0))
    return int(entry)


def precision_at_k(
    retrieved_ids: List[str],
    qrels_for_profile: Dict[str, Union[int, Dict[str, Any]]],
    k: int = 5,
    relevance_threshold: int = 1,
) -> float:
    """Compute Precision at rank k (P@k).

    Formula:
        P@k = (Number of relevant documents in top k) / k

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        qrels_for_profile: Relevance judgments for this profile.
        k: Cutoff rank (standard: 5).
        relevance_threshold: Minimum label considered relevant (default: 1).

    Returns:
        Precision score in range [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    cutoff_docs = retrieved_ids[:k]
    relevant_count = sum(
        1
        for doc_id in cutoff_docs
        if get_relevance_label(qrels_for_profile, doc_id) >= relevance_threshold
    )
    return float(relevant_count / k)


def recall_at_k(
    retrieved_ids: List[str],
    qrels_for_profile: Dict[str, Union[int, Dict[str, Any]]],
    k: int = 10,
    relevance_threshold: int = 1,
) -> Optional[float]:
    """Compute Recall at rank k (Recall@k).

    Formula:
        Recall@k = (Number of relevant documents in top k) / (Total relevant documents in qrels)

    Zero-Relevant Handling:
        If total relevant documents in qrels == 0, returns None (excluded from mean).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        qrels_for_profile: Relevance judgments for this profile.
        k: Cutoff rank (standard: 10).
        relevance_threshold: Minimum label considered relevant (default: 1).

    Returns:
        Recall score in range [0.0, 1.0], or None if no relevant documents exist.
    """
    total_relevant = sum(
        1
        for doc_id in qrels_for_profile.keys()
        if get_relevance_label(qrels_for_profile, doc_id) >= relevance_threshold
    )
    if total_relevant == 0:
        return None

    cutoff_docs = retrieved_ids[:k]
    retrieved_relevant = sum(
        1
        for doc_id in cutoff_docs
        if get_relevance_label(qrels_for_profile, doc_id) >= relevance_threshold
    )
    return float(retrieved_relevant / total_relevant)


def reciprocal_rank(
    retrieved_ids: List[str],
    qrels_for_profile: Dict[str, Union[int, Dict[str, Any]]],
    relevance_threshold: int = 1,
) -> Optional[float]:
    """Compute Reciprocal Rank (RR) for a single query.

    Formula:
        RR = 1.0 / rank of first relevant document (1-indexed).
        If no relevant document is retrieved:
          - Returns 0.0 if total_relevant > 0.
          - Returns None if total_relevant == 0 (excluded from MRR mean).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        qrels_for_profile: Relevance judgments for this profile.
        relevance_threshold: Minimum label considered relevant (default: 1).

    Returns:
        Reciprocal rank float in [0.0, 1.0], or None if no relevant docs exist.
    """
    total_relevant = sum(
        1
        for doc_id in qrels_for_profile.keys()
        if get_relevance_label(qrels_for_profile, doc_id) >= relevance_threshold
    )
    if total_relevant == 0:
        return None

    for rank_idx, doc_id in enumerate(retrieved_ids):
        if get_relevance_label(qrels_for_profile, doc_id) >= relevance_threshold:
            return float(1.0 / (rank_idx + 1))
    return 0.0


def dcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k.

    Standard TREC exponential formulation:
        DCG@k = sum_{i=1}^k (2^{r_i} - 1) / log2(i + 1)

    Args:
        relevance_scores: List of integer relevance grades ordered by rank.
        k: Rank cutoff.

    Returns:
        DCG score float.
    """
    dcg = 0.0
    for idx, r in enumerate(relevance_scores[:k]):
        gain = float((2 ** r) - 1)
        discount = math.log2(idx + 2)  # idx=0 -> log2(2) = 1.0
        dcg += gain / discount
    return dcg


def ndcg_at_k(
    retrieved_ids: List[str],
    qrels_for_profile: Dict[str, Union[int, Dict[str, Any]]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k (nDCG@k).

    Formula:
        nDCG@k = DCG@k / IDCG@k

    Where IDCG@k is the ideal DCG obtained by sorting all available relevance
    judgments for this query descending.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        qrels_for_profile: Relevance judgments for this profile.
        k: Cutoff rank (standard: 10).

    Returns:
        nDCG score float in range [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    retrieved_gains = [
        get_relevance_label(qrels_for_profile, doc_id)
        for doc_id in retrieved_ids[:k]
    ]
    actual_dcg = dcg_at_k(retrieved_gains, k)

    # All judged relevance scores sorted descending for ideal DCG
    all_gains = [
        get_relevance_label(qrels_for_profile, doc_id)
        for doc_id in qrels_for_profile.keys()
    ]
    all_gains.sort(reverse=True)
    ideal_dcg = dcg_at_k(all_gains, k)

    if ideal_dcg <= 0.0:
        return 0.0
    return float(min(1.0, actual_dcg / ideal_dcg))


def unjudged_rate_at_k(
    retrieved_ids: List[str],
    qrels_for_profile: Dict[str, Union[int, Dict[str, Any]]],
    k: int = 10,
) -> float:
    """Compute the fraction of retrieved top-k documents absent from qrels.

    Formula:
        UnjudgedRate@k = (Number of unjudged docs in top k) / (min(k, len(retrieved_ids)))

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        qrels_for_profile: Relevance judgments for this profile.
        k: Rank cutoff.

    Returns:
        Fraction of unjudged documents in [0.0, 1.0].
    """
    cutoff = retrieved_ids[:k]
    if not cutoff:
        return 0.0
    unjudged_count = sum(1 for doc_id in cutoff if doc_id not in qrels_for_profile)
    return float(unjudged_count / len(cutoff))


def evaluate_query(
    retrieved_ids: List[str],
    qrels_for_profile: Dict[str, Union[int, Dict[str, Any]]],
    k_p: int = 5,
    k_recall: int = 10,
    k_ndcg: int = 10,
    relevance_threshold: int = 1,
) -> Dict[str, Any]:
    """Calculate all evaluation metrics for a single profile query.

    Args:
        retrieved_ids: Ranked document IDs returned by retriever.
        qrels_for_profile: Judged candidate pairs for this profile.
        k_p: Cutoff for precision (default: 5).
        k_recall: Cutoff for recall (default: 10).
        k_ndcg: Cutoff for nDCG (default: 10).
        relevance_threshold: Threshold for binary relevance metrics (default: 1).

    Returns:
        Dictionary of computed metric values and candidate counts.
    """
    total_rel = sum(
        1
        for doc_id in qrels_for_profile.keys()
        if get_relevance_label(qrels_for_profile, doc_id) >= relevance_threshold
    )

    p_5 = precision_at_k(retrieved_ids, qrels_for_profile, k=k_p, relevance_threshold=relevance_threshold)
    rec_10 = recall_at_k(retrieved_ids, qrels_for_profile, k=k_recall, relevance_threshold=relevance_threshold)
    rr = reciprocal_rank(retrieved_ids, qrels_for_profile, relevance_threshold=relevance_threshold)
    ndcg_10 = ndcg_at_k(retrieved_ids, qrels_for_profile, k=k_ndcg)
    unjudged_10 = unjudged_rate_at_k(retrieved_ids, qrels_for_profile, k=10)

    return {
        "precision@5": round(p_5, 4),
        "recall@10": round(rec_10, 4) if rec_10 is not None else None,
        "mrr": round(rr, 4) if rr is not None else None,
        "ndcg@10": round(ndcg_10, 4),
        "unjudged_rate@10": round(unjudged_10, 4),
        "total_relevant_docs": total_rel,
        "is_zero_relevant_profile": total_rel == 0,
        "num_retrieved": len(retrieved_ids),
    }


def compute_macro_averages(
    per_profile_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute macro-average metrics across all evaluated queries with explicit zero-handling.

    Zero-handling rules:
      - Recall@10 and MRR exclude profiles where total_relevant_docs == 0.
      - P@5 and nDCG@10 include all profiles.

    Args:
        per_profile_metrics: Dict mapping profile_id to per-query metrics dict.

    Returns:
        Dictionary of macro-averaged scores and execution metadata.
    """
    total_profiles = len(per_profile_metrics)
    if total_profiles == 0:
        return {
            "total_profiles": 0,
            "mean_precision@5": 0.0,
            "mean_recall@10": 0.0,
            "mean_mrr": 0.0,
            "mean_ndcg@10": 0.0,
            "mean_unjudged_rate@10": 0.0,
            "excluded_zero_relevant_count": 0,
            "excluded_zero_relevant_profiles": [],
        }

    p5_vals: List[float] = []
    ndcg10_vals: List[float] = []
    unjudged_vals: List[float] = []

    recall_vals: List[float] = []
    mrr_vals: List[float] = []
    excluded_profiles: List[str] = []

    for prof_id, m in per_profile_metrics.items():
        p5_vals.append(float(m["precision@5"]))
        ndcg10_vals.append(float(m["ndcg@10"]))
        unjudged_vals.append(float(m["unjudged_rate@10"]))

        if m["is_zero_relevant_profile"] or m["recall@10"] is None:
            excluded_profiles.append(prof_id)
        else:
            recall_vals.append(float(m["recall@10"]))
            mrr_vals.append(float(m["mrr"]))

    mean_p5 = round(float(np.mean(p5_vals)), 4) if p5_vals else 0.0
    mean_ndcg10 = round(float(np.mean(ndcg10_vals)), 4) if ndcg10_vals else 0.0
    mean_unjudged = round(float(np.mean(unjudged_vals)), 4) if unjudged_vals else 0.0

    mean_recall10 = round(float(np.mean(recall_vals)), 4) if recall_vals else 0.0
    mean_mrr = round(float(np.mean(mrr_vals)), 4) if mrr_vals else 0.0

    return {
        "total_profiles_evaluated": total_profiles,
        "profiles_with_relevant_docs": len(recall_vals),
        "excluded_zero_relevant_count": len(excluded_profiles),
        "excluded_zero_relevant_profiles": excluded_profiles,
        "mean_precision@5": mean_p5,
        "mean_recall@10": mean_recall10,
        "mean_mrr": mean_mrr,
        "mean_ndcg@10": mean_ndcg10,
        "mean_unjudged_rate@10": mean_unjudged,
        "rules_documentation": {
            "unjudged_documents": "Documents retrieved but absent from qrels are treated as non-relevant (label 0).",
            "zero_relevant_profiles": "Profiles with zero relevant documents in qrels are excluded from Recall@10 and MRR means.",
            "ndcg_formula": "Standard TREC graded formula sum((2^r - 1)/log2(rank+1)) / IDCG@10.",
        },
    }

