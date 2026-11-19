"""Unit tests for retrieval strategies: BM25, Dense, Hybrid (RRF), and Cross-Encoder."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import pytest

from src.models import Profile, SearchResult
from src.retrieval import (
    BM25Retriever,
    CrossEncoderRetriever,
    DenseRetriever,
    HybridRRFRetriever,
    reciprocal_rank_fusion,
    tokenize_text,
)


def test_reciprocal_rank_fusion_hand_computed_example() -> None:
    """Verify RRF arithmetic against an exact hand-computed mathematical example.

    Setup:
      List 1 (e.g. BM25):  [doc_A, doc_B] (ranks 1, 2)
      List 2 (e.g. Dense): [doc_B, doc_C] (ranks 1, 2)
      Parameter rrf_k = 60

    Hand Calculation:
      doc_A: 1/(60+1) + 0        = 1/61            ≈ 0.01639344
      doc_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61     ≈ 0.03252247
      doc_C: 0        + 1/(60+2) = 1/62            ≈ 0.01612903

    Expected Ordering: doc_B > doc_A > doc_C
    """
    list_1 = [
        SearchResult(job_id="doc_A", score=10.0, title="Job A"),
        SearchResult(job_id="doc_B", score=8.0, title="Job B"),
    ]
    list_2 = [
        SearchResult(job_id="doc_B", score=0.95, title="Job B"),
        SearchResult(job_id="doc_C", score=0.85, title="Job C"),
    ]

    fused = reciprocal_rank_fusion(ranked_lists=[list_1, list_2], rrf_k=60)

    assert len(fused) == 3
    # Check exact ordering
    assert fused[0].job_id == "doc_B"
    assert fused[1].job_id == "doc_A"
    assert fused[2].job_id == "doc_C"

    # Check hand-computed scores within 1e-5 tolerance
    expected_b = round(1.0 / 62.0 + 1.0 / 61.0, 6)
    expected_a = round(1.0 / 61.0, 6)
    expected_c = round(1.0 / 62.0, 6)

    assert pytest.approx(fused[0].score, rel=1e-4) == expected_b
    assert pytest.approx(fused[1].score, rel=1e-4) == expected_a
    assert pytest.approx(fused[2].score, rel=1e-4) == expected_c


def test_tokenize_text() -> None:
    """Test text tokenization retains technical symbols and lowercases."""
    text = "Hiring Senior C++ & Python-3 Engineer with .NET and PyTorch!"
    tokens = tokenize_text(text)

    assert "c++" in tokens
    assert "python-3" in tokens
    assert ".net" in tokens
    assert "pytorch" in tokens
    assert "senior" in tokens


def test_bm25_retriever_ranking() -> None:
    """Test BM25 keyword matching correctly ranks documents with matching terms higher."""
    df = pd.DataFrame(
        [
            {
                "id": "ml_job",
                "title": "Machine Learning Engineer",
                "company": "AI Core",
                "location": "Remote",
                "experience_level": "Mid-Senior level",
                "description": "Develop transformer models and vector indexing algorithms.",
            },
            {
                "id": "sales_job",
                "title": "Sales Account Manager",
                "company": "B2B SaaS",
                "location": "New York, NY",
                "experience_level": "Entry level",
                "description": "Manage outbound CRM pipeline and client relationships.",
            },
            {
                "id": "hr_job",
                "title": "People Operations Lead",
                "company": "Growth Hub",
                "location": "Chicago, IL",
                "experience_level": "Mid-Senior level",
                "description": "Coordinate employee onboarding, compliance, and talent development.",
            },
        ]
    )

    retriever = BM25Retriever(df=df)
    profile = Profile(
        target_role="Machine Learning Engineer",
        skills=["Transformer", "Vector Indexing"],
        resume_text="Experienced in building semantic retrieval algorithms.",
    )

    results = retriever.search(profile, k=3, apply_filters=False, apply_boost=False)

    assert len(results) == 3
    assert results[0].job_id == "ml_job"
    assert results[0].score > results[1].score


def test_hybrid_rrf_retriever() -> None:
    """Test that HybridRRFRetriever successfully merges BM25 and Dense candidate streams."""
    df = pd.DataFrame(
        [
            {
                "id": "job_1",
                "title": "Data Scientist",
                "company": "Tech Corp",
                "location": "Remote",
                "experience_level": "Mid-Senior level",
                "description": "Build statistical pipelines.",
            },
            {
                "id": "job_2",
                "title": "Backend Dev",
                "company": "Cloud Corp",
                "location": "Austin, TX",
                "experience_level": "Entry level",
                "description": "Build REST APIs.",
            },
        ]
    )
    bm25 = BM25Retriever(df=df)

    # Mock dense retriever
    dense = MagicMock()
    dense.search.return_value = [
        SearchResult(job_id="job_2", score=0.9, title="Backend Dev"),
        SearchResult(job_id="job_1", score=0.7, title="Data Scientist"),
    ]

    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_k=60)
    profile = Profile(
        target_role="Data Scientist",
        skills=["Statistics"],
        resume_text="Building data systems.",
    )

    results = hybrid.search(profile, k=2, apply_filters=False, apply_boost=False)
    assert len(results) == 2
    # Both IDs are present in the fused list
    assert {r.job_id for r in results} == {"job_1", "job_2"}


def test_cross_encoder_rerank() -> None:
    """Test CrossEncoderRetriever re-orders candidates according to cross-encoder logits."""
    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = [
        SearchResult(
            job_id="job_low_relevance",
            score=0.9,
            title="Accountant",
            company="Finance Co",
            description="Balance sheet auditing.",
        ),
        SearchResult(
            job_id="job_high_relevance",
            score=0.8,
            title="Senior ML Engineer",
            company="AI Startup",
            description="Build retrieval systems.",
        ),
    ]

    mock_cross_model = MagicMock()
    # Return higher logit (+3.0) for job_high_relevance and lower (-2.0) for job_low_relevance
    mock_cross_model.predict.return_value = np.array([-2.0, 3.0])

    reranker = CrossEncoderRetriever(
        hybrid_retriever=mock_hybrid,
        cross_encoder=mock_cross_model,
        top_candidates_depth=2,
    )

    profile = Profile(
        target_role="Senior ML Engineer",
        skills=["Retrieval"],
        resume_text="ML engineer with search background.",
    )

    reranked = reranker.search(profile, k=2, apply_filters=False, apply_boost=False)

    assert len(reranked) == 2
    # job_high_relevance must be moved to the #1 position by the cross-encoder
    assert reranked[0].job_id == "job_high_relevance"
    assert reranked[0].score > reranked[1].score
