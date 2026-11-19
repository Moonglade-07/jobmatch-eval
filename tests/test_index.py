"""Unit tests for embedding, hashing, and ChromaDB vector indexing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Disable Chroma telemetry during testing
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import numpy as np
import pandas as pd
import pytest

from src.index import (
    compute_content_hash,
    format_passage,
    index_postings,
    initialize_chroma_collection,
)


def test_compute_content_hash_deterministic() -> None:
    """Test that content hashing is deterministic and covers all embedding fields."""
    title = "Machine Learning Engineer"
    desc = "Build scalable recommendation systems."
    loc = "San Francisco, CA"
    lvl = "Mid-Senior level"

    hash1 = compute_content_hash(title, desc, loc, lvl)
    hash2 = compute_content_hash(
        title.lower(), desc.lower(), loc.lower(), lvl.lower()
    )
    hash_diff_desc = compute_content_hash(
        title, desc + " Additional text", loc, lvl
    )
    hash_diff_loc = compute_content_hash(title, desc, "New York, NY", lvl)
    hash_diff_lvl = compute_content_hash(title, desc, loc, "Entry level")

    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 produces 64 hex characters
    assert hash1 == hash2
    assert hash1 != hash_diff_desc
    assert hash1 != hash_diff_loc
    assert hash1 != hash_diff_lvl


def test_format_passage() -> None:
    """Test passage formatting structure includes all relevant fields."""
    passage = format_passage(
        title="Senior NLP Engineer",
        company="OpenAI Ecosystem Partner",
        location="San Francisco, CA",
        experience_level="Mid-Senior level",
        description="Design retrieval pipelines and vector search systems.",
    )

    assert "Job Title: Senior NLP Engineer" in passage
    assert "Company: OpenAI Ecosystem Partner" in passage
    assert "Location: San Francisco, CA" in passage
    assert "Experience Level: Mid-Senior level" in passage
    assert "Description:\nDesign retrieval pipelines" in passage


def test_chroma_indexing_and_query(tmp_path: Path) -> None:
    """Test end-to-end vector indexing and querying using ChromaDB in a temporary path."""
    client, collection = initialize_chroma_collection(
        persist_dir=tmp_path / "test_chroma",
        collection_name="test_jobs",
        reset_existing=True,
    )

    sample_data = [
        {
            "id": "job_001",
            "title": "NLP Engineer",
            "company": "AI Labs",
            "location": "San Francisco, CA",
            "experience_level": "Mid-Senior level",
            "description": "Develop transformer models and vector indexing.",
        },
        {
            "id": "job_002",
            "title": "Account Executive",
            "company": "SalesHub",
            "location": "New York, NY",
            "experience_level": "Entry level",
            "description": "Manage outbound enterprise SaaS sales pipeline.",
        },
    ]
    df = pd.DataFrame(sample_data)

    # Use a lightweight mock embedding model for fast unit testing
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 4
    # Provide distinct mock vectors: job 1 is [1, 0, 0, 0], job 2 is [0, 1, 0, 0]
    mock_model.encode.return_value = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )

    metrics = index_postings(
        df=df,
        collection=collection,
        model=mock_model,
        batch_size=2,
    )

    assert metrics["total_documents_indexed"] == 2
    assert metrics["embedding_dimension"] == 4

    # Query ChromaDB for vector similar to job 1
    results = collection.query(
        query_embeddings=[[1.0, 0.0, 0.0, 0.0]],
        n_results=1,
        include=["metadatas", "documents"],
    )

    assert results["ids"][0][0] == "job_001"
    matched_metadata = results["metadatas"][0][0]
    assert matched_metadata["title"] == "NLP Engineer"
    assert matched_metadata["location"] == "San Francisco, CA"
    assert matched_metadata["experience_level"] == "Mid-Senior level"
    assert "content_hash" in matched_metadata
