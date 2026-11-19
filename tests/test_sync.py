"""Unit tests for hash-based incremental vector index synchronization."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import pytest

from src.sync_index import sync_index


class FakeEmbedder:
    """Mock embedding model that counts encoding calls and returns static unit vectors."""

    def __init__(self, dim: int = 4) -> None:
        """Initialize fake embedder with specified dimension."""
        self.dim = dim
        self.call_count = 0
        self.encoded_passages: List[str] = []

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Record encoded sentences and return deterministic unit vectors."""
        self.call_count += 1
        self.encoded_passages.extend(sentences)
        vectors = []
        for i, text in enumerate(sentences):
            vec = np.zeros(self.dim, dtype=np.float32)
            # deterministic vector based on text length
            vec[i % self.dim] = 1.0
            vectors.append(vec)
        return np.array(vectors)


@pytest.fixture
def temp_chroma_collection():
    """Create a clean isolated in-memory ChromaDB collection for fast unit testing."""
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    col_name = f"sync_test_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )
    yield collection
    try:
        client.delete_collection(name=col_name)
    except Exception:
        pass


def test_sync_unchanged_postings_skipped_with_zero_embedding_calls(
    temp_chroma_collection,
) -> None:
    """Test that identical postings are skipped with exactly 0 calls to the embedder."""
    df_initial = pd.DataFrame(
        [
            {
                "id": "1",
                "title": "Data Scientist",
                "company": "Alpha Corp",
                "location": "Boston, MA",
                "experience_level": "Entry level",
                "description": "Build classification models.",
            },
            {
                "id": "2",
                "title": "ML Engineer",
                "company": "Beta Inc",
                "location": "Remote",
                "experience_level": "Mid-Senior level",
                "description": "Deploy deep learning endpoints.",
            },
        ]
    )

    embedder = FakeEmbedder()
    # Initial load: 2 added
    initial_counts = sync_index(
        df=df_initial,
        collection=temp_chroma_collection,
        model=embedder,
        show_progress=False,
    )
    assert initial_counts == {"added": 2, "updated": 0, "skipped": 0, "deleted": 0}
    assert embedder.call_count == 1

    # Second sync with identical data: must skip with 0 calls to embedder
    embedder_second = FakeEmbedder()
    second_counts = sync_index(
        df=df_initial,
        collection=temp_chroma_collection,
        model=embedder_second,
        show_progress=False,
    )

    assert second_counts == {"added": 0, "updated": 0, "skipped": 2, "deleted": 0}
    assert embedder_second.call_count == 0  # CRITICAL: zero embedding overhead!


def test_sync_adds_new_posting(temp_chroma_collection) -> None:
    """Test that a new posting in snapshot is embedded and added."""
    df_initial = pd.DataFrame(
        [
            {
                "id": "1",
                "title": "Backend Dev",
                "company": "Acme",
                "location": "Austin, TX",
                "experience_level": "Entry level",
                "description": "Write FastAPI services.",
            }
        ]
    )
    embedder = FakeEmbedder()
    sync_index(df_initial, temp_chroma_collection, embedder, show_progress=False)

    df_updated = pd.DataFrame(
        [
            {
                "id": "1",
                "title": "Backend Dev",
                "company": "Acme",
                "location": "Austin, TX",
                "experience_level": "Entry level",
                "description": "Write FastAPI services.",
            },
            {
                "id": "2",
                "title": "Frontend Dev",
                "company": "Acme",
                "location": "Austin, TX",
                "experience_level": "Entry level",
                "description": "Write React components.",
            },
        ]
    )

    embedder_2 = FakeEmbedder()
    counts = sync_index(
        df_updated, temp_chroma_collection, embedder_2, show_progress=False
    )

    assert counts == {"added": 1, "updated": 0, "skipped": 1, "deleted": 0}
    assert embedder_2.call_count == 1
    assert len(embedder_2.encoded_passages) == 1
    assert "Frontend Dev" in embedder_2.encoded_passages[0]
    assert temp_chroma_collection.count() == 2


def test_sync_updates_changed_posting(temp_chroma_collection) -> None:
    """Test that modified fields trigger re-embedding and hash update."""
    df_initial = pd.DataFrame(
        [
            {
                "id": "100",
                "title": "Software Engineer",
                "company": "Stripe",
                "location": "Seattle, WA",
                "experience_level": "Entry level",
                "description": "Version 1 description.",
            }
        ]
    )
    embedder = FakeEmbedder()
    sync_index(df_initial, temp_chroma_collection, embedder, show_progress=False)
    old_hash = temp_chroma_collection.get(ids=["100"])["metadatas"][0]["content_hash"]

    # Modify description
    df_modified = pd.DataFrame(
        [
            {
                "id": "100",
                "title": "Software Engineer",
                "company": "Stripe",
                "location": "Seattle, WA",
                "experience_level": "Entry level",
                "description": "Version 2 description with updated requirements.",
            }
        ]
    )
    embedder_2 = FakeEmbedder()
    counts = sync_index(
        df_modified, temp_chroma_collection, embedder_2, show_progress=False
    )

    assert counts == {"added": 0, "updated": 1, "skipped": 0, "deleted": 0}
    assert embedder_2.call_count == 1
    new_hash = temp_chroma_collection.get(ids=["100"])["metadatas"][0]["content_hash"]
    assert new_hash != old_hash


def test_sync_deletes_removed_posting(temp_chroma_collection) -> None:
    """Test that missing postings from snapshot are removed from ChromaDB."""
    df_initial = pd.DataFrame(
        [
            {
                "id": "10",
                "title": "Role A",
                "company": "Company",
                "location": "Remote",
                "experience_level": "Mid-Senior level",
                "description": "Desc A.",
            },
            {
                "id": "20",
                "title": "Role B",
                "company": "Company",
                "location": "Remote",
                "experience_level": "Mid-Senior level",
                "description": "Desc B.",
            },
        ]
    )
    embedder = FakeEmbedder()
    sync_index(df_initial, temp_chroma_collection, embedder, show_progress=False)
    assert temp_chroma_collection.count() == 2

    # Snapshot without ID 20
    df_pruned = pd.DataFrame(
        [
            {
                "id": "10",
                "title": "Role A",
                "company": "Company",
                "location": "Remote",
                "experience_level": "Mid-Senior level",
                "description": "Desc A.",
            }
        ]
    )
    embedder_2 = FakeEmbedder()
    counts = sync_index(
        df_pruned, temp_chroma_collection, embedder_2, show_progress=False
    )

    assert counts == {"added": 0, "updated": 0, "skipped": 1, "deleted": 1}
    assert embedder_2.call_count == 0  # No embedding calls needed for pure deletion
    assert temp_chroma_collection.count() == 1
    assert temp_chroma_collection.get(ids=["20"])["ids"] == []


def test_sync_combined_lifecycle(temp_chroma_collection) -> None:
    """Test simultaneous added, updated, skipped, and deleted postings."""
    # Initial state: [A, B, C]
    df_initial = pd.DataFrame(
        [
            {
                "id": "A",
                "title": "Job A",
                "company": "Co",
                "location": "NY",
                "experience_level": "Entry level",
                "description": "Desc A",
            },
            {
                "id": "B",
                "title": "Job B",
                "company": "Co",
                "location": "SF",
                "experience_level": "Entry level",
                "description": "Desc B",
            },
            {
                "id": "C",
                "title": "Job C",
                "company": "Co",
                "location": "LA",
                "experience_level": "Entry level",
                "description": "Desc C",
            },
        ]
    )
    sync_index(
        df_initial, temp_chroma_collection, FakeEmbedder(), show_progress=False
    )

    # Next state:
    # A -> unchanged (skip)
    # B -> modified description (update)
    # C -> missing (delete)
    # D -> brand new (add)
    df_next = pd.DataFrame(
        [
            {
                "id": "A",
                "title": "Job A",
                "company": "Co",
                "location": "NY",
                "experience_level": "Entry level",
                "description": "Desc A",
            },
            {
                "id": "B",
                "title": "Job B",
                "company": "Co",
                "location": "SF",
                "experience_level": "Entry level",
                "description": "Desc B - Modified!",
            },
            {
                "id": "D",
                "title": "Job D",
                "company": "Co",
                "location": "Chicago",
                "experience_level": "Entry level",
                "description": "Desc D - New!",
            },
        ]
    )

    embedder = FakeEmbedder()
    counts = sync_index(
        df_next, temp_chroma_collection, embedder, show_progress=False
    )

    assert counts == {"added": 1, "updated": 1, "skipped": 1, "deleted": 1}
    assert embedder.call_count == 1
    assert len(embedder.encoded_passages) == 2  # exactly B and D encoded

    current_ids = set(temp_chroma_collection.get()["ids"])
    assert current_ids == {"A", "B", "D"}

