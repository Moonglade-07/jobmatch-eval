"""Incremental vector index synchronization with content hash-based change detection.

Computes SHA-256 hashes over title, description, location, and seniority level.
Skips embedding calls for unchanged postings, re-embeds updated postings,
adds new postings, and removes deleted postings from ChromaDB.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enforce offline operation and disable external telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.index import compute_content_hash, format_passage


def get_existing_index_state(
    collection: chromadb.Collection,
) -> Dict[str, str]:
    """Retrieve mapping of document ID to content hash from ChromaDB.

    Args:
        collection: Active ChromaDB collection.

    Returns:
        Dictionary mapping job_id (str) to existing content_hash (str).
    """
    existing_data = collection.get(include=["metadatas"])
    ids = existing_data.get("ids", [])
    metadatas = existing_data.get("metadatas", [])

    id_to_hash: Dict[str, str] = {}
    for doc_id, meta in zip(ids, metadatas):
        if meta and isinstance(meta, dict):
            id_to_hash[str(doc_id)] = str(meta.get("content_hash", ""))
        else:
            id_to_hash[str(doc_id)] = ""
    return id_to_hash


def categorize_sync_actions(
    df: pd.DataFrame,
    existing_state: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, List[str]]:
    """Compare incoming snapshot against existing index state to determine operations.

    Args:
        df: New snapshot DataFrame containing job postings.
        existing_state: Dictionary of existing document IDs to their content hashes.

    Returns:
        Tuple of (items_to_add, items_to_update, skipped_count, ids_to_delete).
    """
    to_add: List[Dict[str, Any]] = []
    to_update: List[Dict[str, Any]] = []
    skipped_count: int = 0
    incoming_ids = set()

    for _, row in df.iterrows():
        doc_id = str(row["id"])
        incoming_ids.add(doc_id)

        title = str(row["title"])
        desc = str(row["description"])
        loc = str(row["location"])
        lvl = str(row["experience_level"])
        company = str(row.get("company", "Unknown Company"))

        content_hash = compute_content_hash(
            title=title,
            description=desc,
            location=loc,
            experience_level=lvl,
        )

        item = {
            "id": doc_id,
            "title": title,
            "description": desc,
            "location": loc,
            "experience_level": lvl,
            "company": company,
            "content_hash": content_hash,
        }

        if doc_id not in existing_state:
            to_add.append(item)
        elif existing_state[doc_id] != content_hash:
            to_update.append(item)
        else:
            skipped_count += 1

    to_delete = [
        doc_id for doc_id in existing_state if doc_id not in incoming_ids
    ]
    return to_add, to_update, skipped_count, to_delete


def sync_index(
    df: pd.DataFrame,
    collection: chromadb.Collection,
    model: Any,
    batch_size: int = 128,
    show_progress: bool = True,
) -> Dict[str, int]:
    """Synchronize ChromaDB vector store with the provided postings snapshot.

    Only items marked as added or updated are passed to the embedding model.
    Unchanged items are skipped with zero embedding overhead. Items missing
    from the snapshot are removed from the vector store.

    Args:
        df: Postings DataFrame to synchronize.
        collection: Target ChromaDB collection.
        model: SentenceTransformer or compatible encoder instance with .encode().
        batch_size: Batch size for embedding and upsertion.
        show_progress: Whether to display a progress bar.

    Returns:
        Dictionary of operation counts: {'added': int, 'updated': int, 'skipped': int, 'deleted': int}.
    """
    existing_state = get_existing_index_state(collection)

    to_add, to_update, skipped_count, to_delete = categorize_sync_actions(
        df=df,
        existing_state=existing_state,
    )

    # 1. Perform Deletions
    if to_delete:
        collection.delete(ids=to_delete)

    # 2. Perform Embeddings & Upserts only for changed and new documents
    docs_to_upsert = to_add + to_update
    if docs_to_upsert:
        total = len(docs_to_upsert)
        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding & upserting changes")

        for start_idx in iterator:
            batch = docs_to_upsert[start_idx : start_idx + batch_size]
            passages = [
                format_passage(
                    title=item["title"],
                    company=item["company"],
                    location=item["location"],
                    experience_level=item["experience_level"],
                    description=item["description"],
                )
                for item in batch
            ]

            embeddings = model.encode(
                passages,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

            batch_ids = [item["id"] for item in batch]
            batch_metadatas = [
                {
                    "job_id": item["id"],
                    "title": item["title"],
                    "company": item["company"],
                    "location": item["location"],
                    "experience_level": item["experience_level"],
                    "content_hash": item["content_hash"],
                }
                for item in batch
            ]

            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=passages,
                metadatas=batch_metadatas,
            )

    counts = {
        "added": len(to_add),
        "updated": len(to_update),
        "skipped": skipped_count,
        "deleted": len(to_delete),
    }

    print(
        f"Sync complete -> Added: {counts['added']}, "
        f"Updated: {counts['updated']}, "
        f"Skipped: {counts['skipped']}, "
        f"Deleted: {counts['deleted']}"
    )
    return counts

