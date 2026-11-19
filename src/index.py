"""Embedding and vector indexing pipeline using ChromaDB and SentenceTransformers.

Embeds cleaned job postings using a local sentence-transformers model
and stores document embeddings and structured metadata in a persistent
ChromaDB vector store.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Disable ChromaDB telemetry and force offline mode for pre-cached models
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.config import Settings
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_DATA_PATH = Path("data/jobs_clean.parquet")
DEFAULT_CHROMA_DIR = Path("chroma_db")
DEFAULT_STATS_PATH = Path("reports/index_stats.json")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "jobs"
DEFAULT_BATCH_SIZE = 128


def compute_content_hash(title: str, description: str) -> str:
    """Compute deterministic SHA-256 hash for document content.

    Used to track content integrity, identify document revisions, and deduplicate.

    Args:
        title: Normalized job title.
        description: Normalized job description text.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    raw_payload = f"{title.strip().lower()}||{description.strip().lower()}"
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


def format_passage(
    title: str,
    company: str,
    location: str,
    experience_level: str,
    description: str,
) -> str:
    """Format structured job fields into a single passage for dense embedding.

    Includes key metadata prefixes to improve semantic retrieval quality across
    role titles, requirements, seniority, and location constraints.

    Args:
        title: Job title.
        company: Company name.
        location: Normalized location string.
        experience_level: Standardized seniority level.
        description: Full cleaned job description text.

    Returns:
        Formatted passage string.
    """
    return (
        f"Job Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Experience Level: {experience_level}\n\n"
        f"Description:\n{description}"
    )


def load_cleaned_postings(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load cleaned job postings from parquet format.

    Args:
        data_path: Path to the clean jobs parquet file.

    Returns:
        Cleaned pandas DataFrame.

    Raises:
        FileNotFoundError: If the parquet file is not found.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {data_path}. "
            "Please run `src/ingest.py` first to generate cleaned data."
        )
    return pd.read_parquet(data_path)


def initialize_chroma_collection(
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset_existing: bool = True,
) -> Tuple[chromadb.ClientAPI, chromadb.Collection]:
    """Initialize persistent ChromaDB client and retrieve or reset the collection.

    Args:
        persist_dir: Directory where ChromaDB will persist SQLite and vector files.
        collection_name: Name of the vector collection.
        reset_existing: Whether to delete and recreate the collection if it exists.

    Returns:
        Tuple of (chroma_client, collection).
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(anonymized_telemetry=False, is_persistent=True)
    client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)

    if reset_existing:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def index_postings(
    df: pd.DataFrame,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Encode job postings and insert documents, embeddings, and metadata into ChromaDB.

    Args:
        df: Cleaned postings DataFrame.
        collection: Target ChromaDB collection.
        model: Loaded SentenceTransformer model.
        batch_size: Batch size for encoding and insertion.

    Returns:
        Dictionary of indexing metrics (elapsed time, counts, dimensions).
    """
    total_docs = len(df)
    print(f"Preparing {total_docs} job postings for embedding...")

    start_time = time.time()
    total_batches = (total_docs + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="Indexing batches"):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, total_docs)
        batch_df = df.iloc[start_i:end_i]

        passages: List[str] = [
            format_passage(
                title=row["title"],
                company=row["company"],
                location=row["location"],
                experience_level=row["experience_level"],
                description=row["description"],
            )
            for _, row in batch_df.iterrows()
        ]

        # Generate embeddings
        embeddings = model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        ids: List[str] = [str(row["id"]) for _, row in batch_df.iterrows()]

        metadatas: List[Dict[str, Any]] = [
            {
                "job_id": str(row["id"]),
                "title": str(row["title"]),
                "company": str(row["company"]),
                "location": str(row["location"]),
                "experience_level": str(row["experience_level"]),
                "content_hash": compute_content_hash(
                    row["title"], row["description"]
                ),
            }
            for _, row in batch_df.iterrows()
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=passages,
            metadatas=metadatas,
        )

    elapsed_time = round(time.time() - start_time, 2)
    embedding_dim = model.get_sentence_embedding_dimension()

    metrics = {
        "total_documents_indexed": total_docs,
        "elapsed_seconds": elapsed_time,
        "docs_per_second": (
            round(total_docs / elapsed_time, 2) if elapsed_time > 0 else 0
        ),
        "embedding_dimension": embedding_dim,
    }
    return metrics


def run_indexing(
    data_path: Path = DEFAULT_DATA_PATH,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    stats_path: Path = DEFAULT_STATS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Execute the full indexing pipeline: load data, embed passages, store in ChromaDB.

    Args:
        data_path: Path to clean jobs parquet.
        persist_dir: ChromaDB persistence folder.
        stats_path: Output path for indexing report.
        model_name: Hugging Face model identifier for SentenceTransformer.
        collection_name: ChromaDB collection name.
        batch_size: Batch size for embedding and ingestion.

    Returns:
        Summary metrics dictionary.
    """
    print(f"Loading data from {data_path}...")
    df = load_cleaned_postings(data_path)

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Initializing ChromaDB vector store at {persist_dir}...")
    _, collection = initialize_chroma_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        reset_existing=True,
    )

    metrics = index_postings(
        df=df,
        collection=collection,
        model=model,
        batch_size=batch_size,
    )

    stats = {
        "collection_name": collection_name,
        "model_name": model_name,
        "persist_directory": str(persist_dir),
        "embedding_dimension": metrics["embedding_dimension"],
        "total_documents_indexed": metrics["total_documents_indexed"],
        "index_build_time_seconds": metrics["elapsed_seconds"],
        "indexing_throughput_docs_sec": metrics["docs_per_second"],
        "metadata_fields": [
            "job_id",
            "title",
            "company",
            "location",
            "experience_level",
            "content_hash",
        ],
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Indexing complete! Stats saved to {stats_path}:")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    run_indexing()
