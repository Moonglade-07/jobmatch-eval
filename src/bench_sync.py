"""Benchmark comparing full re-indexing vs hash-based incremental synchronization.

Simulates a realistic update cycle (5% modified, 2% added, 2% deleted),
times full re-index vs incremental sync over 3 repetitions, computes speedup,
and verifies state equivalence (IDs + content hashes) against a fresh full index.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enforce offline mode and disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.index import (
    DEFAULT_MODEL_NAME,
    compute_content_hash,
    format_passage,
)
from src.sync_index import sync_index


DEFAULT_DATA_PATH = Path("data/jobs_clean.parquet")
DEFAULT_REPORT_PATH = Path("reports/sync_benchmark.json")
BENCH_CHROMA_DIR = Path("chroma_db_bench")
BENCH_BATCH_SIZE = 128
DEFAULT_SEED = 42
NUM_RUNS = 3


def simulate_dataset_changes(
    df: pd.DataFrame,
    modify_pct: float = 0.05,
    add_pct: float = 0.02,
    delete_pct: float = 0.02,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Reproducibly simulate changes in job postings.

    Deletes delete_pct of postings, modifies modify_pct of descriptions,
    and appends add_pct of new postings.

    Args:
        df: Baseline cleaned DataFrame.
        modify_pct: Fraction of descriptions to alter.
        add_pct: Fraction of new postings to create.
        delete_pct: Fraction of postings to remove.
        seed: Deterministic random seed.

    Returns:
        Tuple of (mutated DataFrame snapshot, change statistics dict).
    """
    rng = np.random.RandomState(seed)
    total_docs = len(df)
    n_delete = int(total_docs * delete_pct)
    n_modify = int(total_docs * modify_pct)
    n_add = int(total_docs * add_pct)

    indices = rng.permutation(total_docs)
    delete_indices = set(indices[:n_delete])
    remaining_indices = [i for i in indices if i not in delete_indices]

    modify_indices = set(remaining_indices[:n_modify])

    mutated_rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        if idx in delete_indices:
            continue
        row_dict = row.to_dict()
        if idx in modify_indices:
            row_dict["description"] = (
                str(row_dict["description"])
                + " [Updated: Requires experience with distributed systems and evaluation.]"
            )
        mutated_rows.append(row_dict)

    # Generate new postings
    for i in range(n_add):
        template_idx = remaining_indices[i % len(remaining_indices)]
        tmpl = df.iloc[template_idx]
        new_item = {
            "id": f"new_posting_{seed}_{i+1:04d}",
            "title": f"Emerging Role - {tmpl['title']}",
            "company": f"NextGen {tmpl['company']}",
            "location": tmpl["location"],
            "experience_level": tmpl["experience_level"],
            "description": (
                f"Newly created position in AI engineering: {tmpl['description'][:300]} "
                "Focus on LLM alignment, evaluation harness development, and search ranking."
            ),
        }
        mutated_rows.append(new_item)

    df_mutated = pd.DataFrame(mutated_rows).reset_index(drop=True)
    change_stats = {
        "original_count": total_docs,
        "modified_count": n_modify,
        "added_count": n_add,
        "deleted_count": n_delete,
        "unchanged_count": total_docs - n_delete - n_modify,
        "mutated_total_count": len(df_mutated),
    }
    return df_mutated, change_stats


def execute_full_reindex(
    df: pd.DataFrame,
    client: chromadb.ClientAPI,
    collection_name: str,
    model: SentenceTransformer,
    batch_size: int = BENCH_BATCH_SIZE,
) -> float:
    """Execute a complete re-index from scratch and measure elapsed time.

    Args:
        df: DataFrame snapshot to index.
        client: ChromaDB client.
        collection_name: Name of collection.
        model: SentenceTransformer model.
        batch_size: Batch size for encoding.

    Returns:
        Elapsed time in seconds.
    """
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    t0 = time.time()
    total = len(df)
    for start_i in range(0, total, batch_size):
        end_i = min(start_i + batch_size, total)
        batch_df = df.iloc[start_i:end_i]

        passages = [
            format_passage(
                title=row["title"],
                company=row["company"],
                location=row["location"],
                experience_level=row["experience_level"],
                description=row["description"],
            )
            for _, row in batch_df.iterrows()
        ]

        with torch.inference_mode():
            embeddings = model.encode(
                passages,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

        ids = [str(row["id"]) for _, row in batch_df.iterrows()]
        metadatas = [
            {
                "job_id": str(row["id"]),
                "title": str(row["title"]),
                "company": str(row["company"]),
                "location": str(row["location"]),
                "experience_level": str(row["experience_level"]),
                "content_hash": compute_content_hash(
                    title=str(row["title"]),
                    description=str(row["description"]),
                    location=str(row["location"]),
                    experience_level=str(row["experience_level"]),
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

    return round(time.time() - t0, 3)


def execute_incremental_sync(
    df_mutated: pd.DataFrame,
    client: chromadb.ClientAPI,
    collection_name: str,
    model: SentenceTransformer,
) -> Tuple[float, Dict[str, int]]:
    """Execute incremental synchronization on an existing collection and measure elapsed time.

    Args:
        df_mutated: Modified DataFrame snapshot.
        client: ChromaDB client.
        collection_name: Target collection.
        model: SentenceTransformer model.

    Returns:
        Tuple of (elapsed time in seconds, sync counts dict).
    """
    collection = client.get_collection(name=collection_name)
    t0 = time.time()
    counts = sync_index(
        df=df_mutated,
        collection=collection,
        model=model,
        batch_size=BENCH_BATCH_SIZE,
        show_progress=False,
    )
    elapsed = round(time.time() - t0, 3)
    return elapsed, counts


def verify_index_equivalence(
    col_synced: chromadb.Collection,
    col_full: chromadb.Collection,
) -> Dict[str, Any]:
    """Verify that a synced index is identical in content and metadata to a fresh index.

    Compares sorted document IDs and content hashes between both collections.

    Args:
        col_synced: Collection updated via incremental sync.
        col_full: Collection created via fresh full re-index.

    Returns:
        Verification dictionary with boolean passed status and discrepancy metrics.
    """
    synced_data = col_synced.get(include=["metadatas"])
    full_data = col_full.get(include=["metadatas"])

    synced_map = {
        str(doc_id): str(meta.get("content_hash", ""))
        for doc_id, meta in zip(synced_data["ids"], synced_data["metadatas"])
    }
    full_map = {
        str(doc_id): str(meta.get("content_hash", ""))
        for doc_id, meta in zip(full_data["ids"], full_data["metadatas"])
    }

    synced_ids = sorted(synced_map.keys())
    full_ids = sorted(full_map.keys())

    id_match = synced_ids == full_ids
    hash_mismatches = 0
    if id_match:
        for doc_id in synced_ids:
            if synced_map[doc_id] != full_map[doc_id]:
                hash_mismatches += 1

    passed = id_match and (hash_mismatches == 0)

    return {
        "passed": bool(passed),
        "synced_document_count": len(synced_ids),
        "full_reindex_document_count": len(full_ids),
        "id_sets_identical": bool(id_match),
        "hash_mismatches_count": hash_mismatches,
    }


def run_benchmark(
    data_path: Path = DEFAULT_DATA_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    bench_dir: Path = BENCH_CHROMA_DIR,
    num_runs: int = NUM_RUNS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Run full benchmark comparing full re-index against incremental synchronization.

    Args:
        data_path: Path to baseline clean parquet.
        report_path: Destination for sync_benchmark.json report.
        bench_dir: Isolated folder for benchmark vector stores.
        num_runs: Number of timing repetitions for each strategy.
        seed: Random seed for dataset mutations.

    Returns:
        Benchmark report dictionary.
    """
    print(f"Loading baseline dataset from {data_path}...")
    df_initial = pd.read_parquet(data_path)
    total_docs = len(df_initial)
    print(f"Baseline dataset loaded: {total_docs} postings.")

    print(
        f"Simulating changes (5% modified, 2% added, 2% deleted) with seed={seed}..."
    )
    df_mutated, change_stats = simulate_dataset_changes(
        df=df_initial,
        modify_pct=0.05,
        add_pct=0.02,
        delete_pct=0.02,
        seed=seed,
    )
    print(
        f"Mutated snapshot prepared: {change_stats['mutated_total_count']} postings "
        f"({change_stats['modified_count']} modified, "
        f"{change_stats['added_count']} added, "
        f"{change_stats['deleted_count']} deleted)."
    )

    if bench_dir.exists():
        shutil.rmtree(bench_dir, ignore_errors=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(bench_dir),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    print(f"Loading embedding model: {DEFAULT_MODEL_NAME}...")
    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    # 1. Benchmark Incremental Sync
    print(f"\n--- Running Incremental Sync Benchmark ({num_runs} runs) ---")
    sync_times: List[float] = []
    last_sync_counts: Dict[str, int] = {}

    for run_idx in range(num_runs):
        # Prepare baseline collection state for this run
        col_name = f"sync_col_run_{run_idx}"
        execute_full_reindex(df_initial, client, col_name, model)

        elapsed, counts = execute_incremental_sync(
            df_mutated, client, col_name, model
        )
        sync_times.append(elapsed)
        last_sync_counts = counts
        print(f"  Sync Run {run_idx + 1}/{num_runs}: {elapsed:.3f} s")

    # 2. Benchmark Full Re-index
    print(f"\n--- Running Full Re-index Benchmark ({num_runs} runs) ---")
    full_times: List[float] = []

    for run_idx in range(num_runs):
        col_name = f"full_col_run_{run_idx}"
        elapsed = execute_full_reindex(df_mutated, client, col_name, model)
        full_times.append(elapsed)
        print(f"  Full Re-index Run {run_idx + 1}/{num_runs}: {elapsed:.3f} s")

    # 3. State Equivalence Verification
    print("\n--- Verifying Index State Equivalence ---")
    col_synced = client.get_collection(name="sync_col_run_0")
    col_full = client.get_collection(name="full_col_run_0")
    verification = verify_index_equivalence(col_synced, col_full)
    print(f"  State equivalence check passed: {verification['passed']}")
    print(
        f"  Synced IDs == Full IDs: {verification['id_sets_identical']}, "
        f"Hash mismatches: {verification['hash_mismatches_count']}"
    )

    # 4. Compute Metrics
    sync_mean = round(statistics.mean(sync_times), 3)
    sync_std = round(statistics.stdev(sync_times), 3) if len(sync_times) > 1 else 0.0

    full_mean = round(statistics.mean(full_times), 3)
    full_std = round(statistics.stdev(full_times), 3) if len(full_times) > 1 else 0.0

    speedup = round(full_mean / sync_mean, 2) if sync_mean > 0 else 0.0

    benchmark_report: Dict[str, Any] = {
        "dataset_baseline_size": total_docs,
        "mutation_parameters": {
            "modify_percentage": 5.0,
            "add_percentage": 2.0,
            "delete_percentage": 2.0,
            "random_seed": seed,
        },
        "change_counts": change_stats,
        "sync_operation_counts": last_sync_counts,
        "benchmark_runs": num_runs,
        "full_reindex": {
            "run_times_seconds": full_times,
            "mean_seconds": full_mean,
            "std_seconds": full_std,
            "throughput_docs_per_sec": (
                round(change_stats["mutated_total_count"] / full_mean, 2)
                if full_mean > 0
                else 0.0
            ),
        },
        "incremental_sync": {
            "run_times_seconds": sync_times,
            "mean_seconds": sync_mean,
            "std_seconds": sync_std,
            "throughput_effective_docs_per_sec": (
                round(change_stats["mutated_total_count"] / sync_mean, 2)
                if sync_mean > 0
                else 0.0
            ),
        },
        "speedup_factor": speedup,
        "state_equivalence_verification": verification,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\nBenchmark complete! Results saved to {report_path}:")
    print(f"  Full Re-index Mean: {full_mean} s (std: {full_std} s)")
    print(f"  Incremental Sync Mean: {sync_mean} s (std: {sync_std} s)")
    print(f"  Speedup Factor: {speedup}x")
    print(f"  Verification Status: {'PASSED' if verification['passed'] else 'FAILED'}")

    # Clean up isolated bench directory to avoid bloating workspace
    try:
        shutil.rmtree(bench_dir, ignore_errors=True)
    except Exception:
        pass

    return benchmark_report


if __name__ == "__main__":
    run_benchmark()
