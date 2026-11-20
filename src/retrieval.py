"""Retrieval strategies for job matching: BM25, Dense, Hybrid (RRF), and Cross-Encoder Re-Ranking.

Provides a unified interface `search(profile: Profile, k: int) -> List[SearchResult]`
implementing:
  1. BM25Retriever: Lexical keyword search via rank_bm25.
  2. DenseRetriever: Semantic vector search via ChromaDB and all-MiniLM-L6-v2.
  3. HybridRRFRetriever: Reciprocal Rank Fusion combining BM25 and Dense (param k=60).
  4. CrossEncoderRetriever: Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) over top 50 fused items.
"""

from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enforce offline mode and disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer
    import torch

from src.index import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME, DEFAULT_MODEL_NAME, format_passage
from src.models import Profile, SearchResult
from src.personalise import (
    apply_skill_boost,
    build_query_text,
    filter_with_fallback,
    is_remote_posting,
)


TOKEN_REGEX = re.compile(
    r"(?:\.[a-zA-Z0-9_\-]+|[a-zA-Z0-9_\-]+(?:\.[a-zA-Z0-9_\-]+)+|[a-zA-Z0-9_\-]+(?:\+\+|#)?|[a-zA-Z0-9_\-]+)"
)
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DATA_PATH = Path("data/jobs_clean.parquet")


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercased terms for lexical indexing and querying.

    Preserves domain tokens including technical symbols (e.g. 'c++', 'c#', '.net', 'python-3').

    Args:
        text: Raw input text string.

    Returns:
        List of alphanumeric and programming token strings.
    """
    if not text or not isinstance(text, str):
        return []
    return [match.group(0).lower() for match in TOKEN_REGEX.finditer(text)]


def reciprocal_rank_fusion(
    ranked_lists: List[List[SearchResult]],
    rrf_k: int = 60,
) -> List[SearchResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(doc) = sum_{list in ranked_lists} (1.0 / (rrf_k + rank_{list}(doc)))

    Rank is 1-indexed. The parameter `rrf_k` smooths the impact of high-ranking items.

    Args:
        ranked_lists: Collection of ranked SearchResult lists from individual retrievers.
        rrf_k: Smoothing constant parameter (standard default: 60).

    Returns:
        Consolidated, deduplicated SearchResult list ordered descending by RRF score.
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, SearchResult] = {}

    for result_list in ranked_lists:
        for rank_0, item in enumerate(result_list):
            rank = rank_0 + 1
            doc_id = item.job_id
            rrf_delta = 1.0 / (rrf_k + rank)

            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_delta
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = item

    fused_results: List[SearchResult] = []
    for doc_id, fused_score in sorted(
        scores.items(), key=lambda kv: kv[1], reverse=True
    ):
        base_item = doc_lookup[doc_id]
        fused_item = SearchResult(
            job_id=base_item.job_id,
            score=round(fused_score, 6),
            title=base_item.title,
            company=base_item.company,
            location=base_item.location,
            experience_level=base_item.experience_level,
            description=base_item.description,
            matched_skills=list(base_item.matched_skills),
            base_score=round(fused_score, 6),
            raw_score=round(fused_score, 6),
            filter_tier=base_item.filter_tier,
        )
        fused_results.append(fused_item)

    return fused_results


class BaseRetriever(ABC):
    """Abstract base class establishing the common search interface for all retrieval strategies."""

    @abstractmethod
    def search(
        self,
        profile: Profile,
        k: int = 10,
        apply_filters: bool = True,
        apply_boost: bool = True,
        boost_weight: float = 0.2,
    ) -> List[SearchResult]:
        """Retrieve and rank job postings for a candidate profile.

        Args:
            profile: Candidate profile.
            k: Number of ranked postings to return.
            apply_filters: Whether to apply location and seniority metadata filtering with fallback.
            apply_boost: Whether to apply explainable skill boost.
            boost_weight: Weight parameter for matched skills boost.

        Returns:
            Ranked list of SearchResult instances (supports tuple unpacking: job_id, score).
        """
        pass


class BM25Retriever(BaseRetriever):
    """Lexical keyword retriever implementing BM25Okapi over job postings."""

    def __init__(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize BM25 index over clean job postings.

        Args:
            data_path: Path to parquet dataset if df is not provided directly.
            df: Optional pre-loaded DataFrame.
        """
        if df is not None:
            self.df = df.copy()
        else:
            if not data_path.exists():
                raise FileNotFoundError(f"Clean postings not found at {data_path}")
            self.df = pd.read_parquet(data_path)

        self.records: List[Dict[str, Any]] = self.df.to_dict("records")
        self.passages: List[str] = [
            format_passage(
                title=str(row["title"]),
                company=str(row["company"]),
                location=str(row["location"]),
                experience_level=str(row["experience_level"]),
                description=str(row["description"]),
            )
            for row in self.records
        ]
        self.corpus_tokens = [tokenize_text(p) for p in self.passages]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        # Pre-warm is_remote cache for fast downstream location filtering
        for row in self.records:
            is_remote_posting(
                title=str(row["title"]),
                description=str(row["description"]),
                location=str(row["location"]),
            )

    def search(
        self,
        profile: Profile,
        k: int = 10,
        apply_filters: bool = True,
        apply_boost: bool = True,
        boost_weight: float = 0.2,
    ) -> List[SearchResult]:
        """Execute BM25 keyword search for candidate profile.

        Args:
            profile: Candidate Profile.
            k: Number of ranked jobs to return.
            apply_filters: Apply location/seniority filter with fallback.
            apply_boost: Apply explainable skill boost.
            boost_weight: Skill boost weighting.

        Returns:
            Ranked list of SearchResult objects.
        """
        query_text = build_query_text(profile)
        query_tokens = tokenize_text(query_text)

        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[::-1]

        raw_results: List[SearchResult] = []
        for idx in top_indices:
            row = self.records[idx]
            raw_s = float(round(doc_scores[idx], 4))
            raw_results.append(
                SearchResult(
                    job_id=str(row["id"]),
                    score=raw_s,
                    title=str(row["title"]),
                    company=str(row["company"]),
                    location=str(row["location"]),
                    experience_level=str(row["experience_level"]),
                    description=str(row["description"]),
                    raw_score=raw_s,
                    base_score=raw_s,
                    filter_tier=1,
                )
            )

        candidate_pool_depth = max(k, 50)
        candidates = raw_results
        if apply_filters:
            candidates, _ = filter_with_fallback(
                candidates,
                preferred_location=profile.preferred_location,
                seniority=profile.seniority,
                k=candidate_pool_depth,
            )
        else:
            candidates = candidates[:candidate_pool_depth]

        if apply_boost:
            candidates = apply_skill_boost(
                candidates,
                skills=profile.skills,
                weight=boost_weight,
                normalize_base_scores=True,
            )

        return candidates[:k]


class DenseRetriever(BaseRetriever):
    """Dense semantic vector retriever using ChromaDB and SentenceTransformer."""

    def __init__(
        self,
        collection: Optional[chromadb.Collection] = None,
        model: Optional[SentenceTransformer] = None,
        persist_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Initialize Dense Retriever with ChromaDB collection and embedding model.

        Args:
            collection: Optional pre-initialized ChromaDB collection.
            model: Optional pre-loaded SentenceTransformer model.
            persist_dir: Path to ChromaDB persistence directory.
            collection_name: ChromaDB collection name.
            model_name: SentenceTransformer Hugging Face model identifier.
        """
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name)

        if collection is not None:
            self.collection = collection
        else:
            client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False, is_persistent=True),
            )
            self.collection = client.get_collection(collection_name)

    def search(
        self,
        profile: Profile,
        k: int = 10,
        apply_filters: bool = True,
        apply_boost: bool = True,
        boost_weight: float = 0.2,
    ) -> List[SearchResult]:
        """Execute dense vector similarity search for candidate profile.

        Args:
            profile: Candidate Profile.
            k: Number of ranked jobs to return.
            apply_filters: Apply location/seniority filter with fallback.
            apply_boost: Apply explainable skill boost.
            boost_weight: Skill boost weighting.

        Returns:
            Ranked list of SearchResult objects.
        """
        query_text = build_query_text(profile)

        import torch
        with torch.inference_mode():
            query_embedding = self.model.encode(
                query_text,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

        # Query a candidate pool sized appropriately for filtering or pure retrieval
        if apply_filters:
            fetch_limit = min(max(k * 4, 150), self.collection.count())
        else:
            fetch_limit = min(max(k, 50), self.collection.count())
        query_output = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_limit,
            include=["metadatas", "documents", "distances"],
        )

        ids = query_output.get("ids", [[]])[0]
        metadatas = query_output.get("metadatas", [[]])[0]
        distances = query_output.get("distances", [[]])[0]
        documents = query_output.get("documents", [[]])[0]

        raw_results: List[SearchResult] = []
        for doc_id, meta, dist, doc in zip(ids, metadatas, distances, documents):
            # Chroma HNSW cosine distance d = 1 - cosine_similarity (for normalized vectors)
            similarity = round(1.0 - float(dist), 4)

            # Extract description snippet from document passage if available
            desc = ""
            if "Description:\n" in doc:
                desc = doc.split("Description:\n", 1)[1]
            else:
                desc = doc

            raw_results.append(
                SearchResult(
                    job_id=str(doc_id),
                    score=similarity,
                    title=str(meta.get("title", "")),
                    company=str(meta.get("company", "")),
                    location=str(meta.get("location", "")),
                    experience_level=str(meta.get("experience_level", "")),
                    description=desc,
                    raw_score=similarity,
                    base_score=similarity,
                    filter_tier=1,
                )
            )

        candidate_pool_depth = max(k, 50)
        candidates = raw_results
        if apply_filters:
            candidates, _ = filter_with_fallback(
                candidates,
                preferred_location=profile.preferred_location,
                seniority=profile.seniority,
                k=candidate_pool_depth,
            )
        else:
            candidates = candidates[:candidate_pool_depth]

        if apply_boost:
            candidates = apply_skill_boost(
                candidates,
                skills=profile.skills,
                weight=boost_weight,
                normalize_base_scores=True,
            )

        return candidates[:k]


class HybridRRFRetriever(BaseRetriever):
    """Hybrid search combining BM25 lexical retrieval and dense vector retrieval via RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            bm25_retriever: Instantiated BM25Retriever.
            dense_retriever: Instantiated DenseRetriever.
            rrf_k: Reciprocal rank fusion smoothing parameter.
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        profile: Profile,
        k: int = 10,
        apply_filters: bool = True,
        apply_boost: bool = True,
        boost_weight: float = 0.2,
    ) -> List[SearchResult]:
        """Execute hybrid search using Reciprocal Rank Fusion.

        Args:
            profile: Candidate Profile.
            k: Target number of jobs.
            apply_filters: Filter location/seniority with fallback.
            apply_boost: Apply explainable skill boost.
            boost_weight: Weight parameter for skill boost.

        Returns:
            Ranked list of fused SearchResult objects.
        """
        candidate_pool_depth = max(k, 50)
        pool_size = max(candidate_pool_depth * 2, 100)

        # Retrieve candidate pools without applying filters yet (filters applied after fusion)
        bm25_pool = self.bm25_retriever.search(
            profile=profile,
            k=pool_size,
            apply_filters=False,
            apply_boost=False,
        )
        dense_pool = self.dense_retriever.search(
            profile=profile,
            k=pool_size,
            apply_filters=False,
            apply_boost=False,
        )

        fused_pool = reciprocal_rank_fusion(
            ranked_lists=[bm25_pool, dense_pool],
            rrf_k=self.rrf_k,
        )

        candidates = fused_pool
        if apply_filters:
            candidates, _ = filter_with_fallback(
                candidates,
                preferred_location=profile.preferred_location,
                seniority=profile.seniority,
                k=candidate_pool_depth,
            )
        else:
            candidates = candidates[:candidate_pool_depth]

        if apply_boost:
            candidates = apply_skill_boost(
                candidates,
                skills=profile.skills,
                weight=boost_weight,
                normalize_base_scores=True,
            )

        return candidates[:k]


class CrossEncoderRetriever(BaseRetriever):
    """Hybrid retrieval followed by deep cross-encoder re-ranking over top candidates."""

    def __init__(
        self,
        hybrid_retriever: HybridRRFRetriever,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        cross_encoder: Optional[CrossEncoder] = None,
        top_candidates_depth: int = 50,
        top_k_candidates: Optional[int] = None,
    ) -> None:
        """Initialize Cross-Encoder Re-Ranker.

        Args:
            hybrid_retriever: Upstream hybrid retriever providing candidate pools.
            model_name: Hugging Face cross-encoder model identifier.
            cross_encoder: Optional pre-loaded CrossEncoder instance.
            top_candidates_depth: Number of fused candidates passed to cross-encoder (default: 50).
            top_k_candidates: Optional alias for top_candidates_depth.
        """
        self.hybrid_retriever = hybrid_retriever
        self.top_candidates_depth = top_k_candidates if top_k_candidates is not None else top_candidates_depth

        import torch
        from sentence_transformers import CrossEncoder

        if cross_encoder is not None:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = CrossEncoder(model_name)

    def search(
        self,
        profile: Profile,
        k: int = 10,
        apply_filters: bool = True,
        apply_boost: bool = True,
        boost_weight: float = 0.2,
    ) -> List[SearchResult]:
        """Execute hybrid search, re-rank top candidates with Cross-Encoder, and apply personalization.

        Args:
            profile: Candidate Profile.
            k: Target number of jobs to return.
            apply_filters: Apply location/seniority filter with fallback.
            apply_boost: Apply explainable skill boost.
            boost_weight: Skill boost weighting parameter.

        Returns:
            Ranked list of re-ranked SearchResult objects.
        """
        query_text = build_query_text(profile)

        # 1. Retrieve top candidates from hybrid fusion (top 50 candidate pool)
        candidate_pool_depth = max(k, self.top_candidates_depth)
        candidates = self.hybrid_retriever.search(
            profile=profile,
            k=candidate_pool_depth,
            apply_filters=apply_filters,
            apply_boost=False,
        )

        if not candidates:
            return []

        # 2. Form cross-encoder pairs: (query, passage)
        pairs: List[Tuple[str, str]] = [
            (
                query_text,
                f"Job Title: {item.title} | Company: {item.company} | "
                f"Location: {item.location} | Level: {item.experience_level}\n"
                f"Description: {item.description[:1000]}",
            )
            for item in candidates
        ]

        import torch
        with torch.inference_mode():
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

        # 3. Update candidate scores and re-sort
        for item, score in zip(candidates, scores):
            # Sigmoid normalization for interpretability
            norm_score = round(float(1.0 / (1.0 + np.exp(-float(score)))), 4)
            item.score = norm_score
            item.raw_score = norm_score
            item.base_score = norm_score

        candidates.sort(key=lambda x: (x.filter_tier, -x.score))

        # 4. Apply explainable skill boost with min-max normalization
        if apply_boost:
            candidates = apply_skill_boost(
                candidates,
                skills=profile.skills,
                weight=boost_weight,
                normalize_base_scores=True,
            )

        return candidates[:k]
