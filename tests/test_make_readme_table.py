"""Unit tests for src/make_readme_table.py."""

from pathlib import Path
import pytest

from src.make_readme_table import (
    load_json,
    generate_results_markdown,
    generate_results_md,
    START_MARKER,
    END_MARKER,
)


def test_load_json_reports():
    """Verify load_json correctly loads report files."""
    data_stats = load_json("data_stats.json")
    assert "row_counts" in data_stats
    assert data_stats["row_counts"]["final_sampled_postings"] == 3000

    llm_quality = load_json("llm_quality.json")
    assert "groundedness" in llm_quality
    assert "validity" in llm_quality


def test_generate_results_markdown_has_required_sections_and_captions():
    """Verify generated results markdown contains required sections, captions, and no forbidden terms."""
    data_stats = load_json("data_stats.json")
    index_stats = load_json("index_stats.json")
    sync_stats = load_json("sync_benchmark.json")
    label_method = load_json("labeling_method.json")
    label_agree = load_json("label_agreement.json")
    ret_main = load_json("retrieval_metrics_main.json")
    ret_pure = load_json("retrieval_metrics_pure.json")
    ret_paired = load_json("retrieval_paired_comparisons.json")
    ret_latency = load_json("retrieval_latency.json")
    ret_ablation = load_json("retrieval_ablation.json")
    ret_circularity = load_json("retrieval_circularity_check.json")
    llm_quality = load_json("llm_quality.json")

    md = generate_results_markdown(
        data_stats,
        index_stats,
        sync_stats,
        label_method,
        label_agree,
        ret_main,
        ret_pure,
        ret_paired,
        ret_latency,
        ret_ablation,
        ret_circularity,
        llm_quality,
    )

    # Mandatory captions
    assert "Evaluated against model-proposed, unverified relevance labels" in md
    assert "Programmatic checks against job text; no human or LLM judge" in md

    # Groundedness both ways
    assert "Main Groundedness (Baseline)" in md
    assert "Main Groundedness (Adjusted for Joined Words)" in md
    assert "91.70%" in md
    assert "94.22%" in md

    # Forbidden phrases check
    assert "ground truth" not in md.lower()
    assert "human-verified" not in md.lower()


def test_generate_results_md_provenance_ledger():
    """Verify RESULTS.md provenance registry exists and covers all stages."""
    data_stats = load_json("data_stats.json")
    index_stats = load_json("index_stats.json")
    sync_stats = load_json("sync_benchmark.json")
    label_method = load_json("labeling_method.json")
    label_agree = load_json("label_agreement.json")
    ret_main = load_json("retrieval_metrics_main.json")
    ret_pure = load_json("retrieval_metrics_pure.json")
    ret_paired = load_json("retrieval_paired_comparisons.json")
    ret_latency = load_json("retrieval_latency.json")
    ret_ablation = load_json("retrieval_ablation.json")
    ret_circularity = load_json("retrieval_circularity_check.json")
    llm_quality = load_json("llm_quality.json")

    results_doc = generate_results_md(
        data_stats,
        index_stats,
        sync_stats,
        label_method,
        label_agree,
        ret_main,
        ret_pure,
        ret_paired,
        ret_latency,
        ret_ablation,
        ret_circularity,
        llm_quality,
    )

    assert "# Experimental Results & Provenance Ledger" in results_doc
    assert "Metric Provenance Registry" in results_doc
    assert "python src/ingest.py" in results_doc
    assert "python src/index.py" in results_doc
    assert "python src/bench_sync.py" in results_doc
    assert "python src/run_retrieval_eval.py" in results_doc
    assert "python src/run_explain_eval.py --full" in results_doc
