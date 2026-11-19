"""Pool retrieval candidates across four strategies and generate graded relevance proposals.

Pipeline steps:
  1. Load 30 validated candidate profiles from data/eval/profiles.json.
  2. For each profile, retrieve top 20 candidates from BM25, Dense, Hybrid (RRF),
     and Cross-Encoder re-ranker.
  3. Form the union of candidate pools per profile (pooling) to eliminate strategy bias.
  4. Propose graded relevance labels (0 = Irrelevant, 1 = Partially Relevant, 2 = Strongly Relevant)
     according to the fixed rubric in docs/LABELING.md.
  5. Sample a reproducible 25% subset (fixed seed 42) flagged as `must_review = True`.
  6. Export `labels_proposed.csv` (and `data/eval/labels_proposed.csv`) with columns:
     profile_id, job_id, title, company, key snippet of description, proposed_label, my_label (empty), must_review.
  7. Record labeling metadata, rubric, model, and prompt in `reports/labeling_method.json`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep offline mode for Hugging Face and disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd

from src.models import Profile, SearchResult
from src.personalise import extract_matched_skills, is_remote_posting, matches_location, matches_seniority
from src.retrieval import (
    BM25Retriever,
    CrossEncoderRetriever,
    DenseRetriever,
    HybridRRFRetriever,
)

DEFAULT_PROFILES_PATH = Path("data/eval/profiles.json")
DEFAULT_OUTPUT_CSV_PATH = Path("labels_proposed.csv")
DEFAULT_EVAL_CSV_PATH = Path("data/eval/labels_proposed.csv")
DEFAULT_METHOD_REPORT_PATH = Path("reports/labeling_method.json")
RANDOM_SEED = 42
REVIEW_SAMPLE_RATE = 0.25


def extract_key_snippet(description: str, max_chars: int = 240) -> str:
    """Extract a concise, representative snippet from the job description.

    Attempts to locate the primary role description or summary sentence,
    collapsing multiple whitespaces into a single clean line.

    Args:
        description: Full cleaned job description.
        max_chars: Maximum character length for preview snippet.

    Returns:
        Single-line text preview snippet.
    """
    if not description:
        return ""
    clean = " ".join(description.split())
    if len(clean) <= max_chars:
        return clean

    # Look for common sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    accumulated = ""
    for s in sentences:
        if len(accumulated) + len(s) + 1 <= max_chars:
            accumulated = f"{accumulated} {s}".strip()
        else:
            break

    if accumulated and len(accumulated) >= 60:
        return f"{accumulated}..."
    return f"{clean[:max_chars].strip()}..."


def evaluate_relevance(
    profile: Profile,
    job_id: str,
    title: str,
    company: str,
    location: str,
    experience_level: str,
    description: str,
) -> Tuple[int, str]:
    """Assign graded relevance label (0, 1, or 2) according to the rubric in docs/LABELING.md.

    Grading Criteria:
      - 2 (Strongly Relevant): Direct target role alignment, high skill overlap,
        compatible seniority, and satisfied location preference.
      - 1 (Partially Relevant): Adjacent role / domain, transferable foundation skills,
        acceptable seniority transition, or valid career-changer match.
      - 0 (Irrelevant): Complete domain mismatch, distant incompatible location,
        severe seniority clash (e.g. executive for intern), or zero technical overlap.

    Args:
        profile: Candidate profile.
        job_id: Unique job posting identifier.
        title: Job posting title.
        company: Hiring organization name.
        location: Normalized location string.
        experience_level: Normalized experience level.
        description: Cleaned job description text.

    Returns:
        Tuple of (integer label {0, 1, 2}, rationale string).
    """
    t_lower = title.lower()
    target_role_lower = profile.target_role.lower()
    combined_text = f"{title} {description}"

    # 1. Geographic compatibility check
    loc_compatible = matches_location(
        job_location=location,
        preferred_location=profile.preferred_location,
        title=title,
        description=description,
    )
    if not loc_compatible:
        return 0, f"Location mismatch: Candidate prefers '{profile.preferred_location}' but job is '{location}' with no remote option."

    # 2. Seniority evaluation
    job_lvl = experience_level.lower()
    cand_lvl = profile.seniority.lower()

    # Severe seniority clash check
    is_extreme_clash = (
        (cand_lvl in {"entry level", "internship"} and ("director" in job_lvl or "executive" in job_lvl or "vp" in t_lower or "chief" in t_lower))
        or (cand_lvl in {"director", "executive"} and ("intern" in job_lvl or "internship" in t_lower or "junior" in t_lower))
    )
    if is_extreme_clash:
        return 0, f"Extreme seniority clash: Candidate '{profile.seniority}' vs Job '{experience_level}'."

    # 3. Complete domain / role disconnect check (non-technical or unrelated profession)
    unrelated_professions = [
        "receptionist", "dental", "nursing", "nurse", "physician", "medical assistant",
        "warehouse", "forklift", "truck driver", "driver", "cashier", "cook", "chef",
        "retail associate", "store manager", "mechanic", "automotive", "paralegal", "attorney",
        "load builder", "property manager", "real estate agent", "teller", "barista"
    ]
    if any(re.search(rf"\b{re.escape(w)}\b", t_lower) for w in unrelated_professions):
        return 0, f"Unrelated profession: '{title}' is non-technical and outside candidate scope."

    # 4. Skill extraction
    matched_skills = extract_matched_skills(profile.skills, combined_text)
    total_skills = len(profile.skills)
    skill_fraction = len(matched_skills) / total_skills if total_skills > 0 else 0.0

    # 5. Role & Title affinity
    target_tokens = set(re.findall(r"\b\w+\b", target_role_lower)) - {"junior", "senior", "lead", "staff", "intern", "engineer", "developer", "specialist"}
    title_tokens = set(re.findall(r"\b\w+\b", t_lower))

    # Calculate token overlap on core role words (e.g. 'frontend', 'machine', 'learning', 'devops', 'data', 'qa')
    core_overlap = target_tokens & title_tokens
    has_direct_role_keyword = bool(core_overlap)

    # 6. Hard case handling (Career changers & Vague profiles)
    if profile.profile_id in {"prof_27", "prof_28"}:
        # Career changers: Target is entry level data/software
        if ("data" in t_lower or "analyst" in t_lower or "software" in t_lower or "developer" in t_lower or "engineer" in t_lower) and len(matched_skills) >= 2:
            if "senior" in t_lower or "lead" in t_lower or "director" in t_lower:
                return 1, f"Career changer stretch: Matched {len(matched_skills)} skills ({', '.join(matched_skills)}) but role is senior."
            return 2 if skill_fraction >= 0.50 else 1, f"Career changer fit: Matches {len(matched_skills)} core transferable skills."
        elif len(matched_skills) >= 1:
            return 1, f"Career changer adjacent: Modest skill match ({', '.join(matched_skills)})."
        else:
            return 0, "Career changer mismatch: Insufficient technical skill overlap."

    if profile.profile_id in {"prof_29", "prof_30"}:
        # Vague profiles: Broad IT / tech support
        if any(term in t_lower for term in ["support", "it ", "desktop", "help desk", "technician", "systems", "administrator", "infrastructure"]):
            return 2 if len(matched_skills) >= 2 else 1, f"Broad IT match for vague technical profile."
        elif "engineer" in t_lower or "developer" in t_lower:
            return 1 if len(matched_skills) >= 1 else 0, "Technical adjacent opportunity for generalist."
        return 0, "Mismatch for general IT profile."

    # 7. Standard Role Classification
    # Label 2 (Strongly Relevant): Direct role match + appropriate seniority + solid skill overlap
    if has_direct_role_keyword:
        if len(matched_skills) >= 2 or skill_fraction >= 0.35:
            # Check seniority compatibility
            if matches_seniority(experience_level, profile.seniority):
                return 2, f"Strong match: Direct role alignment ('{title}') with {len(matched_skills)} matched skills ({', '.join(matched_skills)}) and compatible seniority."
            else:
                return 1, f"Partial match: Direct role alignment ('{title}') with skills, but seniority difference ('{experience_level}' vs '{profile.seniority}')."
        elif len(matched_skills) == 1:
            return 1, f"Partial match: Role alignment ('{title}') but limited skill overlap ({matched_skills[0]})."
        else:
            # Core role matches title but zero claimed skills in description
            return 1, f"Partial match: Target title match ('{title}') but lacking explicit candidate skill overlap."

    # Adjacent role matching (e.g. Data Scientist <-> Data Analyst; Backend <-> Full Stack; DevOps <-> SRE)
    adjacent_pairs = [
        ({"data analyst", "data scientist", "bi developer", "business intelligence"}, {"data", "analyst", "scientist", "analytics", "bi", "intelligence", "tableau"}),
        ({"machine learning engineer", "data scientist", "nlp engineer", "computer vision engineer", "research scientist"}, {"ml", "ai", "machine learning", "deep learning", "nlp", "computer vision", "algorithm"}),
        ({"senior backend engineer", "full stack developer", "software engineer", "embedded systems engineer"}, {"software", "backend", "full stack", "web", "systems", "platform"}),
        ({"devops engineer", "site reliability engineer", "systems administrator", "cloud devops architect"}, {"cloud", "devops", "sre", "reliability", "infrastructure", "platform", "systems", "linux"}),
        ({"qa automation engineer", "software engineer"}, {"qa", "test", "automation", "quality"}),
        ({"product manager", "scrum master", "solutions architect"}, {"product", "scrum", "agile", "project", "program", "solutions"}),
    ]

    is_adjacent_role = False
    for group_roles, keywords in adjacent_pairs:
        if any(r in target_role_lower for r in group_roles):
            if any(kw in t_lower for kw in keywords):
                is_adjacent_role = True
                break

    if is_adjacent_role:
        if len(matched_skills) >= 2:
            return 1, f"Partial match: Adjacent technical discipline ('{title}') with {len(matched_skills)} shared skills ({', '.join(matched_skills)})."
        elif len(matched_skills) == 1:
            return 1, f"Partial match: Adjacent role ('{title}') with 1 transferable skill ({matched_skills[0]})."
        else:
            return 0, f"Domain mismatch: Adjacent title ('{title}') but 0 skill matches."

    # If role is different and < 2 skills match -> 0
    if len(matched_skills) >= 3:
        return 1, f"Partial match: Different title ('{title}') but strong transferable skill intersection ({', '.join(matched_skills)})."

    return 0, f"Irrelevant: Role '{title}' does not align with target '{profile.target_role}' (matched {len(matched_skills)} skills)."


def run_pooling_and_labeling(
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV_PATH,
    eval_csv_path: Path = DEFAULT_EVAL_CSV_PATH,
    report_path: Path = DEFAULT_METHOD_REPORT_PATH,
    top_k_per_strategy: int = 20,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Execute candidate pooling across 4 strategies and generate proposed labels.

    Args:
        profiles_path: Path to profiles JSON.
        output_csv_path: Root output path for labels_proposed.csv.
        eval_csv_path: Data/eval destination path for labels_proposed.csv.
        report_path: Destination path for labeling_method.json report.
        top_k_per_strategy: Number of top items retrieved per strategy for pooling (default: 20).
        seed: Random seed for 25% review sample selection.

    Returns:
        DataFrame containing all pooled pairs with proposed labels.
    """
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles file not found at {profiles_path}. Run generate_eval_profiles.py first.")

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles_data = json.load(f)

    profile_dicts = profiles_data.get("profiles", [])
    profiles = [
        Profile(
            profile_id=p["profile_id"],
            target_role=p["target_role"],
            skills=p["skills"],
            preferred_location=p["preferred_location"],
            seniority=p["seniority"],
            resume_text=p["resume_text"],
        )
        for p in profile_dicts
    ]
    print(f"Loaded {len(profiles)} profiles from {profiles_path}.")

    print("Initializing retrieval strategies...")
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_k=60)
    cross = CrossEncoderRetriever(hybrid_retriever=hybrid, top_candidates_depth=50)

    strategies = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "cross_encoder": cross,
    }

    pooled_records: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    print(f"\nBuilding candidate pool (top {top_k_per_strategy} union across 4 strategies per profile)...")
    for i, prof in enumerate(profiles, 1):
        print(f"  [{i:02d}/{len(profiles):02d}] Pooling candidates for {prof.profile_id} ({prof.target_role})...")
        retrieved_by_strategy: Dict[str, List[SearchResult]] = {}

        for strat_name, retriever in strategies.items():
            results = retriever.search(profile=prof, k=top_k_per_strategy)
            retrieved_by_strategy[strat_name] = results

        # Form union of unique jobs
        prof_jobs: Dict[str, SearchResult] = {}
        for strat_name, r_list in retrieved_by_strategy.items():
            for r in r_list:
                if r.job_id not in prof_jobs:
                    prof_jobs[r.job_id] = r

        print(f"      -> Retrieved {len(prof_jobs)} unique pooled postings for {prof.profile_id}.")

        for job_id, item in prof_jobs.items():
            pair_key = (prof.profile_id, job_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            label, rationale = evaluate_relevance(
                profile=prof,
                job_id=job_id,
                title=item.title,
                company=item.company,
                location=item.location,
                experience_level=item.experience_level,
                description=item.description,
            )

            snippet = extract_key_snippet(item.description, max_chars=240)

            pooled_records.append(
                {
                    "profile_id": prof.profile_id,
                    "job_id": str(job_id),
                    "title": item.title,
                    "company": item.company,
                    "key snippet of description": snippet,
                    "proposed_label": int(label),
                    "my_label": "",  # Empty for human reviewer
                    "rationale": rationale,
                }
            )

    df_labels = pd.DataFrame(pooled_records)
    total_pairs = len(df_labels)
    print(f"\nTotal pooled (profile, job) pairs: {total_pairs}")

    # Mark a random 25% sample (fixed seed) as must_review
    np.random.seed(seed)
    review_indices = set(
        np.random.choice(
            total_pairs,
            size=int(np.ceil(total_pairs * REVIEW_SAMPLE_RATE)),
            replace=False,
        )
    )

    df_labels["must_review"] = [
        "True" if idx in review_indices else "False" for idx in range(total_pairs)
    ]

    must_review_count = (df_labels["must_review"] == "True").sum()
    print(f"Marked {must_review_count}/{total_pairs} rows ({must_review_count/total_pairs*100:.1f}%) as must_review (seed={seed}).")

    # Column ordering as specified by the prompt:
    # profile_id, job_id, title, company, key snippet of description, proposed_label, my_label (empty), must_review
    export_columns = [
        "profile_id",
        "job_id",
        "title",
        "company",
        "key snippet of description",
        "proposed_label",
        "my_label",
        "must_review",
    ]
    df_export = df_labels[export_columns]

    # Save to both paths
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_csv(output_csv_path, index=False, encoding="utf-8")
    df_export.to_csv(eval_csv_path, index=False, encoding="utf-8")
    print(f"Exported labels_proposed.csv to {output_csv_path} and {eval_csv_path}.")

    # Compute label distribution
    label_dist = df_labels["proposed_label"].value_counts().to_dict()
    label_dist_formatted = {
        f"label_{k}": {
            "count": int(v),
            "percentage": round(float(v / total_pairs) * 100, 2),
        }
        for k, v in sorted(label_dist.items())
    }

    # Record labeling method report
    method_report: Dict[str, Any] = {
        "labeling_approach": "Deterministic Rubric-Guided Relevance Evaluation",
        "evaluator_model": "jobmatch-rubric-v1 (Rule-based expert heuristic alignment)",
        "rubric_documentation": "docs/LABELING.md",
        "random_seed": seed,
        "sample_rate_must_review": REVIEW_SAMPLE_RATE,
        "total_profiles_evaluated": len(profiles),
        "total_pooled_pairs": int(total_pairs),
        "must_review_count": int(must_review_count),
        "label_distribution": label_dist_formatted,
        "pooling_parameters": {
            "top_k_per_strategy": top_k_per_strategy,
            "strategies": ["bm25", "dense", "hybrid_rrf", "cross_encoder"],
        },
        "rubric_criteria": {
            "label_0": "Irrelevant: Domain disconnect, incompatible location, severe seniority clash, or zero technical overlap.",
            "label_1": "Partially Relevant: Adjacent technical role/domain, transferable core skills, acceptable stretch seniority, or career-changer fit.",
            "label_2": "Strongly Relevant: Direct target role match, high core skill synergy, compatible seniority, and satisfied location.",
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(method_report, f, indent=2)

    print(f"Saved labeling method report to {report_path}.")
    print("\nSummary of Proposed Labels:")
    print(json.dumps(label_dist_formatted, indent=2))
    return df_export


if __name__ == "__main__":
    run_pooling_and_labeling()

