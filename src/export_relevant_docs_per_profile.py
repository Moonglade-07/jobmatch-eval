"""Export distribution of judged pairs and relevant documents per candidate profile.

Evaluated against model-proposed, unverified relevance labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROFILES_PATH = Path("data/eval/profiles.json")
QRELS_PATH = Path("data/eval/qrels.json")
OUTPUT_PATH = Path("reports/relevant_docs_per_profile.json")

MANDATORY_CAPTION = "Evaluated against model-proposed, unverified relevance labels"
LABEL_ORIGIN_NOTE = (
    "Relevance labels are rule-based proposed labels (jobmatch-rubric-v1) "
    "generated via deterministic heuristics; unverified by humans."
)


def export_relevant_docs_per_profile() -> Dict[str, Any]:
    """Calculate and export relevant document counts per evaluation profile."""
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles_data = json.load(f)["profiles"]

    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        qrels_data = json.load(f)["qrels"]

    profile_summaries: List[Dict[str, Any]] = []
    under_three: List[Dict[str, Any]] = []

    print("\n" + "=" * 95)
    print(f"RELEVANT DOCUMENTS PER EVALUATION PROFILE ({MANDATORY_CAPTION})")
    print("=" * 95)
    print(
        f"{'Profile ID':<10} | {'Target Role':<28} | {'Seniority':<18} | "
        f"{'Judged':<8} | {'Rel (>=1)':<10} | {'Strong (==2)':<12}"
    )
    print("-" * 95)

    for p in profiles_data:
        pid = p["profile_id"]
        role = p["target_role"]
        sen = p["seniority"]
        loc = p["preferred_location"]

        p_qrels = qrels_data.get(pid, {})
        judged_count = len(p_qrels)
        rel_count = sum(
            1
            for d in p_qrels.values()
            if (d["relevance"] if isinstance(d, dict) else int(d)) >= 1
        )
        strong_count = sum(
            1
            for d in p_qrels.values()
            if (d["relevance"] if isinstance(d, dict) else int(d)) == 2
        )

        entry = {
            "profile_id": pid,
            "target_role": role,
            "seniority": sen,
            "preferred_location": loc,
            "judged_pairs_count": judged_count,
            "relevant_count": rel_count,
            "strongly_relevant_count": strong_count,
        }
        profile_summaries.append(entry)

        print(
            f"{pid:<10} | {role:<28} | {sen:<18} | "
            f"{judged_count:<8} | {rel_count:<10} | {strong_count:<12}"
        )

        if rel_count < 3:
            under_three.append({
                "profile_id": pid,
                "target_role": role,
                "seniority": sen,
                "preferred_location": loc,
                "judged_pairs_count": judged_count,
                "relevant_count": rel_count,
                "strongly_relevant_count": strong_count,
            })

    print("=" * 95)
    print(f"\nProfiles with fewer than 3 relevant documents ({len(under_three)} total):")
    for u in under_three:
        print(f"  - {u['profile_id']} ({u['target_role']}, {u['preferred_location']}): {u['relevant_count']} relevant doc(s)")
    print("-" * 95)

    payload = {
        "caption": MANDATORY_CAPTION,
        "note": LABEL_ORIGIN_NOTE,
        "total_profiles": len(profiles_data),
        "profiles_with_fewer_than_3_relevant_docs_count": len(under_three),
        "profiles_with_fewer_than_3_relevant_docs": under_three,
        "coverage_finding": (
            "Niche technical roles (such as Computer Vision Engineer, Mobile Developer, "
            "NLP Engineer, and Machine Learning Engineer) have very sparse coverage in this "
            "3,000-posting LinkedIn sample, yielding < 3 relevant documents under the strict rubric."
        ),
        "profiles": profile_summaries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved profile relevant document distribution to {OUTPUT_PATH}")
    return payload


if __name__ == "__main__":
    export_relevant_docs_per_profile()

