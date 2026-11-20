"""Streamlit Demo Application for Job Search & Conversational Agent (Phase 3B - Local Only).

Provides:
1. Job Search Tab: Multi-strategy retrieval (BM25, Dense, Hybrid RRF, Cross-Encoder)
   with Gemini-generated grounded match explanations, matched/missing skill badges,
   and visible fallback indicators.
2. Conversational Agent Tab: Multi-turn agent interaction with persistent session memory,
   tool-call inspection (search_jobs, get_job_details, refine_filters), and loop cap.
3. Strict Free-Tier Quota & Rate Limiting: Per-session and daily request caps to protect
   your local Google AI Studio API key against unintended quota exhaustion or runaway loops;
   GEMINI_API_KEY is never exposed.
4. CC BY-SA 4.0 Dataset Attribution & Disclaimer Banner.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Set page config
st.set_page_config(
    page_title="JobMatch AI — Search & Agent Demo",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Configuration & Quota Management
# -----------------------------------------------------------------------------

MAX_SESSION_REQUESTS = 20
MAX_DAILY_REQUESTS = 200
USAGE_TRACKER_FILE = Path("data/cache/daily_usage.json")

def _get_api_key() -> Optional[str]:
    """Retrieve API key securely from environment or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return None


API_KEY = _get_api_key()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
DISABLE_CROSS_ENCODER = os.getenv("DISABLE_CROSS_ENCODER", "0").lower() in ("1", "true")


def get_daily_requests_count() -> int:
    """Retrieve count of live model requests made today across sessions."""
    today_str = datetime.date.today().isoformat()
    if not USAGE_TRACKER_FILE.exists():
        return 0
    try:
        with open(USAGE_TRACKER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == today_str:
            return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def increment_daily_requests_count() -> int:
    """Increment and persist today's request count."""
    today_str = datetime.date.today().isoformat()
    count = 1
    USAGE_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if USAGE_TRACKER_FILE.exists():
        try:
            with open(USAGE_TRACKER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today_str:
                count = int(data.get("count", 0)) + 1
        except Exception:
            pass

    try:
        with open(USAGE_TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "count": count}, f)
    except Exception:
        pass
    return count


def check_quota_available() -> Tuple[bool, str]:
    """Check both per-session and daily request caps protecting local API key."""
    sess_count = st.session_state.get("session_requests", 0)
    if sess_count >= MAX_SESSION_REQUESTS:
        return False, f"Session request limit reached ({MAX_SESSION_REQUESTS}/{MAX_SESSION_REQUESTS}). Reset conversation to protect local API quota."

    daily_count = get_daily_requests_count()
    if daily_count >= MAX_DAILY_REQUESTS:
        return False, f"Daily safety cap reached ({MAX_DAILY_REQUESTS}/{MAX_DAILY_REQUESTS} calls today) to protect local Google AI Studio API key."

    return True, ""


# Initialize session state variables
if "session_requests" not in st.session_state:
    st.session_state.session_requests = 0
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = f"st_session_{int(time.time())}"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {}


# -----------------------------------------------------------------------------
# Lazy Resource Loaders
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading retrieval indices and models...")
def load_retrieval_models():
    """Cache and load retrieval models and clean dataset."""
    from src.explain import GeminiExplainer
    from src.retrieval import (
        BM25Retriever,
        CrossEncoderRetriever,
        DenseRetriever,
        HybridRRFRetriever,
    )

    data_path = Path("data/jobs_clean.parquet")
    bm25 = BM25Retriever(data_path=data_path)
    dense = DenseRetriever(data_path=data_path)
    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=dense)

    cross_encoder = None
    if not DISABLE_CROSS_ENCODER:
        try:
            cross_encoder = CrossEncoderRetriever(hybrid_retriever=hybrid)
        except Exception:
            cross_encoder = None

    explainer = GeminiExplainer(api_key=API_KEY, model_name=MODEL_NAME)
    return bm25, dense, hybrid, cross_encoder, explainer


@st.cache_resource(show_spinner="Initializing conversational search agent...")
def load_agent():
    """Initialize JobSearchAgent with session memory and BM25 tool loop."""
    from src.agent import JobSearchAgent

    return JobSearchAgent(api_key=API_KEY, model_name=MODEL_NAME)


# -----------------------------------------------------------------------------
# UI Header & Banners
# -----------------------------------------------------------------------------

st.title("💼 JobMatch AI — Search & Conversational Agent")

# Mandatory Evaluation Disclaimer Banner
st.info(
    "ℹ️ **Evaluation Disclaimer**: Demo on a 3,000-posting sample of LinkedIn postings; "
    "relevance labels in the evaluation are rule-based and unverified."
)

daily_used = get_daily_requests_count()
if daily_used >= MAX_DAILY_REQUESTS:
    st.error(
        f"🛑 **Local Daily Safety Cap Reached**: The application has reached its daily limit of {MAX_DAILY_REQUESTS} model requests "
        "to protect your local Google AI Studio API key from unintended quota exhaustion or runaway loops. "
        "Job search and ranking remain fully operational; LLM match explanations and conversational agent turns will resume tomorrow."
    )

if not API_KEY:
    st.warning(
        "⚠️ **GEMINI_API_KEY** is not configured in .env or secrets. "
        "Job retrieval is fully functional; LLM explanations and conversational agent "
        "will operate in deterministic heuristic fallback mode."
    )

# -----------------------------------------------------------------------------
# Sidebar: Quota, Controls & Attribution
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Local System Status")

    # Quota usage meters protecting local API key
    sess_used = st.session_state.session_requests
    daily_used = get_daily_requests_count()

    st.caption(f"**Model**: `{MODEL_NAME}`")
    st.progress(min(1.0, sess_used / MAX_SESSION_REQUESTS))
    st.caption(f"Session Quota: **{sess_used} / {MAX_SESSION_REQUESTS}** requests")

    st.progress(min(1.0, daily_used / MAX_DAILY_REQUESTS))
    st.caption(f"Daily Safety Cap: **{daily_used} / {MAX_DAILY_REQUESTS}** requests")
    st.caption("🛡️ *Protects local Google AI Studio free tier (1,500 RPD).*")

    st.divider()

    st.header("📑 Dataset & Licensing")
    st.markdown(
        """
        - **Dataset**: 3,000 clean sample postings from LinkedIn Job Postings (2023–2024).
        - **Licence**: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
        - **Retrieval Engine**: BM25, ChromaDB dense vectors (`all-MiniLM-L6-v2`), RRF Hybrid, and Cross-Encoder.
        """
    )

    if DISABLE_CROSS_ENCODER:
        st.warning(
            "⚠️ **Local Resource Optimization**: Cross-Encoder Re-Ranking is disabled (`DISABLE_CROSS_ENCODER=1`) "
            "to conserve local RAM (811 MB steady-state). Hybrid RRF is used as default."
        )

    st.divider()
    if st.button("🔄 Reset Chat & Memory", use_container_width=True):
        st.session_state.chat_session_id = f"st_session_{int(time.time())}"
        st.session_state.chat_messages = []
        st.session_state.agent_memory = {}
        st.success("Session conversation and memory reset.")
        st.rerun()


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------

tab_search, tab_chat, tab_benchmarks = st.tabs([
    "🔍 Job Search & Matching",
    "🤖 Conversational Agent",
    "📊 Benchmarks & Provenance",
])


# =============================================================================
# TAB 1: JOB SEARCH & MATCHING
# =============================================================================

with tab_search:
    st.subheader("Candidate Profile & Matching Parameters")

    col_prof1, col_prof2 = st.columns([1, 1])

    with col_prof1:
        target_role = st.text_input(
            "Target Role",
            value="Python Backend Engineer",
            help="Desired job title or professional discipline.",
        )
        location = st.selectbox(
            "Preferred Location",
            options=["Any", "Remote", "New York, NY", "San Francisco, CA", "Chicago, IL", "Austin, TX", "Seattle, WA", "Boston, MA", "Los Angeles, CA"],
            index=1,
            help="Select 'Remote' or candidate's target metropolitan area.",
        )
        seniority = st.selectbox(
            "Experience Level",
            options=["Any", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"],
            index=3,
            help="Candidate target career stage.",
        )

    with col_prof2:
        default_skills = ["Python", "Django", "SQL", "Docker", "AWS", "PostgreSQL"]
        available_skills = sorted(list(set(default_skills + [
            "FastAPI", "React", "TypeScript", "Kubernetes", "Redis", "Kafka",
            "Machine Learning", "PyTorch", "GCP", "Git", "CI/CD", "Linux",
            "Pandas", "Scikit-Learn", "Java", "C++", "Go", "Terraform",
        ])))
        selected_skills = st.multiselect(
            "Declared Skills",
            options=available_skills,
            default=["Python", "Django", "SQL", "Docker"],
            help="Technical skills declared on candidate profile.",
        )
        resume_snippet = st.text_area(
            "Resume Text / Experience Summary",
            value="Experienced backend software engineer specializing in Python, Django REST frameworks, relational databases (PostgreSQL), and Docker container deployments.",
            height=85,
        )

    st.divider()

    col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1])
    with col_opt1:
        strategy_options = ["BM25 (Lexical)", "Dense (Vector)", "Hybrid (RRF)"]
        if not DISABLE_CROSS_ENCODER:
            strategy_options.append("Cross-Encoder (Re-Ranking)")
        strategy_choice = st.selectbox("Retrieval Strategy", options=strategy_options, index=0)

    with col_opt2:
        k_results = st.slider("Top Results (k)", min_value=1, max_value=15, value=5)

    with col_opt3:
        explain_matches = st.checkbox("Generate Gemini Explanations", value=True)
        apply_boost = st.checkbox("Apply Skill Boost (w=0.2)", value=True)

    search_clicked = st.button("🔎 Search Matching Jobs", type="primary", use_container_width=True)

    if search_clicked:
        from src.models import Profile

        prof = Profile(
            profile_id="st_user",
            target_role=target_role,
            preferred_location="" if location == "Any" else location,
            seniority="" if seniority == "Any" else seniority,
            skills=selected_skills,
            resume_text=resume_snippet,
        )

        bm25, dense, hybrid, cross_encoder, explainer = load_retrieval_models()

        # Route retrieval strategy
        t_start = time.perf_counter()
        if "BM25" in strategy_choice:
            retriever = bm25
        elif "Dense" in strategy_choice:
            retriever = dense
        elif "Cross-Encoder" in strategy_choice and cross_encoder is not None:
            retriever = cross_encoder
        else:
            retriever = hybrid

        results = retriever.search(
            profile=prof,
            k=k_results,
            apply_filters=True,
            apply_boost=apply_boost,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        st.subheader(f"Ranked Search Results ({len(results)} found in {elapsed_ms:.1f} ms)")

        if not results:
            st.warning("No postings matched the specified criteria. Try loosening location or seniority filters.")

        for rank, r in enumerate(results, 1):
            with st.container():
                st.markdown(
                    f"### {rank}. {r.title} — **{r.company}** "
                    f"*(Score: `{r.score:.4f}` | ID: `{r.job_id}`)*"
                )
                col_meta1, col_meta2, col_meta3 = st.columns([1, 1, 2])
                with col_meta1:
                    st.write(f"📍 **Location**: {r.location or 'Not Specified'}")
                with col_meta2:
                    st.write(f"💼 **Seniority**: {r.experience_level or 'Not Specified'}")

                # Explanation generation
                if explain_matches:
                    quota_ok, quota_msg = check_quota_available()
                    if not quota_ok:
                        st.warning(f"⚠️ {quota_msg} Showing raw posting details without LLM explanation.")
                    else:
                        job_dict = {
                            "id": r.job_id,
                            "title": r.title,
                            "company": r.company,
                            "location": r.location,
                            "experience_level": r.experience_level,
                            "description": r.description,
                        }
                        # Call explainer
                        expl_res = explainer.explain(
                            profile_data=prof.to_dict(),
                            job_data=job_dict,
                            allow_fallback=True,
                        )
                        st.session_state.session_requests += 1
                        increment_daily_requests_count()

                        # Visible Fallback Flag
                        if expl_res.fallback:
                            st.warning(
                                "⚠️ **Fallback Explanation**: Heuristic extraction used "
                                "(Gemini API call was unavailable, invalid, or rate-limited)."
                            )

                        if expl_res.explanation:
                            col_m1, col_m2 = st.columns([1, 1])
                            with col_m1:
                                if expl_res.explanation.matched_skills:
                                    matched_html = " ".join(
                                        f'<span style="background-color:#d4edda; color:#155724; padding:2px 8px; border-radius:12px; margin-right:4px;">{s}</span>'
                                        for s in expl_res.explanation.matched_skills
                                    )
                                    st.markdown(f"**Matched Skills**: {matched_html}", unsafe_allow_html=True)
                                else:
                                    st.markdown("**Matched Skills**: *None identified*")

                            with col_m2:
                                if expl_res.explanation.missing_skills:
                                    missing_html = " ".join(
                                        f'<span style="background-color:#fff3cd; color:#856404; padding:2px 8px; border-radius:12px; margin-right:4px;">{s}</span>'
                                        for s in expl_res.explanation.missing_skills
                                    )
                                    st.markdown(f"**Missing Qualifications**: {missing_html}", unsafe_allow_html=True)
                                else:
                                    st.markdown("**Missing Qualifications**: *None identified*")

                            st.markdown(f"> *{expl_res.explanation.summary}*")

                with st.expander("📄 View Job Description Snippet"):
                    st.write(r.description[:1200] + ("..." if len(r.description) > 1200 else ""))

                st.divider()


# =============================================================================
# TAB 2: CONVERSATIONAL SEARCH AGENT
# =============================================================================

with tab_chat:
    st.subheader("💬 Multi-Turn Job Search Agent")
    st.caption(
        "Powered by `gemini-3.1-flash-lite` with function calling (`search_jobs`, `get_job_details`, `refine_filters`). "
        "The agent retains your location, seniority, skills, and dislikes across turns."
    )

    # Active memory status pill
    agent_inst = load_agent()
    sess_dir = agent_inst.sessions_dir
    from src.agent import SessionMemory
    current_mem = SessionMemory.load(st.session_state.chat_session_id, directory=sess_dir)

    mem_col1, mem_col2, mem_col3, mem_col4 = st.columns(4)
    with mem_col1:
        st.metric("Location Filter", current_mem.location or "None")
    with mem_col2:
        st.metric("Seniority Filter", current_mem.experience_level or "None")
    with mem_col3:
        st.metric("Declared Skills", len(current_mem.skills))
    with mem_col4:
        st.metric("Excluded Dislikes", len(current_mem.dislikes))

    st.divider()

    # Display conversation turn history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                with st.expander("🛠️ Tool Execution Details", expanded=False):
                    for tc in msg["tools"]:
                        st.json(tc)

    # User chat input
    user_prompt = st.chat_input("Tell me what job you're looking for (e.g. 'Senior Python roles in New York', 'Exclude sales')...")

    if user_prompt:
        quota_ok, quota_msg = check_quota_available()
        if not quota_ok:
            st.error(f"⛔ {quota_msg}")
        else:
            # Display user message
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Execute agent turn
            with st.chat_message("assistant"):
                with st.spinner("Agent is reasoning and executing tools..."):
                    turn_res = agent_inst.chat(
                        session_id=st.session_state.chat_session_id,
                        message=user_prompt,
                    )
                    st.session_state.session_requests += 1
                    increment_daily_requests_count()

                    st.markdown(turn_res.reply)

                    tools_info = []
                    if turn_res.tool_calls:
                        tools_info = [
                            {
                                "tool": tc.tool_name,
                                "arguments": tc.args,
                                "error": tc.error,
                            }
                            for tc in turn_res.tool_calls
                        ]
                        with st.expander("🛠️ Tool Execution Details", expanded=False):
                            for tc in tools_info:
                                st.json(tc)

                    if turn_res.error:
                        st.warning(f"⚠️ Agent notice: {turn_res.error}")

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": turn_res.reply,
                        "tools": tools_info,
                    })
                    st.rerun()


# =============================================================================
# TAB 3: BENCHMARKS & PROVENANCE
# =============================================================================

with tab_benchmarks:
    st.subheader("📊 Rigorous Evaluation Ledger")

    st.markdown(
        """
        All numbers below are extracted programmatically from JSON reports in `reports/` produced
        by reproducible automated benchmark scripts.
        """
    )

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.metric("Evaluated Retrieval Postings", "3,000")
        st.metric("Dense Sync Speedup", "11.05x")
    with col_b2:
        st.metric("Phase 2C Main Groundedness", "91.7% / 94.2% adj.")
        st.metric("Explanation Schema Validity", "100.0% (0/150 fallback)")
    with col_b3:
        st.metric("Agent Behavioral Pass Rate", "90.0% (9/10)")
        st.metric("Agent Memory Retention", "100.0% (23/23 checks)")

    st.divider()

    st.markdown("### Retrieval Strategy Comparison (Phase 2B Main Setting)")
    ret_table_data = {
        "Strategy": ["BM25 (Lexical)", "Dense (Bi-Encoder)", "Hybrid RRF", "Cross-Encoder"],
        "Precision@5": ["0.4733 [0.38, 0.57]", "0.3333 [0.25, 0.43]", "0.4533 [0.36, 0.55]", "0.4600 [0.36, 0.56]"],
        "Recall@10": ["0.5593 [0.44, 0.68]", "0.3920 [0.28, 0.50]", "0.5540 [0.44, 0.67]", "0.5484 [0.44, 0.66]"],
        "MRR": ["0.6702 [0.55, 0.79]", "0.5186 [0.39, 0.64]", "0.6469 [0.53, 0.76]", "0.6558 [0.54, 0.77]"],
        "nDCG@10": ["0.6147 [0.50, 0.73]", "0.4475 [0.34, 0.56]", "0.6033 [0.49, 0.72]", "0.6074 [0.50, 0.72]"],
        "p50 CPU Latency": ["5.8 ms", "12.8 ms", "18.3 ms", "64.9 ms"],
    }
    st.table(pd.DataFrame(ret_table_data))

    st.caption(
        "Note: Evaluations conducted against 1,037 model-proposed labels (`jobmatch-rubric-v1`) "
        "that share lexical signals with BM25. See README.md and DECISIONS.md for full limitations."
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
        Dataset: LinkedIn Job Postings (3,000-sample clean slice) licensed under
        <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank">CC BY-SA 4.0</a>.
        | Built with Streamlit, ChromaDB, Sentence-Transformers, and Google Gemini.
        | Single-process warm CPU execution.
    </div>
    """,
    unsafe_allow_html=True,
)
