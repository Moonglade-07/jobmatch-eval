"""FastAPI Application for Job Search & Multi-Turn Conversational Agent (Phase 3A).

Endpoints:
- POST /search: Semantic & lexical job retrieval with optional Gemini grounded explanations.
- POST /chat: Multi-turn conversational career agent with function calling and memory.
- GET /health: Health check reporting model, version, and indexed jobs count.

Guarantees:
- Input validation: Oversized inputs (>10,000 chars resume, >2,000 chars message) rejected with HTTP 422.
- Error safety: Global exception handler suppresses stack traces, returning JSON with error_id.
- Configurable CORS via CORS_ORIGINS environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import JobSearchAgent
from src.explain import GeminiExplainer, evaluate_claim_groundedness
from src.models import Profile, SearchResult
from src.retrieval import BM25Retriever

logger = logging.getLogger("jobmatch.api")

DEFAULT_DATA_PATH = Path("data/jobs_clean.parquet")
API_VERSION = "1.0.0"

# -----------------------------------------------------------------------------
# Pydantic Schemas with Strict Character Limits
# -----------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Payload for POST /search."""

    resume_text: Optional[str] = Field(
        default="",
        max_length=10000,
        description="Candidate resume text narrative (max 10,000 chars).",
    )
    target_role: Optional[str] = Field(
        default="",
        max_length=500,
        description="Target role or job title (max 500 chars).",
    )
    preferred_location: Optional[str] = Field(
        default="",
        max_length=500,
        description="Desired location or 'Remote' (max 500 chars).",
    )
    seniority: Optional[str] = Field(
        default="",
        max_length=500,
        description="Target seniority level (max 500 chars).",
    )
    skills: Optional[List[str]] = Field(
        default_factory=list,
        max_length=100,
        description="Declared technical skills (max 100 items).",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of ranked job postings to return (1-50).",
    )
    explain: bool = Field(
        default=False,
        description="Whether to generate grounded LLM match explanations.",
    )


class ExplanationPayload(BaseModel):
    """Grounded explanation payload."""

    matched_skills: List[str]
    missing_skills: List[str]
    summary: str
    grounded: bool
    fallback: bool


class JobResultItem(BaseModel):
    """Ranked job item in search results."""

    job_id: str
    title: str
    company: str
    location: str
    experience_level: str
    score: float
    snippet: str
    explanation: Optional[ExplanationPayload] = None


class SearchResponse(BaseModel):
    """Response schema for POST /search."""

    query: Dict[str, Any]
    total_results: int
    results: List[JobResultItem]


class ChatRequest(BaseModel):
    """Payload for POST /chat."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier for user session (max 128 chars).",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's conversational message (max 2,000 chars).",
    )


class ToolCallItem(BaseModel):
    """Recorded tool call from agent turn."""

    tool_name: str
    args: Dict[str, Any]
    result: Any
    error: Optional[str] = None


class ChatResponse(BaseModel):
    """Response schema for POST /chat."""

    session_id: str
    reply: str
    tool_calls: List[ToolCallItem]
    session_memory: Dict[str, Any]
    steps: int
    latency_ms: float
    cache_hit: bool
    error: Optional[str] = Field(
        None,
        description="Error message if an upstream service error occurred during turn processing.",
    )


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    version: str
    model: str
    indexed_jobs_count: int


# -----------------------------------------------------------------------------
# Global State Container
# -----------------------------------------------------------------------------


class AppState:
    """Singleton container holding indexed data and services."""

    def __init__(self) -> None:
        self.jobs_df: pd.DataFrame = pd.DataFrame()
        self.retriever: Optional[BM25Retriever] = None
        self.explainer: Optional[GeminiExplainer] = None
        self.agent: Optional[JobSearchAgent] = None
        self.model_name: str = "gemini-3.1-flash-lite"

    def initialize(self, data_path: Path = DEFAULT_DATA_PATH) -> None:
        """Load dataset, retriever, explainer, and agent."""
        if data_path.exists():
            self.jobs_df = pd.read_parquet(data_path)
            self.retriever = BM25Retriever(df=self.jobs_df)
        else:
            self.jobs_df = pd.DataFrame()
            self.retriever = None

        self.agent = JobSearchAgent(data_path=data_path)
        self.model_name = self.agent.model_name
        self.explainer = GeminiExplainer(model_name=self.model_name)


app_state = AppState()


# -----------------------------------------------------------------------------
# Application Setup & Lifespan
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    if app_state.retriever is None:
        app_state.initialize()
    yield


app = FastAPI(
    title="Job Match Evaluation & Career Agent API",
    description="Search & conversational agent backend for technical job matching.",
    version=API_VERSION,
    lifespan=lifespan,
)

# Configure CORS
cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Global Exception Handler (Never expose stack traces)
# -----------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return safe JSON with error_id."""
    error_id = str(uuid.uuid4())
    logger.error(f"Unhandled server error [error_id={error_id}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error_id": error_id},
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return system health status, model, and dynamically computed indexed jobs count."""
    if app_state.retriever is None:
        app_state.initialize()
    count = len(app_state.retriever.records) if app_state.retriever is not None else len(app_state.jobs_df)
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        model=app_state.model_name,
        indexed_jobs_count=count,
    )


@app.post("/search", response_model=SearchResponse)
def search_jobs(payload: SearchRequest) -> SearchResponse:
    """Retrieve ranked jobs with optional Gemini match explanations."""
    if app_state.retriever is None:
        app_state.initialize()

    profile = Profile(
        resume_text=payload.resume_text or "",
        target_role=payload.target_role or "",
        preferred_location=payload.preferred_location or "",
        seniority=payload.seniority or "",
        skills=payload.skills or [],
        profile_id="api_request",
    )

    if app_state.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job index is unavailable.",
        )

    raw_results = app_state.retriever.search(
        profile=profile,
        k=payload.k,
        apply_filters=True,
        apply_boost=True,
    )

    results_out: List[JobResultItem] = []

    for r in raw_results:
        explanation_payload: Optional[ExplanationPayload] = None

        if payload.explain and app_state.explainer is not None:
            # Job posting dict
            job_dict = {
                "id": r.job_id,
                "title": r.title,
                "company": r.company,
                "location": r.location,
                "experience_level": r.experience_level,
                "description": r.description,
            }
            expl_res = app_state.explainer.explain(
                profile_data=profile.to_dict(),
                job_data=job_dict,
                use_cache=True,
            )

            if expl_res.explanation:
                job_text = f"{r.title}\n{r.description}"
                grounded_rate, _, _ = evaluate_claim_groundedness(
                    expl_res.explanation.matched_skills, job_text
                )
                explanation_payload = ExplanationPayload(
                    matched_skills=expl_res.explanation.matched_skills,
                    missing_skills=expl_res.explanation.missing_skills,
                    summary=expl_res.explanation.summary,
                    grounded=bool(grounded_rate >= 1.0),
                    fallback=expl_res.fallback,
                )

        snippet_text = r.snippet if hasattr(r, "snippet") and r.snippet else (
            r.description[:250] + ("..." if len(r.description) > 250 else "")
        )

        results_out.append(
            JobResultItem(
                job_id=r.job_id,
                title=r.title,
                company=r.company,
                location=r.location,
                experience_level=r.experience_level,
                score=round(float(r.score), 4),
                snippet=snippet_text,
                explanation=explanation_payload,
            )
        )

    return SearchResponse(
        query={
            "target_role": payload.target_role,
            "preferred_location": payload.preferred_location,
            "seniority": payload.seniority,
            "skills_count": len(payload.skills or []),
            "k": payload.k,
            "explain": payload.explain,
        },
        total_results=len(results_out),
        results=results_out,
    )


@app.post("/chat", response_model=ChatResponse)
def chat_turn(payload: ChatRequest) -> ChatResponse:
    """Execute conversational agent turn with persistent memory and tool calling."""
    if app_state.agent is None:
        app_state.initialize()

    if app_state.agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job search agent is unavailable.",
        )

    turn_res = app_state.agent.chat(
        session_id=payload.session_id,
        message=payload.message,
    )

    tool_calls_out = [
        ToolCallItem(
            tool_name=tc.tool_name,
            args=tc.args,
            result=tc.result,
            error=tc.error,
        )
        for tc in turn_res.tool_calls
    ]

    return ChatResponse(
        session_id=turn_res.session_id,
        reply=turn_res.reply,
        tool_calls=tool_calls_out,
        session_memory=turn_res.session_memory,
        steps=turn_res.steps,
        latency_ms=turn_res.latency_ms,
        cache_hit=turn_res.cache_hit,
        error=str(turn_res.error) if turn_res.error else None,
    )
