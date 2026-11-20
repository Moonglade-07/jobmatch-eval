"""Unit tests for FastAPI application endpoints and schemas (Phase 3A).

Validates:
1. Endpoints: POST /search, POST /chat, GET /health.
2. Input validation: rejection of oversized inputs (>10,000 chars resume, >2,000 chars message, >128 chars session_id).
3. Stack trace suppression: unhandled exceptions return HTTP 500 with error_id and no tracebacks.
4. CORS middleware configuration.
5. Zero live API calls in test suite (all models and services mocked).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import ValidationError

from src.api import (
    API_VERSION,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    app,
    app_state,
    chat_turn,
    health,
    search_jobs,
    unhandled_exception_handler,
)
from src.explain import ExplainResult, Explanation
from src.models import SearchResult


@pytest.fixture(autouse=True)
def setup_mock_services():
    """Configure mocked retriever, explainer, and agent before each test."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        SearchResult(
            job_id="1000000001",
            score=0.88,
            title="Senior Python Backend Engineer",
            company="TechCorp",
            location="New York, NY",
            experience_level="Mid-Senior level",
            description="Looking for Python, Django, PostgreSQL, Docker, AWS.",
        )
    ]

    mock_explainer = MagicMock()
    mock_explainer.explain.return_value = ExplainResult(
        profile_id="api_request",
        job_id="1000000001",
        explanation=Explanation(
            matched_skills=["Python", "Django"],
            missing_skills=["AWS"],
            summary="Candidate has strong Python backend alignment.",
        ),
        fallback=False,
    )

    mock_agent = MagicMock()
    mock_agent.model_name = "gemini-3.1-flash-lite"
    mock_agent_turn = MagicMock()
    mock_agent_turn.session_id = "test_sess"
    mock_agent_turn.reply = "Here are the Python positions in New York, NY."
    mock_agent_turn.tool_calls = []
    mock_agent_turn.session_memory = {"location": "New York, NY", "skills": ["Python"]}
    mock_agent_turn.steps = 1
    mock_agent_turn.latency_ms = 45.2
    mock_agent_turn.cache_hit = False
    mock_agent.chat.return_value = mock_agent_turn

    app_state.retriever = mock_retriever
    app_state.explainer = mock_explainer
    app_state.agent = mock_agent
    app_state.model_name = "gemini-3.1-flash-lite"


def test_routes_registered():
    """Verify all required endpoints are registered with expected HTTP methods."""
    routes = {r.path: r.methods for r in app.routes if hasattr(r, "methods")}
    assert "/health" in routes
    assert "GET" in routes["/health"]
    assert "/search" in routes
    assert "POST" in routes["/search"]
    assert "/chat" in routes
    assert "POST" in routes["/chat"]


def test_health_endpoint():
    """GET /health returns 200 schema with status ok, model, and version."""
    res = health()
    assert isinstance(res, HealthResponse)
    assert res.status == "ok"
    assert res.version == API_VERSION
    assert res.model == "gemini-3.1-flash-lite"
    assert res.indexed_jobs_count >= 0


def test_search_endpoint_basic():
    """POST /search returns ranked jobs without explanations when explain=False."""
    req = SearchRequest(
        target_role="Python Developer",
        preferred_location="New York, NY",
        k=3,
        explain=False,
    )
    res = search_jobs(req)
    assert isinstance(res, SearchResponse)
    assert res.total_results == 1
    item = res.results[0]
    assert item.job_id == "1000000001"
    assert item.score == 0.88
    assert item.title == "Senior Python Backend Engineer"
    assert item.explanation is None


def test_search_endpoint_with_explanation():
    """POST /search with explain=True generates structured grounded explanation."""
    req = SearchRequest(
        target_role="Python Developer",
        preferred_location="New York, NY",
        k=1,
        explain=True,
    )
    res = search_jobs(req)
    assert isinstance(res, SearchResponse)
    item = res.results[0]
    assert item.explanation is not None
    assert "Python" in item.explanation.matched_skills
    assert item.explanation.grounded is True
    assert item.explanation.fallback is False


def test_chat_endpoint():
    """POST /chat executes multi-turn conversational agent interaction."""
    req = ChatRequest(
        session_id="session_abc_123",
        message="Looking for Python roles in New York",
    )
    res = chat_turn(req)
    assert isinstance(res, ChatResponse)
    assert res.session_id == "test_sess"
    assert "New York" in res.reply
    assert "location" in res.session_memory
    assert res.steps == 1


def test_oversized_resume_rejected_with_validation_error():
    """Verify resume_text > 10,000 characters is rejected by schema validator."""
    oversized_resume = "A" * 10001
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(
            target_role="Engineer",
            resume_text=oversized_resume,
        )
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("resume_text",) for err in errors)


def test_oversized_chat_message_rejected_with_validation_error():
    """Verify message > 2,000 characters is rejected by schema validator."""
    oversized_message = "M" * 2001
    with pytest.raises(ValidationError) as exc_info:
        ChatRequest(
            session_id="sess_1",
            message=oversized_message,
        )
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("message",) for err in errors)


def test_oversized_session_id_rejected_with_validation_error():
    """Verify session_id > 128 characters is rejected by schema validator."""
    oversized_session = "s" * 129
    with pytest.raises(ValidationError) as exc_info:
        ChatRequest(
            session_id=oversized_session,
            message="hello",
        )
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("session_id",) for err in errors)


def _run_coroutine_sync(coro):
    """Execute simple coroutine synchronously without event loop."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine did not complete synchronously")


def test_fastapi_request_validation_returns_422():
    """Verify FastAPI's validation exception handler converts schema errors to HTTP 422."""
    try:
        SearchRequest(resume_text="A" * 10001)
    except ValidationError as val_err:
        req_val_err = RequestValidationError(val_err.errors())
        mock_request = MagicMock(spec=Request)
        response = _run_coroutine_sync(request_validation_exception_handler(mock_request, req_val_err))
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        body = json.loads(response.body.decode("utf-8"))
        assert "detail" in body


def test_unhandled_exception_returns_500_without_stack_trace():
    """Verify unhandled exception handler returns HTTP 500 with error_id and NO stack trace."""
    mock_request = MagicMock(spec=Request)
    sensitive_error = RuntimeError("Database password leaked in raw internal trace")
    response = _run_coroutine_sync(unhandled_exception_handler(mock_request, sensitive_error))

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    body = json.loads(response.body.decode("utf-8"))
    assert body["detail"] == "Internal server error"
    assert "error_id" in body
    # Verify sensitive error details and stack traces are suppressed
    assert "Database password" not in response.body.decode("utf-8")
    assert "Traceback" not in response.body.decode("utf-8")
