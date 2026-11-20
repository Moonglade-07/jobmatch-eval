"""Unit tests for JobSearchAgent and SessionMemory (Phase 3A).

All tests mock model responses to guarantee zero live API calls in the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.agent import JobSearchAgent, SessionMemory, DEFAULT_MODEL, MAX_STEPS_PER_TURN
from src.models import SearchResult


@pytest.fixture
def temp_agent_dir(tmp_path: Path):
    """Fixture providing isolated directories and mock jobs dataset."""
    sessions_dir = tmp_path / "sessions"
    cache_dir = tmp_path / "cache"
    data_file = tmp_path / "test_jobs.parquet"

    # Create small test dataset
    df = pd.DataFrame([
        {
            "id": "1000000001",
            "title": "Senior Python Backend Engineer",
            "company": "TechCorp",
            "location": "New York, NY",
            "experience_level": "Mid-Senior level",
            "description": "Looking for Python, Django, PostgreSQL, Docker, AWS.",
        },
        {
            "id": "1000000002",
            "title": "Enterprise Sales Representative",
            "company": "SaaSCo",
            "location": "New York, NY",
            "experience_level": "Mid-Senior level",
            "description": "Enterprise software sales, cold calling, quotas.",
        },
        {
            "id": "1000000003",
            "title": "Junior Python Developer",
            "company": "StartUp",
            "location": "Remote",
            "experience_level": "Entry level",
            "description": "Entry level Python developer with Git and SQL.",
        },
    ])
    df.to_parquet(data_file)

    return {
        "sessions_dir": sessions_dir,
        "cache_dir": cache_dir,
        "data_file": data_file,
    }


def test_session_memory_persistence(temp_agent_dir):
    """Verify session memory saves to disk, reloads, and retains state."""
    sessions_dir = temp_agent_dir["sessions_dir"]
    session_id = "test_sess_01"

    memory = SessionMemory(session_id=session_id)
    memory.location = "San Francisco, CA"
    memory.experience_level = "Entry level"
    memory.skills.append("Python")
    memory.dislikes.append("sales")
    memory.turn_history.append({"role": "user", "text": "Hello"})

    saved_path = memory.save(sessions_dir)
    assert saved_path.exists()

    # Load in new instance
    loaded = SessionMemory.load(session_id=session_id, directory=sessions_dir)
    assert loaded.session_id == session_id
    assert loaded.location == "San Francisco, CA"
    assert loaded.experience_level == "Entry level"
    assert "Python" in loaded.skills
    assert "sales" in loaded.dislikes
    assert len(loaded.turn_history) == 1


def test_agent_loop_cap_enforcement(temp_agent_dir):
    """Verify the agent tool loop never exceeds MAX_STEPS_PER_TURN (5) even if the model loops."""
    # Mock handler that relentlessly asks for function calls
    def looping_mock_handler(contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "refine_filters",
                                    "args": {"skills": ["Python"]},
                                }
                            }
                        ]
                    }
                }
            ]
        }

    agent = JobSearchAgent(
        api_key="mock_key",
        sessions_dir=temp_agent_dir["sessions_dir"],
        cache_dir=temp_agent_dir["cache_dir"],
        data_path=temp_agent_dir["data_file"],
        mock_handler=looping_mock_handler,
    )

    result = agent.chat(session_id="loop_test_sess", message="Keep calling tools")

    # Assert loop was capped at max_steps (5)
    assert result.steps == MAX_STEPS_PER_TURN
    assert len(result.tool_calls) == MAX_STEPS_PER_TURN
    assert result.reply != ""  # Produced fallback reply without crashing


def test_agent_unknown_job_id_handled_gracefully(temp_agent_dir):
    """Verify get_job_details with non-existent job ID returns helpful error, not crash."""
    agent = JobSearchAgent(
        api_key="mock_key",
        sessions_dir=temp_agent_dir["sessions_dir"],
        cache_dir=temp_agent_dir["cache_dir"],
        data_path=temp_agent_dir["data_file"],
    )

    res = agent.execute_tool_get_job_details("9999999999")
    assert "error" in res
    assert "not found in the dataset" in res["error"]
    assert res["job_id"] == "9999999999"


def test_agent_tool_error_handling_in_conversation(temp_agent_dir):
    """Verify agent handles unknown job ID requested by model without raising an exception."""
    turn_counter = 0

    def mock_missing_job_flow(contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal turn_counter
        turn_counter += 1
        if turn_counter == 1:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_job_details",
                                        "args": {"job_id": "invalid_id_999"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "I checked our database, but job posting invalid_id_999 was not found."
                                }
                            ]
                        }
                    }
                ]
            }

    agent = JobSearchAgent(
        api_key="mock_key",
        sessions_dir=temp_agent_dir["sessions_dir"],
        cache_dir=temp_agent_dir["cache_dir"],
        data_path=temp_agent_dir["data_file"],
        mock_handler=mock_missing_job_flow,
    )

    result = agent.chat(session_id="err_sess", message="Show me job invalid_id_999")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "get_job_details"
    assert result.tool_calls[0].error is not None
    assert "not found" in result.reply.lower()


def test_agent_dislike_filtering_in_search(temp_agent_dir):
    """Verify search_jobs filters out postings matching user's declared dislikes."""
    agent = JobSearchAgent(
        api_key="mock_key",
        sessions_dir=temp_agent_dir["sessions_dir"],
        cache_dir=temp_agent_dir["cache_dir"],
        data_path=temp_agent_dir["data_file"],
    )

    memory = SessionMemory(session_id="dislike_sess")
    memory.dislikes.append("sales")

    results = agent.execute_tool_search_jobs(
        query="Engineer",
        k=5,
        memory=memory,
    )

    titles = [r["title"].lower() for r in results]
    assert not any("sales" in t for t in titles)
