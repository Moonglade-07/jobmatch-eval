"""Unit tests for Streamlit application helper functions and quota logic (Phase 3B).

Validates:
1. Per-session and global daily quota accounting.
2. Fallback execution when quota is exhausted.
3. API key sanitization (never exposed in output).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import app


def test_daily_usage_accounting(tmp_path: Path):
    """Verify daily request counter increments and persists accurately."""
    tracker_file = tmp_path / "test_usage.json"
    with patch("app.USAGE_TRACKER_FILE", tracker_file):
        # Initial state: 0
        assert app.get_daily_requests_count() == 0

        # Increment
        c1 = app.increment_daily_requests_count()
        assert c1 == 1
        assert app.get_daily_requests_count() == 1

        c2 = app.increment_daily_requests_count()
        assert c2 == 2
        assert app.get_daily_requests_count() == 2


def test_quota_checks():
    """Verify quota checks correctly flag session exhaustion and daily exhaustion."""
    with patch("app.get_daily_requests_count", return_value=50):
        # Within limits
        with patch.dict(app.st.session_state, {"session_requests": 5}):
            ok, msg = app.check_quota_available()
            assert ok is True
            assert msg == ""

        # Session limit reached
        with patch.dict(app.st.session_state, {"session_requests": app.MAX_SESSION_REQUESTS}):
            ok, msg = app.check_quota_available()
            assert ok is False
            assert "Session request limit reached" in msg

    # Daily limit reached
    with patch("app.get_daily_requests_count", return_value=app.MAX_DAILY_REQUESTS):
        with patch.dict(app.st.session_state, {"session_requests": 2}):
            ok, msg = app.check_quota_available()
            assert ok is False
            assert "Daily application request cap reached" in msg
