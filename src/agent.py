"""Job Search Agent with Function Calling, Memory, and Retrieval Tools (Phase 3A).

Implements:
1. Gemini Function Calling with 3 tools:
   - search_jobs(query, location, level, k)
   - get_job_details(job_id)
   - refine_filters(location, level, skills, dislikes)
2. Persistent Session Memory on disk (data/sessions/{session_id}.json).
3. 5-step loop cap per turn.
4. Robust error handling (unknown job_id, tool errors).
5. Grounded citations (only cite jobs returned by tools).
6. Rate limiting and disk caching.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Profile, SearchResult
from src.retrieval import BM25Retriever

DEFAULT_SESSIONS_DIR = Path("data/sessions")
DEFAULT_CACHE_DIR = Path("data/cache/agent")
DEFAULT_DATA_PATH = Path("data/jobs_clean.parquet")
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_STEPS_PER_TURN = 5

SYSTEM_PROMPT = """You are an intelligent, grounded technical job-matching career agent.
Your mission is to help candidates discover, inspect, and evaluate relevant job postings.

STRICT OPERATIONAL RULES:
1. NEVER invent, hallucinate, or assume job postings, job IDs, requirements, or company names. You may ONLY cite jobs explicitly returned by your tools.
2. If the user asks for jobs, call `search_jobs` using their target role/query and any active location or seniority filters stored in memory.
3. If the user asks about a specific job posting, call `get_job_details` with the exact job ID.
4. If the user states or updates search preferences (e.g., location, seniority level, skills, or dislikes such as 'not interested in sales'), call `refine_filters` or apply them directly to `search_jobs`.
5. Maintain conversational context across turns. If the user previously specified 'New York, NY', keep searching in 'New York, NY' unless they change it (e.g. 'now only remote' -> change location to 'Remote').
6. If a tool reports an error (e.g. job ID not found), inform the user politely and gracefully without crashing.
7. Keep responses concise, clear, and focused on technical job requirements."""

TOOL_DECLARATIONS = [
    {
        "name": "search_jobs",
        "description": "Search for relevant job postings using query terms and optional location and experience level constraints.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Search keywords or target role headline (e.g. 'Python developer', 'DevOps engineer').",
                },
                "location": {
                    "type": "STRING",
                    "description": "Target job location, e.g. 'New York, NY' or 'Remote'.",
                },
                "level": {
                    "type": "STRING",
                    "description": "Target seniority level, e.g. 'Entry level', 'Mid-Senior level', 'Director'.",
                },
                "k": {
                    "type": "INTEGER",
                    "description": "Number of top matching postings to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_job_details",
        "description": "Retrieve full details and full description for a specific job posting by its unique job ID.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "job_id": {
                    "type": "STRING",
                    "description": "The exact ID of the job posting to inspect.",
                }
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "refine_filters",
        "description": "Update and store user preferences in persistent session memory (location, seniority, skills, or dislikes).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "Preferred location or 'Remote'.",
                },
                "level": {
                    "type": "STRING",
                    "description": "Preferred seniority level (e.g. 'Entry level', 'Mid-Senior level').",
                },
                "skills": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Technical skills declared by candidate to prioritize.",
                },
                "dislikes": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Roles, domains, or terms to exclude (e.g. ['sales', 'marketing', 'consulting']).",
                },
            },
        },
    },
]


@dataclass
class ToolCallRecord:
    """Record of an executed tool call."""

    tool_name: str
    args: Dict[str, Any]
    result: Any
    error: Optional[str] = None


@dataclass
class SessionMemory:
    """Persistent session memory holding user preferences across conversation turns."""

    session_id: str
    location: Optional[str] = None
    experience_level: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)
    turn_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary."""
        return {
            "session_id": self.session_id,
            "location": self.location,
            "experience_level": self.experience_level,
            "skills": list(self.skills),
            "dislikes": list(self.dislikes),
            "turn_history": list(self.turn_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionMemory:
        """Construct SessionMemory from dictionary."""
        return cls(
            session_id=str(data.get("session_id", "default")),
            location=data.get("location"),
            experience_level=data.get("experience_level"),
            skills=list(data.get("skills", [])),
            dislikes=list(data.get("dislikes", [])),
            turn_history=list(data.get("turn_history", [])),
        )

    def save(self, directory: Path) -> Path:
        """Persist session memory to JSON file on disk."""
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"{self.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return file_path

    @classmethod
    def load(cls, session_id: str, directory: Path) -> SessionMemory:
        """Load session memory from disk if present, else create new."""
        file_path = directory / f"{session_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return cls.from_dict(json.load(f))
            except Exception:
                pass
        return cls(session_id=session_id)


@dataclass
class AgentTurnResult:
    """Outcome of an agent interaction turn."""

    session_id: str
    reply: str
    tool_calls: List[ToolCallRecord]
    session_memory: Dict[str, Any]
    steps: int
    latency_ms: float
    cache_hit: bool = False
    error: Optional[str] = None


class JobSearchAgent:
    """Technical Job-Search Career Agent with Gemini Function Calling & Tool Loop."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        sessions_dir: Path = DEFAULT_SESSIONS_DIR,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        data_path: Path = DEFAULT_DATA_PATH,
        max_steps: int = MAX_STEPS_PER_TURN,
        mock_handler: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            api_key: Gemini API key (defaults to .env).
            model_name: Gemini model string.
            sessions_dir: Directory for persistent session JSON files.
            cache_dir: Directory for model response disk cache.
            data_path: Path to clean job postings parquet.
            max_steps: Maximum tool loop steps per turn (capped at 5).
            mock_handler: Optional custom handler for unit testing without live API calls.
        """
        self.api_key = api_key or self._load_env_var("GEMINI_API_KEY")
        self.model_name = model_name or self._load_env_var("GEMINI_MODEL") or DEFAULT_MODEL
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = min(max_steps, MAX_STEPS_PER_TURN)
        self.mock_handler = mock_handler

        # Load jobs dataset for details tool
        if data_path.exists():
            self.jobs_df = pd.read_parquet(data_path)
            self.jobs_lookup = {
                str(r["id"]).strip(): r for r in self.jobs_df.to_dict("records")
            }
        else:
            self.jobs_df = pd.DataFrame()
            self.jobs_lookup = {}

        # Best retrieval strategy (BM25 Only as verified in Phase 2B)
        self._retriever: Optional[BM25Retriever] = None

    @property
    def retriever(self) -> BM25Retriever:
        """Lazy-initialize BM25 retriever."""
        if self._retriever is None:
            self._retriever = BM25Retriever(df=self.jobs_df)
        return self._retriever

    @staticmethod
    def _load_env_var(name: str) -> Optional[str]:
        """Load environment variable from os.environ or .env file."""
        if name in os.environ:
            return os.environ[name]
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == name:
                        return v.strip()
        return None

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def execute_tool_search_jobs(
        self,
        query: str,
        location: Optional[str] = None,
        level: Optional[str] = None,
        k: int = 5,
        memory: Optional[SessionMemory] = None,
    ) -> List[Dict[str, Any]]:
        """Tool 1: Execute BM25 search over job postings with active memory constraints."""
        eff_location = location or (memory.location if memory else "") or ""
        eff_level = level or (memory.experience_level if memory else "") or ""
        skills = memory.skills if memory else []

        profile = Profile(
            resume_text="",
            target_role=query,
            preferred_location=eff_location,
            seniority=eff_level,
            skills=skills,
            profile_id="agent_query",
        )

        # Retrieve a broader candidate set to allow filtering of dislikes
        fetch_k = max(k * 3, 15)
        raw_results = self.retriever.search(
            profile=profile,
            k=fetch_k,
            apply_filters=True,
            apply_boost=True,
            boost_weight=0.2,
        )

        dislikes = [d.lower() for d in (memory.dislikes if memory else [])]

        filtered_results = []
        for r in raw_results:
            title_lower = r.title.lower()
            desc = getattr(r, "description", "") or ""
            desc_lower = desc.lower()
            # Exclude if disliked term appears in title
            if any(bad_word in title_lower for bad_word in dislikes):
                continue
            filtered_results.append({
                "job_id": r.job_id,
                "title": r.title,
                "company": r.company,
                "location": r.location,
                "experience_level": r.experience_level,
                "snippet": desc[:250] + ("..." if len(desc) > 250 else ""),
            })
            if len(filtered_results) >= k:
                break

        return filtered_results

    def execute_tool_get_job_details(self, job_id: str) -> Dict[str, Any]:
        """Tool 2: Retrieve full posting details by job ID."""
        clean_id = str(job_id).strip()
        job = self.jobs_lookup.get(clean_id)
        if not job:
            return {
                "error": f"Job ID '{clean_id}' not found in the dataset.",
                "job_id": clean_id,
            }
        return {
            "job_id": str(job["id"]),
            "title": str(job["title"]),
            "company": str(job["company"]),
            "location": str(job["location"]),
            "experience_level": str(job["experience_level"]),
            "description": str(job["description"]),
        }

    def execute_tool_refine_filters(
        self,
        location: Optional[str] = None,
        level: Optional[str] = None,
        skills: Optional[List[str]] = None,
        dislikes: Optional[List[str]] = None,
        memory: Optional[SessionMemory] = None,
    ) -> Dict[str, Any]:
        """Tool 3: Update session memory with explicit user preferences."""
        if memory is not None:
            if location:
                memory.location = location
            if level:
                memory.experience_level = level
            if skills:
                for s in skills:
                    if s not in memory.skills:
                        memory.skills.append(s)
            if dislikes:
                for d in dislikes:
                    if d not in memory.dislikes:
                        memory.dislikes.append(d)

        return {
            "status": "preferences_updated",
            "current_filters": memory.to_dict() if memory else {},
        }

    # -------------------------------------------------------------------------
    # Conversational Memory & Heuristic Synchronization
    # -------------------------------------------------------------------------

    def _sync_memory_from_user_message(
        self, message: str, memory: SessionMemory
    ) -> None:
        """Heuristically synchronize session memory from implicit user statements."""
        msg_lower = message.lower()

        # Location overrides
        if "only remote" in msg_lower or "remote only" in msg_lower or "switch to remote" in msg_lower:
            memory.location = "Remote"
        elif "new york" in msg_lower:
            memory.location = "New York, NY"
        elif "chicago" in msg_lower:
            memory.location = "Chicago, IL"
        elif "houston" in msg_lower:
            memory.location = "Houston, TX"
        elif "dallas" in msg_lower:
            memory.location = "Dallas, TX"
        elif "seattle" in msg_lower:
            memory.location = "Seattle, WA"
        elif "boston" in msg_lower:
            memory.location = "Boston, MA"

        # Seniority overrides
        if "no senior" in msg_lower or "entry level" in msg_lower or "recent grad" in msg_lower:
            memory.experience_level = "Entry level"
        elif "mid-senior" in msg_lower or "mid senior" in msg_lower:
            memory.experience_level = "Mid-Senior level"
        elif "director" in msg_lower:
            memory.experience_level = "Director"

        # Dislikes
        if "not interested in sales" in msg_lower or "no sales" in msg_lower or "exclude sales" in msg_lower:
            if "sales" not in memory.dislikes:
                memory.dislikes.append("sales")
        if "marketing" in msg_lower and ("not interested" in msg_lower or "no " in msg_lower):
            if "marketing" not in memory.dislikes:
                memory.dislikes.append("marketing")
        if "consulting" in msg_lower and ("don't want" in msg_lower or "no " in msg_lower or "not interested" in msg_lower):
            if "consulting" not in memory.dislikes:
                memory.dislikes.append("consulting")

        # Explicit technical skills
        for tech in ["Python", "Django", "React", "TypeScript", "SQL", "Tableau", "AWS", "Kubernetes", "Docker", "Machine Learning", "PyTorch"]:
            if re.search(rf"\b{re.escape(tech)}\b", message, re.IGNORECASE):
                if tech not in memory.skills:
                    memory.skills.append(tech)

    # -------------------------------------------------------------------------
    # Model API Call & Tool Loop Execution
    # -------------------------------------------------------------------------

    def _call_gemini_api(
        self, contents: List[Dict[str, Any]]
    ) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        """Call Gemini API generateContent endpoint with function calling declarations."""
        if not self.api_key:
            return 401, None, "GEMINI_API_KEY is not configured in .env."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
            "generationConfig": {
                "temperature": 0.2,
            },
        }

        delay = 2.0
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=30.0)
                if resp.status_code == 200:
                    return 200, resp.json(), None
                elif resp.status_code in (429, 500, 503):
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    return resp.status_code, None, f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                time.sleep(delay)
                delay *= 2.0
        return 500, None, "Gemini API request failed after retries."

    def chat(self, session_id: str, message: str) -> AgentTurnResult:
        """Execute a conversational interaction turn with persistent session memory and tool loop.

        Args:
            session_id: Unique identifier for the user session.
            message: User's chat message.

        Returns:
            AgentTurnResult containing reply, tool calls, and memory state.
        """
        t0 = time.perf_counter()
        memory = SessionMemory.load(session_id=session_id, directory=self.sessions_dir)

        # Synchronize memory state from user message
        self._sync_memory_from_user_message(message, memory)

        tool_calls_executed: List[ToolCallRecord] = []
        cache_hit = False

        # Build contents from prior turn history + current user message
        contents: List[Dict[str, Any]] = []
        for prior in memory.turn_history[-6:]:  # Keep recent turns in context
            role = prior.get("role", "user")
            text = prior.get("text", "")
            if text:
                contents.append({"role": role, "parts": [{"text": text}]})

        contents.append({"role": "user", "parts": [{"text": message}]})

        step_count = 0
        final_reply = ""
        turn_error = None

        # Execute Tool Loop (Capped at MAX_STEPS_PER_TURN)
        while step_count < self.max_steps:
            step_count += 1

            # Check mock handler if provided
            if self.mock_handler is not None:
                resp_json = self.mock_handler(contents)
                status_code = 200
                err = None
            else:
                # Disk cache lookup
                cache_key = hashlib.sha256(
                    json.dumps(contents, sort_keys=True).encode("utf-8")
                ).hexdigest()
                cache_file = self.cache_dir / f"{cache_key}.json"

                if cache_file.exists():
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            resp_json = json.load(f)
                        status_code = 200
                        err = None
                        cache_hit = True
                    except Exception:
                        status_code, resp_json, err = self._call_gemini_api(contents)
                else:
                    status_code, resp_json, err = self._call_gemini_api(contents)
                    if status_code == 200 and resp_json:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(resp_json, f, indent=2)

            if status_code != 200 or not resp_json:
                turn_error = err or f"Model call failed with status {status_code}"
                final_reply = (
                    "I encountered a temporary service error while processing your request. "
                    "Please try again shortly."
                )
                break

            candidates = resp_json.get("candidates", [])
            if not candidates:
                final_reply = "I was unable to generate a response. Please try rephrasing."
                break

            candidate_content = candidates[0].get("content", {})
            parts = candidate_content.get("parts", [])

            # Check if model requested a function call
            func_call = None
            text_parts = []
            for p in parts:
                if "functionCall" in p:
                    func_call = p["functionCall"]
                if "text" in p and p["text"]:
                    text_parts.append(p["text"])

            if not func_call:
                # Model produced final text response without tool call
                final_reply = " ".join(text_parts).strip()
                break

            # Execute requested tool
            fn_name = func_call.get("name", "")
            fn_args = func_call.get("args", {})

            tool_record = ToolCallRecord(tool_name=fn_name, args=fn_args, result=None)

            try:
                if fn_name == "search_jobs":
                    query_arg = fn_args.get("query", message)
                    loc_arg = fn_args.get("location")
                    lvl_arg = fn_args.get("level")
                    k_arg = int(fn_args.get("k", 5))
                    t_result = self.execute_tool_search_jobs(
                        query=query_arg,
                        location=loc_arg,
                        level=lvl_arg,
                        k=k_arg,
                        memory=memory,
                    )
                elif fn_name == "get_job_details":
                    jid_arg = str(fn_args.get("job_id", "")).strip()
                    t_result = self.execute_tool_get_job_details(job_id=jid_arg)
                    if "error" in t_result:
                        tool_record.error = t_result["error"]
                elif fn_name == "refine_filters":
                    t_result = self.execute_tool_refine_filters(
                        location=fn_args.get("location"),
                        level=fn_args.get("level"),
                        skills=fn_args.get("skills"),
                        dislikes=fn_args.get("dislikes"),
                        memory=memory,
                    )
                else:
                    t_result = {"error": f"Unknown tool: '{fn_name}'"}
                    tool_record.error = t_result["error"]
            except Exception as e:
                t_result = {"error": f"Exception executing {fn_name}: {str(e)}"}
                tool_record.error = str(e)

            tool_record.result = t_result
            tool_calls_executed.append(tool_record)

            # Append assistant's functionCall turn (preserving thoughtSignature and IDs)
            contents.append(copy.deepcopy(candidate_content))

            # Append functionResponse turn
            fn_resp_obj: Dict[str, Any] = {
                "name": fn_name,
                "response": {"name": fn_name, "content": t_result},
            }
            if "id" in func_call:
                fn_resp_obj["id"] = func_call["id"]

            contents.append({
                "role": "user",
                "parts": [{"functionResponse": fn_resp_obj}],
            })

        if not final_reply and tool_calls_executed:
            # If loop reached max_steps without free text, generate a fallback summary
            last_tc = tool_calls_executed[-1]
            if last_tc.tool_name == "search_jobs" and isinstance(last_tc.result, list):
                jobs_count = len(last_tc.result)
                final_reply = f"I found {jobs_count} matching positions based on your criteria."
            else:
                final_reply = "I have updated your preferences and retrieved the latest job data."

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Update and persist session memory turn history
        memory.turn_history.append({"role": "user", "text": message})
        memory.turn_history.append({"role": "model", "text": final_reply})
        memory.save(self.sessions_dir)

        return AgentTurnResult(
            session_id=session_id,
            reply=final_reply,
            tool_calls=tool_calls_executed,
            session_memory=memory.to_dict(),
            steps=step_count,
            latency_ms=round(latency_ms, 2),
            cache_hit=cache_hit,
            error=turn_error,
        )
