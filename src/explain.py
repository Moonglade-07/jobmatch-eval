"""LLM Explanation Generation & Groundedness Evaluation Engine (Phase 2C).

Evaluates LLM-generated match explanations for validity, literal groundedness,
profile support, and contradiction leakage without relying on an LLM judge.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
PROMPT_VERSION = "explain-v1.0"
DEFAULT_CACHE_DIR = Path("data/cache/explanations")
MANDATORY_CAPTION = "Programmatic checks against job text; no human or LLM judge"
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are an expert career alignment evaluator and technical recruiter.
Your task is to analyze candidate suitability for a specific job posting and provide an explainable summary in valid JSON format.

CRITICAL INSTRUCTIONS:
1. matched_skills: List ONLY technical skills, tools, or domain qualifications from the candidate's profile that LITERALLY appear in the job posting text. Do NOT assume, extrapolate, or hallucinate skills not explicitly written in the job description or title.
2. missing_skills: List candidate skills that do not appear in the job posting, or key technical requirements from the job posting that the candidate lacks.
3. summary: Exactly ONE clear, concise sentence summarizing the alignment between the candidate and the job. Do NOT output more than one sentence.
4. Output format: You must return valid raw JSON only, matching the schema:
{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "summary": "One sentence summary."
}
"""

USER_PROMPT_TEMPLATE = """CANDIDATE PROFILE:
Target Role: {target_role}
Seniority: {seniority}
Preferred Location: {preferred_location}
Declared Skills: {skills}
Resume Narrative:
{resume_text}

JOB POSTING:
Job ID: {job_id}
Title: {title}
Company: {company}
Location: {location}
Seniority: {experience_level}
Job Description:
{description}

Analyze the candidate profile against the job posting. Return ONLY raw JSON with keys: matched_skills, missing_skills, summary.
"""

REPAIR_PROMPT_TEMPLATE = """Your previous response was invalid.
Validation Error Encountered:
{error_message}

Your Previous Raw Response:
{raw_response}

Please correct the output. Return ONLY valid raw JSON matching the required schema:
{{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "summary": "One sentence summary."
}}
Remember: "summary" must be EXACTLY ONE sentence.
"""


# ---------------------------------------------------------------------------
# Pydantic Schema
# ---------------------------------------------------------------------------
class Explanation(BaseModel):
    """Structured explanation for candidate-to-job matching."""

    matched_skills: List[str] = Field(
        default_factory=list,
        description="Candidate skills literally appearing in the job posting text.",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Candidate skills missing in the posting or missing job requirements.",
    )
    summary: str = Field(
        ...,
        description="Exactly one sentence summarizing the candidate-job fit.",
    )

    @field_validator("summary")
    @classmethod
    def validate_single_sentence(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Summary must not be empty.")
        # Splitting on sentence terminators followed by whitespace and uppercase/digit
        sentences = [p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", s) if p.strip()]
        if len(sentences) > 1:
            raise ValueError(
                f"Summary must be exactly one sentence, but found {len(sentences)} sentences: {sentences}"
            )
        return s


class ExplainResult(BaseModel):
    """Complete result container for an explanation attempt."""

    profile_id: str
    job_id: str
    explanation: Optional[Explanation] = None
    raw_response: Optional[str] = None
    cache_hit: bool = False
    retry_count: int = 0
    fallback: bool = False
    fallback_reason: Optional[str] = None  # invalid_json, validation_error, api_error, timeout, quota
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    candidate_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str = DEFAULT_MODEL
    prompt_version: str = PROMPT_VERSION


# ---------------------------------------------------------------------------
# Groundedness & String Matching Utilities
# ---------------------------------------------------------------------------
def is_skill_grounded_regex(skill: str, text: str) -> bool:
    """Check if skill literally appears in text using boundary-safe regex.

    Handles symbols such as C++, C#, .NET while preventing partial word matches
    (e.g., 'C' in 'Cloud' or 'Java' in 'JavaScript').

    Args:
        skill: Skill name string.
        text: Target text to inspect.

    Returns:
        True if the skill appears with safe word boundaries.
    """
    cleaned = skill.strip()
    if not cleaned or not text:
        return False

    escaped = re.escape(cleaned)
    pattern = rf"(?<![\w\+\#]){escaped}(?![\w\+\#])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def is_skill_grounded_strict(skill: str, text: str) -> bool:
    """Check if skill appears in text using exact case-sensitive substring matching.

    Args:
        skill: Skill name string.
        text: Target text to inspect.

    Returns:
        True if skill exists verbatim in text.
    """
    cleaned = skill.strip()
    if not cleaned or not text:
        return False
    return cleaned in text


def evaluate_claim_groundedness(
    matched_skills: List[str], job_text: str
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate groundedness (regex-normalized and strict) for a list of matched skills.

    Args:
        matched_skills: Skills claimed by the model as matching.
        job_text: Full job posting text.

    Returns:
        Tuple of (grounded_ratio, strict_grounded_ratio, per_skill_detail).
    """
    if not matched_skills:
        return 1.0, 1.0, []

    details = []
    grounded_count = 0
    strict_count = 0

    for s in matched_skills:
        g = is_skill_grounded_regex(s, job_text)
        gs = is_skill_grounded_strict(s, job_text)
        if g:
            grounded_count += 1
        if gs:
            strict_count += 1
        details.append({"skill": s, "grounded": g, "grounded_strict": gs})

    return grounded_count / len(matched_skills), strict_count / len(matched_skills), details


def evaluate_profile_support(matched_skills: List[str], profile_text: str) -> float:
    """Evaluate whether claimed matched skills actually exist in candidate profile.

    Args:
        matched_skills: Skills claimed as matching by LLM.
        profile_text: Profile skills and resume text.

    Returns:
        Fraction of claimed skills supported by candidate profile.
    """
    if not matched_skills:
        return 1.0
    supported = sum(1 for s in matched_skills if is_skill_grounded_regex(s, profile_text))
    return supported / len(matched_skills)


def evaluate_missing_skills_leakage(
    missing_skills: List[str], job_text: str, profile_text: str
) -> float:
    """Evaluate contradiction leakage: missing_skills that appear in BOTH job and profile.

    Args:
        missing_skills: Skills claimed as missing by LLM.
        job_text: Job posting text.
        profile_text: Candidate profile text.

    Returns:
        Fraction of missing_skills that contradictorily appear in both texts.
    """
    if not missing_skills:
        return 0.0

    contradictions = 0
    for s in missing_skills:
        in_job = is_skill_grounded_regex(s, job_text)
        in_profile = is_skill_grounded_regex(s, profile_text)
        if in_job and in_profile:
            contradictions += 1

    return contradictions / len(missing_skills)


# ---------------------------------------------------------------------------
# Gemini Explainer Client
# ---------------------------------------------------------------------------
class GeminiExplainer:
    """Production explainer client managing Gemini API calls, caching, and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        prompt_version: str = PROMPT_VERSION,
        request_delay: float = 1.0,
        timeout: float = 25.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize Gemini explainer.

        Args:
            api_key: Gemini API key. If None, loaded from .env.
            model_name: Model identifier. If None, loaded from .env or default.
            cache_dir: Directory for on-disk response caching.
            prompt_version: Prompt version string.
            request_delay: Delay in seconds between live API requests.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum exponential backoff retries for transient HTTP errors.
        """
        self.api_key = api_key or self._load_env_var("GEMINI_API_KEY")
        self.model_name = model_name or self._load_env_var("GEMINI_MODEL") or DEFAULT_MODEL
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_version = prompt_version
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def _load_env_var(name: str) -> Optional[str]:
        """Load variable from os.environ or .env file."""
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

    def _get_cache_path(self, profile_id: str, job_id: str) -> Path:
        """Compute deterministic on-disk cache path."""
        composite = f"{profile_id}_{job_id}_{self.model_name}_{self.prompt_version}"
        key_hash = hashlib.sha256(composite.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def _send_generate_content(
        self, contents: List[Dict[str, Any]], system_prompt: str = SYSTEM_INSTRUCTION
    ) -> Tuple[int, Optional[Dict[str, Any]], float, Optional[str]]:
        """Send generateContent request to Gemini REST API with exponential backoff.

        Returns:
            Tuple of (status_code, response_json, latency_ms, error_string).
        """
        if not self.api_key:
            return 401, None, 0.0, "GEMINI_API_KEY is not configured in .env."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
            },
        }

        delay = 2.0
        last_error = ""
        total_latency = 0.0

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                call_latency = (time.perf_counter() - t0) * 1000.0
                total_latency += call_latency

                if resp.status_code == 200:
                    return 200, resp.json(), call_latency, None

                # Quota / rate limit / server error handling
                status = resp.status_code
                err_text = resp.text
                last_error = f"HTTP {status}: {err_text[:200]}"

                if status == 429:
                    if "RESOURCE_EXHAUSTED" in err_text or "quota" in err_text.lower():
                        if "daily" in err_text.lower():
                            return 429, None, total_latency, "quota_daily_exhausted"
                    # Rate limit jittered sleep
                    sleep_time = delay + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                    delay *= 2
                    continue
                elif status in (500, 503, 504):
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    return status, None, call_latency, last_error

            except requests.exceptions.Timeout:
                call_latency = (time.perf_counter() - t0) * 1000.0
                total_latency += call_latency
                last_error = "timeout"
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    return 408, None, total_latency, "timeout"
            except requests.exceptions.RequestException as e:
                call_latency = (time.perf_counter() - t0) * 1000.0
                total_latency += call_latency
                last_error = f"network_error: {str(e)}"
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    return 500, None, total_latency, last_error

        return 500, None, total_latency, last_error

    def explain(
        self,
        profile_data: Dict[str, Any],
        job_data: Dict[str, Any],
        use_cache: bool = True,
    ) -> ExplainResult:
        """Generate structured explanation for candidate profile and job.

        Executes initial call, validates with Pydantic, retries ONCE with repair prompt
        on validation/JSON error, and falls back cleanly on terminal failures.

        Args:
            profile_data: Candidate profile dictionary.
            job_data: Job posting dictionary.
            use_cache: Whether to read/write persistent disk cache.

        Returns:
            ExplainResult populated with explanation or fallback flag.
        """
        pid = profile_data.get("profile_id", "unknown")
        jid = str(job_data.get("id") or job_data.get("job_id", "unknown"))

        cache_path = self._get_cache_path(pid, jid)

        # 1. Disk Cache Lookup
        if use_cache and cache_path.exists():
            try:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                expl = Explanation(**cached_data["explanation"]) if cached_data.get("explanation") else None
                return ExplainResult(
                    profile_id=pid,
                    job_id=jid,
                    explanation=expl,
                    raw_response=cached_data.get("raw_response"),
                    cache_hit=True,
                    retry_count=cached_data.get("retry_count", 0),
                    fallback=cached_data.get("fallback", False),
                    fallback_reason=cached_data.get("fallback_reason"),
                    latency_ms=cached_data.get("latency_ms", 0.0),
                    prompt_tokens=cached_data.get("prompt_tokens"),
                    candidate_tokens=cached_data.get("candidate_tokens"),
                    total_tokens=cached_data.get("total_tokens"),
                    model=self.model_name,
                    prompt_version=self.prompt_version,
                )
            except Exception:
                # If cache corrupted, proceed to live call
                pass

        # 2. Build Initial User Prompt
        skills_str = ", ".join(profile_data.get("skills", []))
        user_prompt = USER_PROMPT_TEMPLATE.format(
            target_role=profile_data.get("target_role", ""),
            seniority=profile_data.get("seniority", ""),
            preferred_location=profile_data.get("preferred_location", ""),
            skills=skills_str,
            resume_text=profile_data.get("resume_text", ""),
            job_id=jid,
            title=job_data.get("title", ""),
            company=job_data.get("company", ""),
            location=job_data.get("location", ""),
            experience_level=job_data.get("experience_level", ""),
            description=job_data.get("description", "")[:4000],
        )

        contents = [{"role": "user", "parts": [{"text": user_prompt}]}]

        # Live API Call 1
        status, resp_json, latency_1, err_str = self._send_generate_content(contents)

        if self.request_delay > 0:
            time.sleep(self.request_delay)

        if status != 200 or not resp_json:
            reason = "quota" if "quota" in str(err_str) else ("timeout" if err_str == "timeout" else "api_error")
            res = ExplainResult(
                profile_id=pid,
                job_id=jid,
                fallback=True,
                fallback_reason=reason,
                latency_ms=latency_1,
                model=self.model_name,
                prompt_version=self.prompt_version,
            )
            return res

        # Extract tokens and text from call 1
        usage = resp_json.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        candidate_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")

        candidates = resp_json.get("candidates", [])
        if not candidates or "content" not in candidates[0]:
            return ExplainResult(
                profile_id=pid,
                job_id=jid,
                fallback=True,
                fallback_reason="api_error",
                latency_ms=latency_1,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
                model=self.model_name,
                prompt_version=self.prompt_version,
            )

        raw_text = candidates[0]["content"]["parts"][0].get("text", "").strip()

        # Try parsing JSON and validating Pydantic
        parse_error = None
        try:
            parsed = json.loads(raw_text)
            explanation = Explanation(**parsed)
            # Success on first attempt!
            res = ExplainResult(
                profile_id=pid,
                job_id=jid,
                explanation=explanation,
                raw_response=raw_text,
                cache_hit=False,
                retry_count=0,
                fallback=False,
                latency_ms=latency_1,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
                model=self.model_name,
                prompt_version=self.prompt_version,
            )
            self._save_cache(cache_path, res)
            return res
        except json.JSONDecodeError as e:
            parse_error = f"Invalid JSON: {str(e)}"
            failure_type = "invalid_json"
        except ValidationError as e:
            parse_error = f"Pydantic Validation Error: {str(e)}"
            failure_type = "validation_error"

        # 3. Exactly ONE Retry with Repair Prompt
        repair_msg = REPAIR_PROMPT_TEMPLATE.format(
            error_message=parse_error, raw_response=raw_text
        )
        retry_contents = [
            {"role": "user", "parts": [{"text": user_prompt}]},
            {"role": "model", "parts": [{"text": raw_text}]},
            {"role": "user", "parts": [{"text": repair_msg}]},
        ]

        status_2, resp_json_2, latency_2, err_str_2 = self._send_generate_content(retry_contents)
        total_lat = latency_1 + latency_2

        if self.request_delay > 0:
            time.sleep(self.request_delay)

        if status_2 != 200 or not resp_json_2:
            reason = "quota" if "quota" in str(err_str_2) else ("timeout" if err_str_2 == "timeout" else "api_error")
            res = ExplainResult(
                profile_id=pid,
                job_id=jid,
                raw_response=raw_text,
                retry_count=1,
                fallback=True,
                fallback_reason=reason,
                latency_ms=total_lat,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
                model=self.model_name,
                prompt_version=self.prompt_version,
            )
            self._save_cache(cache_path, res)
            return res

        # Extract tokens from retry
        usage_2 = resp_json_2.get("usageMetadata", {})
        if usage_2:
            prompt_tokens = (prompt_tokens or 0) + usage_2.get("promptTokenCount", 0)
            candidate_tokens = (candidate_tokens or 0) + usage_2.get("candidatesTokenCount", 0)
            total_tokens = (total_tokens or 0) + usage_2.get("totalTokenCount", 0)

        candidates_2 = resp_json_2.get("candidates", [])
        if not candidates_2:
            res = ExplainResult(
                profile_id=pid,
                job_id=jid,
                raw_response=raw_text,
                retry_count=1,
                fallback=True,
                fallback_reason="api_error",
                latency_ms=total_lat,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
                model=self.model_name,
                prompt_version=self.prompt_version,
            )
            self._save_cache(cache_path, res)
            return res

        raw_text_2 = candidates_2[0]["content"]["parts"][0].get("text", "").strip()

        try:
            parsed_2 = json.loads(raw_text_2)
            explanation_2 = Explanation(**parsed_2)
            res = ExplainResult(
                profile_id=pid,
                job_id=jid,
                explanation=explanation_2,
                raw_response=raw_text_2,
                cache_hit=False,
                retry_count=1,
                fallback=False,
                latency_ms=total_lat,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
                model=self.model_name,
                prompt_version=self.prompt_version,
            )
            self._save_cache(cache_path, res)
            return res
        except (json.JSONDecodeError, ValidationError) as e:
            final_reason = "invalid_json" if isinstance(e, json.JSONDecodeError) else "validation_error"
            res = ExplainResult(
                profile_id=pid,
                job_id=jid,
                raw_response=raw_text_2,
                retry_count=1,
                fallback=True,
                fallback_reason=final_reason,
                latency_ms=total_lat,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
                model=self.model_name,
                prompt_version=self.prompt_version,
            )
            self._save_cache(cache_path, res)
            return res

    @staticmethod
    def _save_cache(path: Path, result: ExplainResult) -> None:
        """Persist result safely to disk cache."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = result.model_dump()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Programmatic Evaluation & Bootstrap Metrics (No LLM Judge)
# ---------------------------------------------------------------------------
def compute_bootstrap_profile_ci(
    profile_values: List[float], n_resamples: int = 1000, seed: int = 42
) -> Optional[List[float]]:
    """Calculate 95% bootstrap CI resampling over profiles (clusters).

    Args:
        profile_values: Per-profile aggregated metric values.
        n_resamples: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        List of [ci_lower, ci_upper] if variance exists, else None ("n too small").
    """
    if not profile_values or len(profile_values) < 2:
        return None

    arr = np.array(profile_values, dtype=float)
    if float(np.var(arr)) == 0.0 or np.isnan(float(np.var(arr))):
        return None

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=float)
    n = len(arr)

    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return [round(low, 4), round(high, 4)]


def format_ci(ci: Optional[List[float]]) -> str:
    """Format bootstrap CI for reporting tables, returning 'n too small' when no variance."""
    if ci is None:
        return "n too small"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def evaluate_explanations(
    results: List[ExplainResult],
    profiles_dict: Dict[str, Any],
    jobs_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute programmatic groundedness, validity, leakage, and latency metrics.

    Args:
        results: List of ExplainResult objects.
        profiles_dict: Mapping of profile_id to profile data dict.
        jobs_dict: Mapping of job_id to job posting data dict.

    Returns:
        Comprehensive metric dictionary matching Phase 2C specification.
    """
    n_calls = len(results)
    if n_calls == 0:
        return {"error": "No explanation results provided"}

    # Group results by profile_id
    by_profile: Dict[str, List[ExplainResult]] = {}
    for r in results:
        by_profile.setdefault(r.profile_id, []).append(r)

    n_profiles = len(by_profile)

    # 1. Validity and Fallback Counters
    first_try_valid = sum(1 for r in results if not r.fallback and r.retry_count == 0)
    retry_valid = sum(1 for r in results if not r.fallback and r.retry_count == 1)
    retries_needed = sum(1 for r in results if r.retry_count > 0)
    valid_after_retry = first_try_valid + retry_valid  # total valid results after allowed retry
    fallbacks = [r for r in results if r.fallback]
    n_fallbacks = len(fallbacks)

    fallback_breakdown: Dict[str, int] = {
        "invalid_json": 0,
        "validation_error": 0,
        "api_error": 0,
        "timeout": 0,
        "quota": 0,
    }
    for f in fallbacks:
        reason = f.fallback_reason or "api_error"
        fallback_breakdown[reason] = fallback_breakdown.get(reason, 0) + 1

    first_try_rate = first_try_valid / n_calls
    valid_after_retry_rate = valid_after_retry / n_calls  # guaranteed >= first_try_rate
    fallback_rate = n_fallbacks / n_calls

    # 2. Groundedness, Profile Support, Contradiction Leakage per profile
    prof_groundedness: List[float] = []
    prof_groundedness_strict: List[float] = []
    prof_profile_support: List[float] = []
    prof_leakage: List[float] = []

    total_claimed_skills = 0
    total_grounded_skills = 0
    total_strict_grounded_skills = 0
    total_supported_skills = 0
    total_missing_skills = 0
    total_contradiction_skills = 0

    for pid, p_results in by_profile.items():
        prof = profiles_dict.get(pid, {})
        prof_text = f"{prof.get('target_role', '')} {' '.join(prof.get('skills', []))} {prof.get('resume_text', '')}"

        p_g_list = []
        p_gs_list = []
        p_supp_list = []
        p_leak_list = []

        for r in p_results:
            if r.fallback or not r.explanation:
                continue
            job = jobs_dict.get(r.job_id, {})
            job_text = f"{job.get('title', '')} {job.get('description', '')}"

            matched = r.explanation.matched_skills
            missing = r.explanation.missing_skills

            # Groundedness
            g_ratio, gs_ratio, details = evaluate_claim_groundedness(matched, job_text)
            p_g_list.append(g_ratio)
            p_gs_list.append(gs_ratio)
            total_claimed_skills += len(matched)
            total_grounded_skills += sum(1 for d in details if d["grounded"])
            total_strict_grounded_skills += sum(1 for d in details if d["grounded_strict"])

            # Profile support
            supp_ratio = evaluate_profile_support(matched, prof_text)
            p_supp_list.append(supp_ratio)
            total_supported_skills += sum(
                1 for s in matched if is_skill_grounded_regex(s, prof_text)
            )

            # Contradiction leakage (self-contradiction only)
            leak_ratio = evaluate_missing_skills_leakage(missing, job_text, prof_text)
            p_leak_list.append(leak_ratio)
            total_missing_skills += len(missing)
            for m in missing:
                if is_skill_grounded_regex(m, job_text) and is_skill_grounded_regex(m, prof_text):
                    total_contradiction_skills += 1

        prof_groundedness.append(float(np.mean(p_g_list)) if p_g_list else 1.0)
        prof_groundedness_strict.append(float(np.mean(p_gs_list)) if p_gs_list else 1.0)
        prof_profile_support.append(float(np.mean(p_supp_list)) if p_supp_list else 1.0)
        prof_leakage.append(float(np.mean(p_leak_list)) if p_leak_list else 0.0)

    # 3. Macro vs Pooled Metrics
    # Groundedness:
    groundedness_main_pooled = (
        total_grounded_skills / total_claimed_skills if total_claimed_skills > 0 else 1.0
    )
    groundedness_main_macro = float(np.mean(prof_groundedness)) if prof_groundedness else 1.0
    groundedness_strict_pooled = (
        total_strict_grounded_skills / total_claimed_skills if total_claimed_skills > 0 else 1.0
    )
    groundedness_strict_macro = (
        float(np.mean(prof_groundedness_strict)) if prof_groundedness_strict else 1.0
    )

    # Profile Support:
    profile_support_pooled = (
        total_supported_skills / total_claimed_skills if total_claimed_skills > 0 else 1.0
    )
    profile_support_macro = (
        float(np.mean(prof_profile_support)) if prof_profile_support else 1.0
    )

    # Contradiction Leakage:
    leakage_pooled = (
        total_contradiction_skills / total_missing_skills if total_missing_skills > 0 else 0.0
    )
    leakage_macro = float(np.mean(prof_leakage)) if prof_leakage else 0.0

    ci_grounded = compute_bootstrap_profile_ci(prof_groundedness)
    ci_grounded_strict = compute_bootstrap_profile_ci(prof_groundedness_strict)
    ci_support = compute_bootstrap_profile_ci(prof_profile_support)
    ci_leakage = compute_bootstrap_profile_ci(prof_leakage)

    # Validity CIs (resampling per-profile validity rates)
    prof_first_try = [
        sum(1 for r in p_res if not r.fallback and r.retry_count == 0) / len(p_res)
        for p_res in by_profile.values()
    ]
    prof_fallbacks = [
        sum(1 for r in p_res if r.fallback) / len(p_res) for p_res in by_profile.values()
    ]
    ci_first_try = compute_bootstrap_profile_ci(prof_first_try)
    ci_fallback = compute_bootstrap_profile_ci(prof_fallbacks)

    # 4. Latency & Tokens (Live calls only)
    live_results = [r for r in results if not r.cache_hit]
    latencies = [r.latency_ms for r in live_results if r.latency_ms > 0]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    prompt_toks = sum(r.prompt_tokens or 0 for r in live_results)
    cand_toks = sum(r.candidate_tokens or 0 for r in live_results)
    tot_toks = sum(r.total_tokens or 0 for r in live_results)

    cache_hits = sum(1 for r in results if r.cache_hit)
    cache_hit_rate = cache_hits / n_calls

    model_used = results[0].model if results else DEFAULT_MODEL
    prompt_ver = results[0].prompt_version if results else PROMPT_VERSION

    return {
        "caption": MANDATORY_CAPTION,
        "n_calls": n_calls,
        "n_profiles": n_profiles,
        "cache_hits": cache_hits,
        "live_calls": len(live_results),
        "cache_hit_rate": round(cache_hit_rate, 4),
        "gemini_model": model_used,
        "prompt_version": prompt_ver,
        "validity": {
            "json_valid_first_try_rate": round(first_try_rate, 4),
            "json_valid_first_try_ci_95": ci_first_try if ci_first_try is not None else "n too small",
            "retries_needed_count": retries_needed,
            "retries_needed_rate": round(retries_needed / n_calls, 4),
            "json_valid_after_retry_rate": round(valid_after_retry_rate, 4),
            "fallback_rate": round(fallback_rate, 4),
            "fallback_ci_95": ci_fallback if ci_fallback is not None else "n too small",
            "fallback_breakdown": fallback_breakdown,
        },
        "groundedness": {
            "groundedness_main_pooled": round(groundedness_main_pooled, 4),
            "groundedness_main_macro": round(groundedness_main_macro, 4),
            "groundedness_main_macro_ci_95": ci_grounded if ci_grounded is not None else "n too small",
            "groundedness_strict_pooled": round(groundedness_strict_pooled, 4),
            "groundedness_strict_macro": round(groundedness_strict_macro, 4),
            "groundedness_strict_macro_ci_95": ci_grounded_strict if ci_grounded_strict is not None else "n too small",
            "total_claimed_matched_skills": total_claimed_skills,
            "total_grounded_skills": total_grounded_skills,
            "total_strict_grounded_skills": total_strict_grounded_skills,
            # Backwards compatibility keys
            "groundedness_main": round(groundedness_main_pooled, 4),
            "groundedness_strict": round(groundedness_strict_pooled, 4),
        },
        "quality_signals": {
            "profile_support_pooled": round(profile_support_pooled, 4),
            "profile_support_macro": round(profile_support_macro, 4),
            "profile_support_macro_ci_95": ci_support if ci_support is not None else "n too small",
            "missing_skills_contradiction_pooled": round(leakage_pooled, 4),
            "missing_skills_contradiction_macro": round(leakage_macro, 4),
            "missing_skills_contradiction_macro_ci_95": ci_leakage if ci_leakage is not None else "n too small",
            "total_missing_skills_claimed": total_missing_skills,
            "total_contradiction_skills": total_contradiction_skills,
            "contradiction_definition": "Self-contradiction only: skills claimed as missing that appear in both the job text and the candidate profile; does not measure correctness of missing_skills.",
            # Backwards compatibility keys
            "profile_support": round(profile_support_pooled, 4),
            "missing_skills_leakage": round(leakage_pooled, 4),
        },
        "latency_live_ms": {
            "mean_ms": round(mean_latency, 2),
            "p95_ms": round(p95_latency, 2),
            "measured_live_calls": len(latencies),
        },
        "token_usage_live": {
            "prompt_tokens": prompt_toks,
            "candidate_tokens": cand_toks,
            "total_tokens": tot_toks,
        },
    }
