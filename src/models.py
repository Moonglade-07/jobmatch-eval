"""Domain data models and schemas for candidate profiles and search results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple


@dataclass
class Profile:
    """Candidate profile representation for search and ranking.

    Attributes:
        resume_text: Raw or summarized narrative text of candidate experience.
        skills: List of technical or domain skills claimed by the candidate.
        preferred_location: Target geographic location or remote preference.
        seniority: Desired seniority tier (e.g. 'Entry level', 'Mid-Senior level').
        target_role: Target role headline (e.g. 'Machine Learning Engineer').
    """

    resume_text: str
    skills: List[str] = field(default_factory=list)
    preferred_location: str = ""
    seniority: str = ""
    target_role: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile instance to dictionary.

        Returns:
            Dictionary representation of candidate profile.
        """
        return {
            "resume_text": self.resume_text,
            "skills": list(self.skills),
            "preferred_location": self.preferred_location,
            "seniority": self.seniority,
            "target_role": self.target_role,
        }


@dataclass
class SearchResult:
    """Individual ranked job posting result from a retrieval or ranking strategy.

    Supports tuple unpacking `job_id, score = result` for compatibility with
    standard `(job_id, score)` ranking interfaces.

    Attributes:
        job_id: Unique identifier of the job posting.
        score: Final ranking score assigned by the retrieval or ranking pipeline.
        title: Job headline or role title.
        company: Employing company or organization.
        location: Normalized job location.
        experience_level: Standardized seniority level of the position.
        description: Cleaned job description text.
        matched_skills: List of candidate skills explicitly found in posting text.
        base_score: Retrieval score before personalization adjustments (min-max normalized to [0, 1]).
        raw_score: Original raw score emitted by the retrieval strategy.
        skill_boost: Additive score boost earned from matching skills.
        filter_tier: Metadata filter fallback tier (1=strict match, 2=relaxed seniority, 3=global backfill).
    """

    job_id: str
    score: float
    title: str = ""
    company: str = ""
    location: str = ""
    experience_level: str = ""
    description: str = ""
    matched_skills: List[str] = field(default_factory=list)
    base_score: float = 0.0
    raw_score: float = 0.0
    skill_boost: float = 0.0
    filter_tier: int = 1

    def __iter__(self) -> Iterator[Any]:
        """Allow tuple unpacking: job_id, score = result.

        Returns:
            Iterator yielding job_id and score.
        """
        return iter((self.job_id, self.score))

    def __getitem__(self, index: int) -> Any:
        """Allow index access: result[0] is job_id, result[1] is score.

        Args:
            index: Integer index (0 or 1).

        Returns:
            job_id for 0, score for 1.
        """
        return (self.job_id, self.score)[index]

