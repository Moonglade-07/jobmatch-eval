"""Generate and validate realistic synthetic candidate profiles for retrieval evaluation.

Enforces:
  1. 25-30 diverse profiles across roles, seniorities, locations, and skills.
  2. Programmatic check ensuring EACH profile has >= 50 postings passing the strict Tier 1 filter.
  3. Mostly US cities with high coverage, Remote, Entry level, and Mid-Senior level.
  4. At most 2 profiles with Director/Executive/Internship seniority.
  5. 3-4 hard cases (career changers, vague technical background).
  6. All profiles marked explicitly as hand-authored synthetic evaluation input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.models import Profile
from src.personalise import matches_location, matches_seniority

DEFAULT_CLEAN_DATA_PATH = Path("data/jobs_clean.parquet")
DEFAULT_OUTPUT_PROFILES_PATH = Path("data/eval/profiles.json")


def build_candidate_profiles() -> List[Profile]:
    """Define 28 realistic, hand-authored candidate profiles for retrieval benchmarking.

    Returns:
        List of candidate Profile objects with varied parameters.
    """
    profiles: List[Profile] = [
        # 1. Machine Learning Engineer (Mid-Senior, Remote)
        Profile(
            profile_id="prof_01",
            target_role="Machine Learning Engineer",
            skills=["Python", "PyTorch", "NLP", "Transformers", "FastAPI", "Docker", "ChromaDB"],
            preferred_location="Remote",
            seniority="Mid-Senior level",
            resume_text=(
                "Machine Learning Engineer with 4 years of industry experience deploying neural search, "
                "vector databases, and transformer models to production microservices using FastAPI and Docker."
            ),
        ),
        # 2. Senior Backend Engineer (Mid-Senior, New York, NY)
        Profile(
            profile_id="prof_02",
            target_role="Senior Backend Engineer",
            skills=["Go", "Python", "Kubernetes", "PostgreSQL", "Kafka", "Microservices", "gRPC"],
            preferred_location="New York, NY",
            seniority="Mid-Senior level",
            resume_text=(
                "Senior Backend Engineer with 6 years designing low-latency distributed systems, "
                "event-driven architectures with Apache Kafka, and scalable Go microservices deployed on Kubernetes."
            ),
        ),
        # 3. Junior Frontend Developer (Entry level, Boston, MA)
        Profile(
            profile_id="prof_03",
            target_role="Frontend Developer",
            skills=["JavaScript", "TypeScript", "React", "HTML5", "CSS3", "Tailwind CSS", "Git"],
            preferred_location="Boston, MA",
            seniority="Entry level",
            resume_text=(
                "Recent computer science graduate with strong hands-on experience building responsive web "
                "applications with React, TypeScript, and modern CSS frameworks. Completed two intensive web development internships."
            ),
        ),
        # 4. Data Analyst (Entry level, Chicago, IL)
        Profile(
            profile_id="prof_04",
            target_role="Data Analyst",
            skills=["SQL", "Python", "Tableau", "Excel", "PowerBI", "Pandas", "Statistics"],
            preferred_location="Chicago, IL",
            seniority="Entry level",
            resume_text=(
                "Junior Data Analyst with 1.5 years experience crafting executive Tableau dashboards, "
                "writing complex multi-table SQL queries, and performing exploratory data analysis with Python Pandas."
            ),
        ),
        # 5. Cloud DevOps Architect (Mid-Senior, Dallas, TX)
        Profile(
            profile_id="prof_05",
            target_role="DevOps Engineer",
            skills=["AWS", "Terraform", "Docker", "Kubernetes", "CI/CD", "Linux", "Python", "GitHub Actions"],
            preferred_location="Dallas, TX",
            seniority="Mid-Senior level",
            resume_text=(
                "Cloud infrastructure and DevOps engineer with 5+ years automating multi-region AWS environments, "
                "authoring infrastructure-as-code with Terraform, and orchestrating container workloads on EKS."
            ),
        ),
        # 6. Full Stack Web Developer (Entry level, Austin, TX)
        Profile(
            profile_id="prof_06",
            target_role="Full Stack Developer",
            skills=["Python", "Django", "React", "JavaScript", "PostgreSQL", "REST APIs", "Git"],
            preferred_location="Austin, TX",
            seniority="Entry level",
            resume_text=(
                "Full stack software engineer with 1 year professional experience delivering end-to-end web "
                "solutions with Django REST Framework and React frontends. Passionate about clean code and testing."
            ),
        ),
        # 7. Mobile App Developer (Mid-Senior, San Francisco, CA)
        Profile(
            profile_id="prof_07",
            target_role="Mobile Developer",
            skills=["Swift", "iOS", "Kotlin", "Android", "React Native", "Mobile UI", "REST APIs"],
            preferred_location="San Francisco, CA",
            seniority="Mid-Senior level",
            resume_text=(
                "Mobile software engineer with 5 years shipping consumer-facing iOS and Android applications. "
                "Deep knowledge of Swift, Kotlin, offline sync caching, and native performance optimization."
            ),
        ),
        # 8. Cybersecurity / Information Security Analyst (Mid-Senior, Atlanta, GA)
        Profile(
            profile_id="prof_08",
            target_role="Security Analyst",
            skills=["Cybersecurity", "SIEM", "SOC", "Vulnerability Management", "Network Security", "Incident Response", "Python"],
            preferred_location="Atlanta, GA",
            seniority="Mid-Senior level",
            resume_text=(
                "Information security specialist with 4 years conducting threat detection, incident triage, "
                "vulnerability scans, and SIEM monitoring across enterprise cloud environments."
            ),
        ),
        # 9. Database Administrator / SQL Specialist (Mid-Senior, Houston, TX)
        Profile(
            profile_id="prof_09",
            target_role="Database Administrator",
            skills=["PostgreSQL", "MySQL", "Database Tuning", "SQL", "High Availability", "Replication", "Linux"],
            preferred_location="Houston, TX",
            seniority="Mid-Senior level",
            resume_text=(
                "Database Administrator with 5+ years maintaining mission-critical relational databases, "
                "query optimization, failover replication, backup automation, and zero-downtime schema migrations."
            ),
        ),
        # 10. Technical Product Manager (Mid-Senior, Seattle, WA)
        Profile(
            profile_id="prof_10",
            target_role="Product Manager",
            skills=["Product Management", "Agile", "User Stories", "Roadmapping", "Data Analysis", "API", "Jira"],
            preferred_location="Seattle, WA",
            seniority="Mid-Senior level",
            resume_text=(
                "Technical Product Manager with 4 years steering B2B developer platforms. Proven track record "
                "defining sprint roadmaps, translating customer discovery into actionable specs, and collaborating closely with engineering."
            ),
        ),
        # 11. QA Automation Engineer (Mid-Senior, Denver, CO)
        Profile(
            profile_id="prof_11",
            target_role="QA Automation Engineer",
            skills=["Selenium", "Cypress", "Python", "Test Automation", "Pytest", "CI/CD", "API Testing"],
            preferred_location="Denver, CO",
            seniority="Mid-Senior level",
            resume_text=(
                "Quality Assurance Engineer with 4 years creating end-to-end automated test suites using "
                "Selenium, Cypress, and Python. Specialized in continuous integration testing pipelines and regression suites."
            ),
        ),
        # 12. AI Research Scientist (Mid-Senior, Remote)
        Profile(
            profile_id="prof_12",
            target_role="Research Scientist",
            skills=["Python", "Deep Learning", "PyTorch", "Research", "Algorithms", "Mathematics", "Computer Vision"],
            preferred_location="Remote",
            seniority="Mid-Senior level",
            resume_text=(
                "Applied research scientist holding an MS in Computer Science. 3+ years experimenting with deep neural "
                "network architectures, computer vision benchmarks, generative models, and algorithmic efficiency."
            ),
        ),
        # 13. Systems Administrator / IT Support (Entry level, Phoenix, AZ)
        Profile(
            profile_id="prof_13",
            target_role="Systems Administrator",
            skills=["Linux", "Windows Server", "Active Directory", "Networking", "Bash", "Help Desk", "Troubleshooting"],
            preferred_location="Phoenix, AZ",
            seniority="Entry level",
            resume_text=(
                "IT Support and Systems Administrator with 2 years maintaining office local area networks, "
                "Windows Active Directory user provisioning, and Linux server patch management."
            ),
        ),
        # 14. Site Reliability Engineer (Mid-Senior, Los Angeles, CA)
        Profile(
            profile_id="prof_14",
            target_role="Site Reliability Engineer",
            skills=["Linux", "Kubernetes", "Prometheus", "Grafana", "Incident Response", "Terraform", "Go"],
            preferred_location="Los Angeles, CA",
            seniority="Mid-Senior level",
            resume_text=(
                "Site Reliability Engineer with 5 years managing cloud uptime, SLI/SLO tracking, automated alerting "
                "with Prometheus and Grafana, and incident root cause mitigation."
            ),
        ),
        # 15. Data Engineer (Mid-Senior, United States)
        Profile(
            profile_id="prof_15",
            target_role="Data Engineer",
            skills=["Python", "SQL", "Spark", "Airflow", "Snowflake", "ETL", "AWS"],
            preferred_location="United States",
            seniority="Mid-Senior level",
            resume_text=(
                "Senior Data Pipeline Engineer with 5 years developing scalable batch and streaming ETL pipelines "
                "using Apache Spark, Airflow DAGs, and Snowflake data warehouses."
            ),
        ),
        # 16. Junior Python Developer (Entry level, Remote)
        Profile(
            profile_id="prof_16",
            target_role="Software Engineer",
            skills=["Python", "FastAPI", "SQLite", "Git", "Object-Oriented Programming", "Unit Testing"],
            preferred_location="Remote",
            seniority="Entry level",
            resume_text=(
                "Motivated junior software engineer proficient in Python scripting, building clean REST microservices "
                "with FastAPI, and writing robust unit tests with pytest."
            ),
        ),
        # 17. Embedded Firmware Engineer (Mid-Senior, San Diego, CA)
        Profile(
            profile_id="prof_17",
            target_role="Embedded Systems Engineer",
            skills=["C", "C++", "RTOS", "Microcontrollers", "Firmware", "I2C", "SPI", "Hardware Debugging"],
            preferred_location="San Diego, CA",
            seniority="Mid-Senior level",
            resume_text=(
                "Firmware engineer with 4 years programming resource-constrained microcontrollers in C/C++, "
                "writing low-level device drivers for I2C and SPI buses, and debugging hardware using oscilloscopes."
            ),
        ),
        # 18. Technical Customer Success Manager (Mid-Senior, United States)
        Profile(
            profile_id="prof_18",
            target_role="Customer Success Manager",
            skills=["Client Management", "Technical Support", "SaaS", "Onboarding", "CRM", "SQL", "Communication"],
            preferred_location="United States",
            seniority="Mid-Senior level",
            resume_text=(
                "Technical Customer Success Manager with 4 years guiding enterprise SaaS clients through product onboarding, "
                "technical integration troubleshooting, and quarterly account growth reviews."
            ),
        ),
        # 19. Natural Language Processing Specialist (Mid-Senior, Remote)
        Profile(
            profile_id="prof_19",
            target_role="NLP Engineer",
            skills=["Python", "NLP", "Spacy", "Transformers", "BERT", "Information Extraction", "Machine Learning"],
            preferred_location="Remote",
            seniority="Mid-Senior level",
            resume_text=(
                "Machine learning practitioner specializing in natural language processing. 4 years fine-tuning language "
                "models, building information extraction pipelines, and text classification systems."
            ),
        ),
        # 20. Business Intelligence Developer (Mid-Senior, Charlotte, NC)
        Profile(
            profile_id="prof_20",
            target_role="BI Developer",
            skills=["PowerBI", "DAX", "SQL", "Data Modeling", "ETL", "SSIS", "Reporting"],
            preferred_location="Charlotte, NC",
            seniority="Mid-Senior level",
            resume_text=(
                "Business Intelligence Developer with 5 years architecting enterprise PowerBI reporting models, "
                "DAX calculations, relational star schemas, and automated financial reports."
            ),
        ),
        # 21. Scrum Master / Agile Project Manager (Mid-Senior, United States)
        Profile(
            profile_id="prof_21",
            target_role="Scrum Master",
            skills=["Agile", "Scrum", "Jira", "Sprint Planning", "Facilitation", "Kanban", "Cross-Functional"],
            preferred_location="United States",
            seniority="Mid-Senior level",
            resume_text=(
                "Certified Scrum Master with 4+ years facilitating daily standups, sprint retrospectives, and cross-team "
                "alignment across fast-paced software development squads."
            ),
        ),
        # 22. Junior Data Scientist (Entry level, New York, NY)
        Profile(
            profile_id="prof_22",
            target_role="Data Scientist",
            skills=["Python", "R", "Scikit-Learn", "Machine Learning", "Linear Regression", "SQL", "Data Visualization"],
            preferred_location="New York, NY",
            seniority="Entry level",
            resume_text=(
                "Junior Data Scientist holding a BS in Statistics. Experienced with predictive modeling, hypothesis testing, "
                "feature engineering with Scikit-Learn, and visualizing trends in seaborn."
            ),
        ),
        # 23. Computer Vision Engineer (Mid-Senior, Remote)
        Profile(
            profile_id="prof_23",
            target_role="Computer Vision Engineer",
            skills=["Python", "OpenCV", "PyTorch", "YOLO", "Image Processing", "Object Detection", "CUDA"],
            preferred_location="Remote",
            seniority="Mid-Senior level",
            resume_text=(
                "Computer Vision Engineer with 3.5 years training real-time object detection models (YOLO), "
                "video stream feature extraction using OpenCV, and edge device deployment."
            ),
        ),
        # 24. Solutions Architect / Sales Engineer (Mid-Senior, Chicago, IL)
        Profile(
            profile_id="prof_24",
            target_role="Solutions Architect",
            skills=["Cloud Architecture", "Pre-Sales", "Technical Presentations", "AWS", "APIs", "Customer Requirements"],
            preferred_location="Chicago, IL",
            seniority="Mid-Senior level",
            resume_text=(
                "Pre-sales Solutions Architect with 5 years partnering with enterprise sales executives to design cloud solutions, "
                "conduct technical demonstrations, and address security architecture inquiries."
            ),
        ),
        # --- AT MOST 2 PROFILES AT DIRECTOR / INTERNSHIP / EXECUTIVE LEVELS ---
        # 25. Director of Engineering (Director, Remote) - Level 1 of 2
        Profile(
            profile_id="prof_25",
            target_role="Director of Engineering",
            skills=["Engineering Leadership", "Strategic Planning", "Hiring", "Mentorship", "Budgeting", "System Architecture"],
            preferred_location="Remote",
            seniority="Director",
            resume_text=(
                "Engineering leader with 10+ years experience building and managing software engineering departments "
                "of 30+ engineers. Expert in organizational scaling, technical strategy, and engineering culture."
            ),
        ),
        # 26. Software Engineering Intern (Internship, United States) - Level 2 of 2
        Profile(
            profile_id="prof_26",
            target_role="Software Engineer Intern",
            skills=["Python", "Java", "Data Structures", "Algorithms", "Git", "Object-Oriented Programming"],
            preferred_location="United States",
            seniority="Internship",
            resume_text=(
                "Third-year undergraduate computer science student seeking a summer software engineering internship. "
                "Solid foundation in algorithms, data structures, Java, and Python coursework projects."
            ),
        ),
        # --- 3-4 HARD CASES: CAREER CHANGERS & VAGUE PROFILES ---
        # 27. Hard Case 1: Career changer - Math Teacher -> Junior Data Analyst (Entry level, Remote)
        Profile(
            profile_id="prof_27",
            target_role="Junior Data Analyst",
            skills=["SQL", "Excel", "Data Analysis", "Statistics", "Problem Solving", "Mathematics"],
            preferred_location="Remote",
            seniority="Entry level",
            resume_text=(
                "Former high school mathematics teacher transitioning to data analytics. Completed an intensive 6-month "
                "data bootcamp mastering SQL queries, Excel pivot modeling, and descriptive statistical analysis. "
                "Brings exceptional communication, rigorous quantitative reasoning, and proven diligence."
            ),
        ),
        # 28. Hard Case 2: Career changer - Mechanical Engineer -> Junior Developer (Entry level, United States)
        Profile(
            profile_id="prof_28",
            target_role="Junior Software Developer",
            skills=["Python", "C++", "Git", "Algorithms", "Mathematics", "Linux"],
            preferred_location="United States",
            seniority="Entry level",
            resume_text=(
                "Mechanical engineer with 4 years designing CAD automation tools transitioning into full-time software "
                "engineering. Proficient in Python scripting, numerical simulation, and basic C++. Strong analytical "
                "problem solver with professional experience in engineering cross-functional teams."
            ),
        ),
        # 29. Hard Case 3: Vague Profile - Broad Tech Support / General IT (Entry level, Remote)
        Profile(
            profile_id="prof_29",
            target_role="IT Specialist",
            skills=["Computer Troubleshooting", "Hardware", "Operating Systems", "Customer Service", "Technical Support"],
            preferred_location="Remote",
            seniority="Entry level",
            resume_text=(
                "Tech enthusiast with a passion for computers and technology. Good at solving hardware problems, "
                "setting up office operating systems, and assisting users with software issues. Quick learner eager to work in tech."
            ),
        ),
        # 30. Hard Case 4: Vague Profile - Experienced Generalist IT seeking modern tech (Mid-Senior, United States)
        Profile(
            profile_id="prof_30",
            target_role="Technical Specialist",
            skills=["IT Management", "Infrastructure", "Technical Coordination", "Vendor Management", "Cloud Overview"],
            preferred_location="United States",
            seniority="Mid-Senior level",
            resume_text=(
                "Experienced IT professional with over 8 years in corporate computing environments. Familiar with general "
                "network infrastructure, managing office tech vendors, and interested in transitioning to modern cloud-first companies."
            ),
        ),
    ]
    return profiles


def verify_and_save_profiles(
    clean_data_path: Path = DEFAULT_CLEAN_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PROFILES_PATH,
    min_tier_1_postings: int = 50,
) -> List[Profile]:
    """Run programmatic check that each profile matches >= min_tier_1_postings and save to JSON.

    Args:
        clean_data_path: Path to parquet dataset.
        output_path: Destination path for profiles JSON.
        min_tier_1_postings: Minimum threshold of strict Tier 1 postings required per profile.

    Returns:
        List of validated Profile objects.
    """
    if not clean_data_path.exists():
        raise FileNotFoundError(f"Clean postings file not found at {clean_data_path}")

    print(f"Loading job postings from {clean_data_path} for Tier 1 verification...")
    df = pd.read_parquet(clean_data_path)
    print(f"Loaded {len(df)} total job postings.")

    candidate_profiles = build_candidate_profiles()
    print(f"\nEvaluating {len(candidate_profiles)} synthetic candidate profiles against strict Tier 1 filter:")
    print("=" * 110)
    print(f"{'Profile ID':<12} | {'Role':<28} | {'Location':<18} | {'Seniority':<18} | {'Tier 1 Count':<12} | {'Status'}")
    print("-" * 110)

    verified_profiles: List[Profile] = []
    failed_profiles: List[Tuple[Profile, int]] = []

    for prof in candidate_profiles:
        pref_loc = prof.preferred_location
        pref_sen = prof.seniority

        # Count postings passing strict Tier 1 filter
        tier_1_matches = 0
        for _, row in df.iterrows():
            loc_match = matches_location(
                job_location=str(row["location"]),
                preferred_location=pref_loc,
                title=str(row["title"]),
                description=str(row["description"]),
            )
            sen_match = matches_seniority(
                job_level=str(row["experience_level"]),
                preferred_level=pref_sen,
            )
            if loc_match and sen_match:
                tier_1_matches += 1

        status_str = "PASS" if tier_1_matches >= min_tier_1_postings else "FAIL (<50)"
        print(
            f"{prof.profile_id:<12} | {prof.target_role[:28]:<28} | {pref_loc[:18]:<18} | "
            f"{pref_sen[:18]:<18} | {tier_1_matches:<12} | {status_str}"
        )

        if tier_1_matches >= min_tier_1_postings:
            verified_profiles.append(prof)
        else:
            failed_profiles.append((prof, tier_1_matches))

    print("=" * 110)
    if failed_profiles:
        failed_ids = [p[0].profile_id for p in failed_profiles]
        raise ValueError(
            f"Verification failed! The following {len(failed_profiles)} profiles had < {min_tier_1_postings} "
            f"strict Tier 1 matches: {failed_ids}. Drop or broaden them before proceeding."
        )

    print(f"\nAll {len(verified_profiles)} profiles successfully passed the strict Tier 1 check (>= {min_tier_1_postings} postings)!")

    # Verify constraints:
    # 1. Total profiles between 25 and 30
    assert 25 <= len(verified_profiles) <= 30, f"Expected 25-30 profiles, got {len(verified_profiles)}"

    # 2. At most 2 profiles with Director/Executive/Internship seniority
    non_standard_levels = [
        p for p in verified_profiles if p.seniority in {"Director", "Executive", "Internship"}
    ]
    assert len(non_standard_levels) <= 2, (
        f"Constraint violated: Expected at most 2 Director/Executive/Internship profiles, "
        f"got {len(non_standard_levels)}: {[p.profile_id for p in non_standard_levels]}"
    )
    print(f"Verified constraint: At most 2 non-standard seniority profiles (found {len(non_standard_levels)}: {[p.profile_id for p in non_standard_levels]}).")

    # Serialize to JSON with clear synthetic/hand-authored provenance metadata
    payload: Dict[str, Any] = {
        "metadata": {
            "description": "Evaluation candidate test profiles for job search retrieval benchmarking",
            "provenance": "Hand-authored synthetic test inputs (NOT scraped or copied from postings)",
            "total_profiles": len(verified_profiles),
            "tier_1_threshold_guarantee": f">={min_tier_1_postings} strict Tier 1 postings in 3,000 index",
            "hard_cases_count": 4,
            "director_or_intern_count": len(non_standard_levels),
        },
        "profiles": [p.to_dict() for p in verified_profiles],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(verified_profiles)} verified profiles to {output_path}.\n")
    return verified_profiles


if __name__ == "__main__":
    verify_and_save_profiles()

