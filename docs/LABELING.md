# Job Relevance Evaluation Rubric (Graded 0 / 1 / 2)

This document establishes the official labeling guidelines and rubric used to annotate pooled candidate-job pairs for retrieval evaluation in `jobmatch-eval`.

---

## 1. Evaluation Scale Overview

Every pair of `(Candidate Profile, Job Posting)` is assigned an integer label from $\{0, 1, 2\}$ reflecting technical fit, role suitability, seniority alignment, and geographic compatibility:

| Label | Designation | Core Concept |
|---|---|---|
| **0** | **Irrelevant** | The job is not suitable. Major mismatch in domain, target role, technical skills, location, or seniority. |
| **1** | **Partially Relevant** | Plausible fit or adjacent role. Candidate satisfies core competencies but lacks niche technologies or is an adjacent domain match. |
| **2** | **Strongly Relevant** | Excellent match. Direct target role alignment, high core skill synergy, appropriate seniority, and geographic compatibility. |

---

## 2. Detailed Dimension Breakdown

### Label 2: Strongly Relevant (Direct Fit)
A posting receives a **2** if **all** primary dimensions align:
1. **Role Alignment**: The job title and principal responsibilities directly match the candidate's target role headline (e.g., *Machine Learning Engineer* for an ML profile, *Senior Backend Engineer* for a Backend profile).
2. **Skill Overlap**: Strong intersection between candidate's core stack and job requirements (e.g., PyTorch + Transformers + FastAPI matching an NLP/LLM job; or Go + Kafka + Kubernetes matching a distributed backend role).
3. **Seniority Compatibility**: Matches the candidate's experience level (e.g., *Entry level* for new graduates / bootcamp grads; *Mid-Senior level* for candidates with 3-7 years experience).
4. **Location**: Compatible with preferences (offers Remote if candidate seeks Remote, or situated in candidate's metropolitan target).

*Examples*:
- Profile: *Senior Backend Engineer (Go, Kubernetes, Kafka, NYC)* $\to$ Job: *Senior Platform Engineer (Go, Kafka, Microservices, NYC)* $\implies$ **2**
- Profile: *Junior Frontend Developer (React, TypeScript, CSS, Boston)* $\to$ Job: *Junior Web Developer (React, JavaScript, Boston)* $\implies$ **2**
- Profile: *Machine Learning Engineer (PyTorch, Transformers, Remote)* $\to$ Job: *AI / ML Engineer (Python, PyTorch, LLM Retrieval, Remote)* $\implies$ **2**

---

### Label 1: Partially Relevant (Adjacent / Stretch Fit)
A posting receives a **1** if it is a plausible, related opportunity with minor gaps:
1. **Adjacent Role / Domain**: The role is in a neighboring specialization where skills are transferable (e.g., *Data Analyst* applied to a *Business Intelligence* or *Junior Data Scientist* posting; *DevOps Engineer* applied to a *Site Reliability Engineer* posting; *Full Stack* applied to *Frontend*).
2. **Moderate Skill Coverage**: The candidate matches primary foundation skills (e.g., Python, SQL, REST APIs) but lacks specific proprietary or secondary technologies mentioned in the job description.
3. **Acceptable Seniority Transition**: Near the target tier (e.g., mid-level posting for an entry-level candidate with strong portfolio, or senior posting for mid-level).
4. **Career Changers / Vague Profiles**: For hard-case profiles (e.g., career changer with quantitative/math background), entry-level technical roles with foundational overlaps qualify as partially relevant.

*Examples*:
- Profile: *Data Analyst (SQL, Tableau, Excel, Chicago)* $\to$ Job: *Junior Business Intelligence Developer (PowerBI, SQL, Chicago)* $\implies$ **1**
- Profile: *DevOps Engineer (AWS, Terraform, Docker, Dallas)* $\to$ Job: *Cloud Systems Administrator (AWS, Linux, Python, Dallas)* $\implies$ **1**
- Profile: *Career Changer (Math Teacher $\to$ Data Analyst)* $\to$ Job: *Operations Reporting Analyst (Excel, SQL, Remote)* $\implies$ **1**

---

### Label 0: Irrelevant (Mismatch)
A posting receives a **0** if there is a fundamental mismatch on one or more critical dimensions:
1. **Domain / Function Disconnect**: Job belongs to an entirely different profession (e.g., *Receptionist*, *Sales Representative*, *Accountant*, *Nurse*, *Mechanical Technician* for a Software Engineer candidate).
2. **Specialization Clashing**: Completely incompatible software disciplines without transferable overlap (e.g., *Embedded Firmware C/C++ microcontrollers* for a *React/CSS Frontend Developer*).
3. **Seniority Extreme**: Extreme gap in seniority (e.g., *Executive VP / Chief Technology Officer* for an *Entry level / Intern* candidate, or *Summer Intern* for a 10-year engineering leader).
4. **Location Barrier**: The role strictly mandates onsite presence in a completely distant geography incompatible with candidate's location and offers no remote flexibility.

*Examples*:
- Profile: *Machine Learning Engineer (Remote)* $\to$ Job: *Legal Assistant / Receptionist (Bridgeport, CT)* $\implies$ **0**
- Profile: *Frontend React Developer (Boston)* $\to$ Job: *Senior Hardware RF Testing Engineer (San Diego, CA)* $\implies$ **0**
- Profile: *Software Engineering Intern* $\to$ Job: *Vice President of Enterprise Sales (Houston, TX)* $\implies$ **0**

---

## 3. Labeling Procedure & Audit Sampling

1. **Candidate Pool Generation**: Top 20 results from each of four distinct retrieval strategies (BM25, Dense Vector, Hybrid RRF, Cross-Encoder Re-rank) are unioned per profile to prevent retrieval bias.
2. **Deterministic Model Proposals**: Initial graded proposals (0, 1, or 2) are computed deterministically adhering strictly to the above rubric.
3. **Quality Control (25% Must-Review Sample)**: A stratified random 25% subset of pooled pairs is marked as `must_review=True` (using a fixed random seed) for human audit.
4. **Disagreement Analysis**: Discrepancies between proposed and reviewer labels are measured via Cohen's weighted kappa and confusion matrices in Phase 2B.

