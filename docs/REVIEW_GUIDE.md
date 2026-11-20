# Human Review Guide & Candidate Profiles Reference

This guide provides everything needed to independently review and annotate [`labels_to_review.csv`](file:///d:/TEST%20PROJECTS/job%20search%20project/labels_to_review.csv) without needing to inspect other files or code.

---

## 1. Quick Review Instructions

1. Open **[`labels_to_review.csv`](file:///d:/TEST%20PROJECTS/job%20search%20project/labels_to_review.csv)** (277 total rows).
2. For each row, inspect the candidate profile (referenced by `profile_id`), the posting `title`, `company`, and `description snippet`.
3. In the **`my_label`** column, enter an integer:
   - **`0`**: **Irrelevant** (mismatched field/domain, distant onsite requirement with no remote flexibility, or severe seniority clash).
   - **`1`**: **Partially Relevant** (adjacent technical domain, transferable foundation skills, acceptable stretch seniority, or career-changer match).
   - **`2`**: **Strongly Relevant** (direct target role match, high core skill synergy, appropriate seniority, and compatible location).
4. When finished, save your reviewed file as **`labels_reviewed.csv`** in the repository root.

---

## 2. Decision Rubric Summary

| Label | Designation | Core Decision Rule |
|---|---|---|
| **0** | **Irrelevant** | The job is **not suitable**. Non-technical / unrelated profession (e.g., *Receptionist*, *Nurse*, *Driver*), incompatible software discipline without transferability (e.g., *Embedded C/C++* for *React Frontend*), extreme seniority clash (*VP/Executive* for *Intern/Entry level*), or distant mandatory onsite requirement. |
| **1** | **Partially Relevant** | A **plausible adjacent or stretch opportunity**. Matches core technical foundations but lacks specialized tools (e.g., *Data Analyst* matching *BI Developer*; *Backend* matching *Full Stack*); acceptable stretch seniority; or valid entry-level career-changer match. |
| **2** | **Strongly Relevant** | An **excellent direct fit**. Direct target role alignment, high skill overlap ($\ge 2$ core skills or $\ge 35\%$ stack match), compatible seniority tier, and satisfied location preference. |

---

## 3. Candidate Profiles Reference (`prof_01` to `prof_30`)

### `prof_01`: Machine Learning Engineer (Mid-Senior level | Remote)
- **Target Role**: Machine Learning Engineer
- **Core Skills**: Python, PyTorch, NLP, Transformers, FastAPI, Docker, ChromaDB
- **Background**: Machine Learning Engineer with 4 years of industry experience deploying neural search, vector databases, and transformer models to production microservices using FastAPI and Docker.

### `prof_02`: Senior Backend Engineer (Mid-Senior level | New York, NY)
- **Target Role**: Senior Backend Engineer
- **Core Skills**: Go, Python, Kubernetes, PostgreSQL, Kafka, Microservices, gRPC
- **Background**: Senior Backend Engineer with 6 years designing low-latency distributed systems, event-driven architectures with Apache Kafka, and scalable Go microservices running on Kubernetes.

### `prof_03`: Frontend Developer (Entry level | Boston, MA)
- **Target Role**: Frontend Developer
- **Core Skills**: JavaScript, TypeScript, React, HTML5, CSS3, Tailwind CSS, Git
- **Background**: Recent computer science graduate with strong hands-on experience building responsive web applications with React, TypeScript, and modern CSS frameworks. Completed two intensive web development internships.

### `prof_04`: Data Analyst (Entry level | Chicago, IL)
- **Target Role**: Data Analyst
- **Core Skills**: SQL, Python, Tableau, Excel, PowerBI, Pandas, Statistics
- **Background**: Junior Data Analyst with 1.5 years experience crafting executive Tableau dashboards, writing complex multi-table SQL queries, and performing exploratory data analysis with Python Pandas.

### `prof_05`: DevOps Engineer (Mid-Senior level | Dallas, TX)
- **Target Role**: DevOps Engineer
- **Core Skills**: AWS, Terraform, Docker, Kubernetes, CI/CD, Linux, Python, GitHub Actions
- **Background**: Cloud infrastructure and DevOps engineer with 5+ years automating multi-region AWS environments, authoring infrastructure-as-code with Terraform, and orchestrating container workloads on EKS.

### `prof_06`: Full Stack Developer (Entry level | Austin, TX)
- **Target Role**: Full Stack Developer
- **Core Skills**: Python, Django, React, JavaScript, PostgreSQL, REST APIs, Git
- **Background**: Full stack software engineer with 1 year professional experience delivering end-to-end web solutions with Django REST Framework and React frontends. Passionate about clean code and testing.

### `prof_07`: Mobile Developer (Mid-Senior level | San Francisco, CA)
- **Target Role**: Mobile Developer
- **Core Skills**: Swift, iOS, Kotlin, Android, React Native, Mobile UI, REST APIs
- **Background**: Mobile software engineer with 5 years shipping consumer-facing iOS and Android applications. Deep knowledge of Swift, Kotlin, offline sync caching, and native performance optimization.

### `prof_08`: Security Analyst (Mid-Senior level | Atlanta, GA)
- **Target Role**: Security Analyst
- **Core Skills**: Cybersecurity, SIEM, SOC, Vulnerability Management, Network Security, Incident Response, Python
- **Background**: Information security specialist with 4 years conducting threat detection, incident triage, vulnerability scans, and SIEM monitoring across enterprise cloud environments.

### `prof_09`: Database Administrator (Mid-Senior level | Houston, TX)
- **Target Role**: Database Administrator
- **Core Skills**: PostgreSQL, MySQL, Database Tuning, SQL, High Availability, Replication, Linux
- **Background**: Database Administrator with 5+ years maintaining mission-critical relational databases, query optimization, failover replication, backup automation, and zero-downtime schema migrations.

### `prof_10`: Product Manager (Mid-Senior level | Seattle, WA)
- **Target Role**: Product Manager
- **Core Skills**: Product Management, Agile, User Stories, Roadmapping, Data Analysis, API, Jira
- **Background**: Technical Product Manager with 4 years steering B2B developer platforms. Proven track record defining sprint roadmaps, translating customer discovery into actionable specs, and collaborating closely with engineering.

### `prof_11`: QA Automation Engineer (Mid-Senior level | Denver, CO)
- **Target Role**: QA Automation Engineer
- **Core Skills**: Selenium, Cypress, Python, Test Automation, Pytest, CI/CD, API Testing
- **Background**: Quality Assurance Engineer with 4 years creating end-to-end automated test suites using Selenium, Cypress, and Python. Specialized in continuous integration testing pipelines and regression suites.

### `prof_12`: Research Scientist (Mid-Senior level | Remote)
- **Target Role**: Research Scientist
- **Core Skills**: Python, Deep Learning, PyTorch, Research, Algorithms, Mathematics, Computer Vision
- **Background**: Applied research scientist holding an MS in Computer Science. 3+ years experimenting with deep neural network architectures, computer vision benchmarks, generative models, and algorithmic efficiency.

### `prof_13`: Systems Administrator (Entry level | Phoenix, AZ)
- **Target Role**: Systems Administrator
- **Core Skills**: Linux, Windows Server, Active Directory, Networking, Bash, Help Desk, Troubleshooting
- **Background**: IT Support and Systems Administrator with 2 years maintaining office local area networks, Windows Active Directory user provisioning, and Linux server patch management.

### `prof_14`: Site Reliability Engineer (Mid-Senior level | Los Angeles, CA)
- **Target Role**: Site Reliability Engineer
- **Core Skills**: Linux, Kubernetes, Prometheus, Grafana, Incident Response, Terraform, Go
- **Background**: Site Reliability Engineer with 5 years managing cloud uptime, SLI/SLO tracking, automated alerting with Prometheus and Grafana, and incident root cause mitigation.

### `prof_15`: Data Engineer (Mid-Senior level | United States)
- **Target Role**: Data Engineer
- **Core Skills**: Python, SQL, Spark, Airflow, Snowflake, ETL, AWS
- **Background**: Senior Data Pipeline Engineer with 5 years developing scalable batch and streaming ETL pipelines using Apache Spark, Airflow DAGs, and Snowflake data warehouses.

### `prof_16`: Software Engineer (Entry level | Remote)
- **Target Role**: Software Engineer
- **Core Skills**: Python, FastAPI, SQLite, Git, Object-Oriented Programming, Unit Testing
- **Background**: Motivated junior software engineer proficient in Python scripting, building clean REST microservices with FastAPI, and writing robust unit tests with pytest.

### `prof_17`: Embedded Systems Engineer (Mid-Senior level | San Diego, CA)
- **Target Role**: Embedded Systems Engineer
- **Core Skills**: C, C++, RTOS, Microcontrollers, Firmware, I2C, SPI, Hardware Debugging
- **Background**: Firmware engineer with 4 years programming resource-constrained microcontrollers in C/C++, writing low-level device drivers for I2C and SPI buses, and debugging hardware using oscilloscopes.

### `prof_18`: Customer Success Manager (Mid-Senior level | United States)
- **Target Role**: Customer Success Manager
- **Core Skills**: Client Management, Technical Support, SaaS, Onboarding, CRM, SQL, Communication
- **Background**: Technical Customer Success Manager with 4 years guiding enterprise SaaS clients through product onboarding, technical integration troubleshooting, and quarterly account growth reviews.

### `prof_19`: NLP Engineer (Mid-Senior level | Remote)
- **Target Role**: NLP Engineer
- **Core Skills**: Python, NLP, Spacy, Transformers, BERT, Information Extraction, Machine Learning
- **Background**: Machine learning practitioner specializing in natural language processing. 4 years fine-tuning language models, building information extraction pipelines, and text classification systems.

### `prof_20`: BI Developer (Mid-Senior level | Charlotte, NC)
- **Target Role**: BI Developer
- **Core Skills**: PowerBI, DAX, SQL, Data Modeling, ETL, SSIS, Reporting
- **Background**: Business Intelligence Developer with 5 years architecting enterprise PowerBI reporting models, DAX calculations, relational star schemas, and automated financial reports.

### `prof_21`: Scrum Master (Mid-Senior level | United States)
- **Target Role**: Scrum Master
- **Core Skills**: Agile, Scrum, Jira, Sprint Planning, Facilitation, Kanban, Cross-Functional
- **Background**: Certified Scrum Master with 4+ years facilitating daily standups, sprint retrospectives, and cross-team alignment across fast-paced software development squads.

### `prof_22`: Data Scientist (Entry level | New York, NY)
- **Target Role**: Data Scientist
- **Core Skills**: Python, R, Scikit-Learn, Machine Learning, Linear Regression, SQL, Data Visualization
- **Background**: Junior Data Scientist holding a BS in Statistics. Experienced with predictive modeling, hypothesis testing, feature engineering with Scikit-Learn, and visualizing trends in seaborn.

### `prof_23`: Computer Vision Engineer (Mid-Senior level | Remote)
- **Target Role**: Computer Vision Engineer
- **Core Skills**: Python, OpenCV, PyTorch, YOLO, Image Processing, Object Detection, CUDA
- **Background**: Computer Vision Engineer with 3.5 years training real-time object detection models (YOLO), video stream feature extraction using OpenCV, and edge device deployment.

### `prof_24`: Solutions Architect (Mid-Senior level | Chicago, IL)
- **Target Role**: Solutions Architect
- **Core Skills**: Cloud Architecture, Pre-Sales, Technical Presentations, AWS, APIs, Customer Requirements
- **Background**: Pre-sales Solutions Architect with 5 years partnering with enterprise sales executives to design cloud solutions, conduct technical demonstrations, and address security architecture inquiries.

### `prof_25`: Director of Engineering (Director | Remote)
- **Target Role**: Director of Engineering
- **Core Skills**: Engineering Leadership, Strategic Planning, Hiring, Mentorship, Budgeting, System Architecture
- **Background**: Engineering leader with 10+ years experience building and managing software engineering departments of 30+ engineers. Expert in organizational scaling, technical strategy, and engineering culture.

### `prof_26`: Software Engineer Intern (Internship | United States)
- **Target Role**: Software Engineer Intern
- **Core Skills**: Python, Java, Data Structures, Algorithms, Git, Object-Oriented Programming
- **Background**: Third-year undergraduate computer science student seeking a summer software engineering internship. Solid foundation in algorithms, data structures, Java, and Python coursework projects.

### `prof_27`: Junior Data Analyst (Entry level | Remote) *(Career Changer: Math Teacher)*
- **Target Role**: Junior Data Analyst
- **Core Skills**: SQL, Excel, Data Analysis, Statistics, Problem Solving, Mathematics
- **Background**: Former high school mathematics teacher transitioning to data analytics. Completed an intensive 6-month data bootcamp mastering SQL queries, Excel pivot modeling, and descriptive statistical analysis. Brings exceptional communication, rigorous quantitative reasoning, and proven diligence.

### `prof_28`: Junior Software Developer (Entry level | United States) *(Career Changer: Mechanical Engineer)*
- **Target Role**: Junior Software Developer
- **Core Skills**: Python, C++, Git, Algorithms, Mathematics, Linux
- **Background**: Mechanical engineer with 4 years designing CAD automation tools transitioning into full-time software engineering. Proficient in Python scripting, numerical simulation, and basic C++. Strong analytical problem solver with professional experience in engineering cross-functional teams.

### `prof_29`: IT Specialist (Entry level | Remote) *(Vague Technical Profile)*
- **Target Role**: IT Specialist
- **Core Skills**: Computer Troubleshooting, Hardware, Operating Systems, Customer Service, Technical Support
- **Background**: Tech enthusiast with a passion for computers and technology. Good at solving hardware problems, setting up office operating systems, and assisting team members with software issues. Quick learner eager to work in tech.

### `prof_30`: Technical Specialist (Mid-Senior level | United States) *(Vague IT Generalist)*
- **Target Role**: Technical Specialist
- **Core Skills**: IT Management, Infrastructure, Technical Coordination, Vendor Management, Cloud Overview
- **Background**: Experienced IT professional with over 8 years in corporate computing environments. Familiar with general network infrastructure, managing office tech vendors, and interested in transitioning to modern cloud-first companies.
