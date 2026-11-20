# Multi-stage / optimized slim Dockerfile for local Streamlit Job Search & Agent Demo
# Dataset: 3,000-posting clean derivative sample licensed under CC BY-SA 4.0
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Install minimal build tools for compiling C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, clean evaluation dataset sample, and reports
# Raw uncleaned dataset (raw_postings.csv) is deliberately excluded (.dockerignore)
COPY src/ ./src/
COPY tests/ ./tests/
COPY reports/ ./reports/
COPY data/jobs_clean.parquet ./data/jobs_clean.parquet
COPY data/eval/ ./data/eval/
COPY pyproject.toml .
COPY .env.example .
COPY app.py .

# Pre-build ChromaDB dense vector index at image build time
# Guarantees self-contained, reproducible startup without volume mounts
RUN python src/index.py

# Expose standard Streamlit port
EXPOSE 8501

# Healthcheck for local container execution
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app on port 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
