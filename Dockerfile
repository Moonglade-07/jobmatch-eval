FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and configuration
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml .
COPY .env.example .

# Set offline environment variables for cached models
ENV ANONYMIZED_TELEMETRY=False
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Default execution: run test suite
CMD ["pytest", "-v", "tests/"]

