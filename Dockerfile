# LLM-EvalLab Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create directories for data and runs
RUN mkdir -p /app/data/datasets /app/data/corpus /app/runs /app/configs

# Expose ports for API and dashboard
EXPOSE 8080 8501

# Default command
CMD ["evalab", "serve"]
