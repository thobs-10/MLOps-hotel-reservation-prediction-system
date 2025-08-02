# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first for better caching
COPY pyproject.toml .

# Install dependencies using pip directly from pyproject.toml
RUN pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Create model directories
RUN mkdir -p /app/src/models/model_registry
RUN mkdir -p /app/src/models/trained_models
RUN mkdir -p /app/src/models/artifacts

# Expose port
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
