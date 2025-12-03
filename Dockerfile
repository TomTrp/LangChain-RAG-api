# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

# Set workdir
WORKDIR /app

# Install system dependencies about build ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++6 \
    libffi-dev \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY src/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final minimal image
FROM python:3.11-slim

WORKDIR /app

# Install necessary runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libffi-dev \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy builder only installed library
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ .

# Create cache HuggingFace and pip
RUN rm -rf /root/.cache/huggingface /root/.cache/pip

# Open port 8000
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
