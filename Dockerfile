FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for numpy/faiss/scipy style builds; drop if wheels suffice
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source after deps to leverage Docker layer caching
COPY . .

EXPOSE 8000
CMD ["uvicorn", "agent.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
