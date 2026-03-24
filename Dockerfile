# ── Stage 1: Build Next.js frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --prefer-offline
COPY frontend/ ./
RUN npm run build          # outputs to frontend/out/ (output: 'export')

# ── Stage 2: Python API ───────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface

# System deps for numpy/faiss/scipy-style builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leverage Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Copy the pre-built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/out ./frontend/out

EXPOSE 7860

# CMD ["sh", "-c", "uvicorn agent.server:app --host 0.0.0.0 --port ${PORT} --workers 2"]
CMD ["uvicorn", "agent.server:app", "--host", "0.0.0.0", "--port", "7860"]
