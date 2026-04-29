# CPU-only benchmark image. Identical binary runs on GCP c3-standard-8 and Hetzner CX53.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-bench.txt .
RUN pip install --no-cache-dir -r requirements-bench.txt

COPY src/ ./src/
COPY scripts/run_benchmark_matrix.py ./scripts/
COPY configs/ ./configs/

# Models pulled at runtime from HF Hub (not baked in — keeps image small and reusable).
# Test sets mounted at runtime via -v $(pwd)/test_sets:/app/test_sets.
# Results written to /app/runs (mount via -v $(pwd)/runs:/app/runs).

ENTRYPOINT ["python", "-m", "scripts.run_benchmark_matrix"]
CMD ["--help"]
