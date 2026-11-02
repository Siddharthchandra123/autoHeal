# GPU-capable runtime requires a platform-specific base image in production.
# This base is for portable controller testing / KFP mock jobs only.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY autotune/ ./autotune/
COPY config/ ./config/
COPY sample_data/ ./sample_data/
CMD ["python", "-m", "autotune.orchestrator", "--mode", "mock"]
