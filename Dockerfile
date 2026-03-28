# Sancta — Docker image for Ubuntu VM
# Runs SIEM dashboard; agent is started from the dashboard when you click Start.

FROM python:3.11-slim

WORKDIR /app

# System deps for PyTorch / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch for smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: use CPU-only PyTorch to reduce image size (~1.5GB smaller)
# Uncomment if you hit OOM or want a lighter image:
# RUN pip uninstall -y torch && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

# SIEM binds to 0.0.0.0 so it's reachable from host
ENV SIEM_HOST=0.0.0.0
EXPOSE 8787

# Run SIEM dashboard; agent is spawned by dashboard when user clicks Start
CMD ["python", "-m", "uvicorn", "backend.siem_server:app", "--host", "0.0.0.0", "--port", "8787"]
