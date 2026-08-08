# Self-hosted FLUX.2 Klein 9B inference server (BentoML) for the DGX Spark.
# Build on the server itself (arm64) so the image matches the host arch:
#   docker build -t flux2-klein:latest .
# The model weights are mounted at runtime via FLUX2_MODEL_PATH; nothing is
# baked into this image.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Ubuntu 24.04 ships Python 3.12; the PyTorch cu130 aarch64 wheels target it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY service.py ./

# Model weights live on the host ("external module") and are mounted read-only:
#   docker run -v /path/to/models/FLUX.2-klein-9B:/models/flux2-klein:ro ...
ENV FLUX2_MODEL_PATH=/models/flux2-klein
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:3000/healthz >/dev/null || exit 1

CMD ["bentoml", "serve", "service.py:Flux2Klein", "--host", "0.0.0.0", "--port", "3000"]
