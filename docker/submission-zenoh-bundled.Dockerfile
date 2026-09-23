FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG EXECUTION_MODE=async

LABEL org.opencontainers.image.source-kit="origami-inference-kit-async"
LABEL org.opencontainers.image.description="Comp-action-chunk origami policy for origami-zenoh-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    linux-libc-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

COPY requirements.submission.txt /tmp/requirements.submission.txt

RUN python -m pip install --upgrade pip setuptools wheel \
  && python -m pip install --extra-index-url "${TORCH_INDEX_URL}" -r /tmp/requirements.submission.txt

COPY . /app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

RUN useradd --system --create-home --uid 1000 policy \
  && chmod 0755 /app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/docker/submission_zenoh_entrypoint.sh \
  && chown -R policy:policy /app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

ENV PYTHONPATH=/app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2 \
    PATH=/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:${PATH} \
    ORIGAMI_MODEL_BUNDLE=/app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/model_bundle \
    EXECUTION_MODE=${EXECUTION_MODE} \
    HOME=/tmp/origami-home \
    XDG_CACHE_HOME=/tmp/origami-cache \
    HF_HOME=/tmp/origami-hf \
    HUGGINGFACE_HUB_CACHE=/tmp/origami-hf/hub \
    HF_DATASETS_CACHE=/tmp/origami-hf/datasets \
    HF_LEROBOT_HOME=/tmp/origami-hf/lerobot \
    TORCH_HOME=/tmp/origami-torch \
    JAX_COMPILATION_CACHE_DIR=/tmp/origami-jax-cache \
    TMPDIR=/tmp/origami-tmp \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform

USER policy
HEALTHCHECK NONE
ENTRYPOINT ["/app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/docker/submission_zenoh_entrypoint.sh"]
CMD ["serve"]
