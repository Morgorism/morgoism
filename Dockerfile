# Verified official PyTorch CUDA runtime tag (see README_DEPLOY.md).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
    JOB_DIR=/workspace/jobs \
    MODEL_CONFIG=/workspace/rerank_config.json

WORKDIR /app

# PDF rendering. MinerU parser is intentionally NOT installed by default:
# the supplied conversion script requires the `mineru-kit` CLI from the
# original team environment. MinerU ZIP and MD work without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc texlive-xetex texlive-latex-recommended texlive-latex-extra \
    fonts-noto-cjk fontconfig git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY components/horizon-reranker/requirements.txt /tmp/requirements-reranker.txt
COPY deployment/requirements-api.txt /tmp/requirements-api.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-reranker.txt \
    -r /tmp/requirements-api.txt

COPY . /app
RUN mkdir -p /workspace/jobs /workspace/hf_cache

EXPOSE 8000
# Keep exactly one API process per GPU; multiple workers would load multiple 7B models.
CMD ["python", "-m", "uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
