# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements-local-torch.txt ./docker/requirements-local-torch.txt
COPY docker/requirements-local.txt ./docker/requirements-local.txt
# --timeout/--retries: ride out a slow or flaky connection instead of
# aborting on pip's short default read timeout (real machines, including a
# client's, won't always have a fast/stable line during a large download).
#
# --mount=type=cache on pip's own download cache: persists across build
# attempts independent of whether a layer succeeds or fails (unlike the
# regular image layer cache), so a retry after a network drop only
# re-fetches what wasn't already downloaded, instead of starting over.
# This cache lives outside the final image, so it doesn't bloat it.
#
# torch installed in its own layer first (by far the largest download, and
# the one most likely to hit a timeout) so a failed `docker compose build`
# retry resumes from Docker's cache instead of re-downloading everything
# that already succeeded from scratch.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 600 --retries 15 \
    -r docker/requirements-local-torch.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 600 --retries 15 \
    -r docker/requirements-local.txt
# torch's CPU wheel index can pull in an old torchao; newer peft versions
# hard-error on it during LoRA loading even though this project never uses
# torchao at all (see README). `|| true`: this uninstall is genuinely
# best-effort (fine if torchao was never installed) -- unlike the pip
# installs above, whose failure must NOT be swallowed.
RUN pip uninstall -y -q torchao || true

# Everything the chat app needs at runtime to run the REAL fine-tuned
# model in-container: the src package, the seed dialogues (few-shot
# prompt examples), and the trained LoRA adapter itself. Excludes
# data/processed, data/splits, notebooks, etc. -- see .dockerignore.
COPY src ./src
COPY data/authored ./data/authored
COPY checkpoints/naija-switch-lora ./checkpoints/naija-switch-lora

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs/conversations \
    && chown -R appuser:appuser /app
# Not switching to USER appuser here: docker-compose.yml mounts a named
# volume at HF_HOME (below) for model-cache persistence, and Docker always
# creates a fresh named volume owned by root -- appuser couldn't write to
# it despite the chown above (that ran before the volume existed). The
# entrypoint script fixes the volume's ownership at container start (as
# root, a one-time cheap op) and then drops to appuser for the actual app,
# so the app itself still never runs as root.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Defaults for a client handoff: run the real fine-tuned adapter locally,
# no HF_TOKEN required. docker-compose.yml can still override these.
ENV PYTHONUNBUFFERED=1 \
    INFERENCE_BACKEND=local \
    LOCAL_BASE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
    LOCAL_LORA_ADAPTER_DIR=checkpoints/naija-switch-lora \
    HF_HOME=/app/.cache/huggingface
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
