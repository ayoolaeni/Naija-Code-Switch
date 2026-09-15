#!/bin/sh
# Docker creates the named `hf-cache` volume (see docker-compose.yml) owned
# by root, but the app runs as the non-root `appuser` for security -- so
# without this, the container can't write into its own model cache and
# every request to download the base model fails with a PermissionError.
# This entrypoint runs once as root just to fix that, then drops to
# appuser for the actual app, same as if USER appuser had been set in the
# Dockerfile directly.
set -e
mkdir -p /app/.cache/huggingface
chown -R appuser:appuser /app/.cache/huggingface

exec su -s /bin/sh appuser -c "streamlit run src/app/ui.py \
    --server.port=8501 --server.address=0.0.0.0 \
    --server.headless=true --browser.gatherUsageStats=false"
