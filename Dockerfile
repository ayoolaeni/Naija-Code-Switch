FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements-app.txt ./docker/requirements-app.txt
RUN pip install --no-cache-dir -r docker/requirements-app.txt

# Only what the chat app needs at runtime: the src package and the seed
# dialogues it reads for few-shot prompt examples. Excludes data/processed,
# data/splits, checkpoints, notebooks, etc. -- see .dockerignore.
COPY src ./src
COPY data/authored ./data/authored

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs/conversations \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "src/app/ui.py", \
    "--server.port=8501", "--server.address=0.0.0.0", \
    "--server.headless=true", "--browser.gatherUsageStats=false"]
