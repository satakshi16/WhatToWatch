# syntax=docker/dockerfile:1.7
#
# Multi-stage image (P2-01, NFR-21).
#   target `app` — FastAPI api + the job worker (same image, different command); ml group
#   target `ui`  — Streamlit UI; only the ui group (httpx + streamlit), no `app/` code
#
# No secrets are baked in: `.env` is excluded by .dockerignore and config is passed at
# runtime (`env_file:` in docker-compose.yml). `make image-scan` verifies this.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.18

FROM ${UV_IMAGE} AS uv

# ---------- dependency builders ----------
FROM ${PYTHON_IMAGE} AS deps-base
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

FROM deps-base AS deps-app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-default-groups --group ml

FROM deps-base AS deps-ui
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --only-group ui

# ---------- runtime ----------
FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

FROM runtime AS app
COPY --from=deps-app /opt/venv /opt/venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
# Mount points for the named volumes: data/ (SQLite fallback, embeddings) and the
# Hugging Face model cache. Created here so fresh volumes inherit `app` ownership.
ENV HF_HOME=/models
RUN mkdir -p /app/data /models && chown app:app /app/data /models
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM runtime AS ui
COPY --from=deps-ui /opt/venv /opt/venv
COPY --chown=app:app ui ./ui
USER app
EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
