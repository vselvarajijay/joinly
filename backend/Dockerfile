# Tool server for joinly GPT-Live mode. Build from the repository root:
#   docker build -f backend/Dockerfile -t joinly-backend .
FROM python:3.12-slim

ENV UV_LINK_MODE="copy" \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app

# workspace metadata is needed to resolve the shared lockfile
COPY uv.lock pyproject.toml ./
COPY client/pyproject.toml client/pyproject.toml
COPY common/pyproject.toml common/pyproject.toml
COPY backend backend

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package joinly-backend

RUN groupadd --gid 1001 app && useradd --uid 1001 --gid 1001 -m app
USER app

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"]

# 0.0.0.0 inside the container only; docker-compose publishes no ports
ENTRYPOINT ["/app/.venv/bin/joinly-backend", "--host", "0.0.0.0", "--port", "8080"]
