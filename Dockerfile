FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY credit_intelligence ./credit_intelligence

ENV PATH="/app/.venv/bin:${PATH}" \
    JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64" \
    CREDIT_DATA_DIR="/app/data"

EXPOSE 8000 8501
