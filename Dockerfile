# syntax=docker/dockerfile:1.7
#
# M5 Forecaster — production-ish image for the FastAPI serving package.
#
# Three stages:
#   1. builder  — uv builds a wheel from pyproject.toml + src/
#   2. deps     — uv resolves locked deps + installs the wheel into /opt/venv
#   3. runtime  — distroless-ish slim Python with the venv copied in, non-root user
#
# Build:    docker build -t m5-forecaster:local .
# Run:      docker run --rm -p 8000:8000 \
#               -v "$(pwd)/artifacts/models/lgbm/latest:/srv/model:ro" \
#               m5-forecaster:local
#
# Or use docker-compose.yaml (mounts artifact + sets env).

ARG PYTHON_VERSION=3.12

# ─── Stage 1: build a wheel from the source tree ─────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# uv's official image ships a static binary; copy it in instead of `pip install uv`
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy

WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /dist


# ─── Stage 2: resolve locked deps + install the wheel ────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# libgomp1 is required by lightgbm at import time (not just runtime).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build
COPY pyproject.toml uv.lock ./

# Install pinned deps from the lockfile (main + serve, skip dev/notebook). Skipping
# `--install-project` keeps the venv free of an editable path so the layer is cacheable.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group serve

# Then drop in our wheel without dependency resolution (deps already locked above).
COPY --from=builder /dist /dist
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps /dist/*.whl


# ─── Stage 3: minimal runtime image ──────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Minimal runtime libs: libgomp1 for lightgbm, curl for HEALTHCHECK, certs for TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — the container runs as uid 1001 (matches docker-compose).
RUN groupadd --system --gid 1001 app \
    && useradd  --system --uid 1001 --gid app --home /app --shell /usr/sbin/nologin app

WORKDIR /app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    M5_SERVE_HOST=0.0.0.0 \
    M5_SERVE_PORT=8000 \
    M5_SERVE_LOG_JSON=true \
    M5_SERVE_MODEL_DIR=/srv/model

COPY --from=deps /opt/venv /opt/venv

# Mount point for the model artifact (read-only in compose / k8s).
RUN mkdir -p /srv/model && chown -R app:app /srv/model /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["python", "-m", "m5.serve"]
