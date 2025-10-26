# syntax=docker/dockerfile:1

# Multi-stage Dockerfile for XRay Analysis using uv-only
# Stage 1: builder/test (installs dev deps, runs tests)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project files
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY tests /app/tests

# Install dev dependencies defined in pyproject
RUN uv sync --system --group dev

# Lint & test (fail the build if these fail)
RUN uv run --system ruff format --check && \
    uv run --system ruff check && \
    uv run --system pytest -q

# Stage 2: runtime image (slim) with only runtime deps
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy only the source and install runtime dependencies
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src

# Install the package with runtime dependencies from pyproject
RUN uv pip install --system /workspace

# Default command runs the analysis on a sample file
ENTRYPOINT ["uv", "run", "--system", "xray"]
CMD ["--input", "data/dummy.xlsx", "--output", "artifacts"]
