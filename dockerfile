FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code and required files
COPY production_app ./production_app
COPY ui ./ui
COPY models ./models
COPY data ./data

EXPOSE 8000 8501

# Default command (Compose overrides this for the UI container)
CMD ["uv", "run", "uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]