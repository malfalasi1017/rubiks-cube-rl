FROM python:3.12.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONWRITEBYTECODE=1

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]