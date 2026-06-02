# Reproducible CPU environment for slippage-predictor.
# Build:  docker build -t slippage-predictor .
# Test:   docker run --rm slippage-predictor pytest tests/ -q
# Demo:   docker run --rm slippage-predictor make demo
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first (better layer caching) using only the metadata.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Now copy the rest of the project (configs, tests, scripts, docs).
COPY . .

# Default to running the test suite; override the command for training/eval.
CMD ["pytest", "tests/", "-q"]
