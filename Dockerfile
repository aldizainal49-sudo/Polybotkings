FROM python:3.11-slim

WORKDIR /app

# Make Python output unbuffered (important for docker logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Copy build files needed by pip (pyproject.toml references README.md)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install -e .

# Download textblob corpora (used by sentiment engine)
RUN python -m textblob.download_corpora || true

# Create data/logs directories
RUN mkdir -p /app/data /app/logs

# Copy any remaining files (e.g. .env.example, configs, entrypoint)
COPY . .

# Install entrypoint script that auto-fixes volume permissions at runtime
RUN cp /app/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Non-root user (UID 1000 - matches typical host user for clean volume mounts)
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app

# NOTE: We intentionally start as root and drop to botuser inside the
# entrypoint, so the entrypoint can chown the volume mounts first.

# Health check endpoint (served by embedded dashboard inside main.py)
EXPOSE 8080

# Default: run the bot (which also starts the embedded health server)
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "polybotking.main"]
