FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Create data/logs directories
RUN mkdir -p /app/data /app/logs

# Copy source code
COPY . .

# Non-root user
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

# Health check endpoint
EXPOSE 8080

# Default: run the bot
CMD ["python", "-m", "polybotking.main"]
