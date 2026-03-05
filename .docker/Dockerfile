# ═══════════════════════════════════════════════════════════════
# Lya - Autonomous AI Telegram Bot
# Single-stage build for VPS deployment
# ═══════════════════════════════════════════════════════════════
FROM python:3.12-slim

LABEL maintainer="Shoji"
LABEL description="Lya - Autonomous AGI Telegram Bot"

WORKDIR /app

# Install system dependencies: ffmpeg (for yt-dlp audio/video), git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no poetry in production — use pip)
COPY pyproject.toml ./

# Install core + runtime extras via pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pydantic">=2.5.0" \
    pydantic-settings">=2.1.0" \
    python-dotenv">=1.0.0" \
    structlog">=23.0.0" \
    httpx">=0.25.0" \
    python-telegram-bot">=20.0" \
    chromadb">=0.4.0" \
    aiohttp">=3.9.0" \
    beautifulsoup4">=4.12.0" \
    yt-dlp">=2024.0.0"

# Copy application source
COPY src/ ./src/
COPY run_lya.py ./

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/app/src \
    LYA_ENV=production \
    LYA_WORKSPACE_PATH=/app/workspace

# Create workspace directory
RUN mkdir -p /app/workspace

# No port needed — Telegram bot uses long polling, not HTTP
# Health check via process existence
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -f "run_lya.py" || exit 1

# Run the Telegram bot
CMD ["python", "run_lya.py"]
