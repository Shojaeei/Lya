#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Lya - Single Command VPS Deployment (System-wide Python)
# Usage: ./deploy.sh
#
# Installs everything system-wide (no venv, no Docker).
# For a fresh Ubuntu VPS:
#   git clone <repo> && cd Lya && chmod +x deploy.sh && ./deploy.sh
# ═══════════════════════════════════════════════════════════════
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[LYA]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Lya - Autonomous AI Telegram Bot Deployment${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# ─── Check .env ───────────────────────────────────────────────
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        warn ".env not found — copied from .env.example"
        warn "Edit .env with your bot token and API keys before running!"
        echo ""
        echo "Required settings in .env:"
        echo "  LYA_TELEGRAM_BOT_TOKEN=<your-bot-token>"
        echo "  LYA_TELEGRAM_ALLOWED_USERS=<your-telegram-user-id>"
        echo "  LYA_LLM_OLLAMA_API_KEY=<your-api-key>"
        echo "  LYA_LLM_BASE_URL=http://localhost:11434"
        echo ""
        err "Please edit .env and run deploy.sh again."
    else
        err ".env file not found and no .env.example to copy from."
    fi
fi

# ─── Install system packages ─────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    curl \
    > /dev/null 2>&1
log "System packages installed (python3, pip, ffmpeg, git)"

# ─── Install Python packages system-wide ─────────────────────
log "Installing Python packages system-wide..."

# Override PEP 668 externally-managed-environment protection
export PIP_BREAK_SYSTEM_PACKAGES=1

pip3 install \
    "pydantic>=2.5.0" \
    "pydantic-settings>=2.1.0" \
    "python-dotenv>=1.0.0" \
    "structlog>=23.0.0" \
    "httpx>=0.25.0" \
    "python-telegram-bot>=20.0" \
    "chromadb>=0.4.0" \
    "aiohttp>=3.9.0" \
    "beautifulsoup4>=4.12.0" \
    "yt-dlp>=2024.0.0"
log "Python packages installed system-wide"

# ─── Create workspace ────────────────────────────────────────
mkdir -p ~/.lya
log "Workspace ready at ~/.lya"

# ─── Find python3 path ───────────────────────────────────────
PYTHON_BIN=$(which python3)
log "Using Python: $PYTHON_BIN"

# ─── Create systemd service ──────────────────────────────────
SERVICE_FILE="/etc/systemd/system/lya.service"
log "Creating systemd service..."

sudo tee "$SERVICE_FILE" > /dev/null << SERVICEEOF
[Unit]
Description=Lya Autonomous AI Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/src
Environment=PYTHONUTF8=1
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_BIN $PROJECT_DIR/run_lya.py
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=lya

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$PROJECT_DIR $(eval echo ~$(whoami))/.lya /tmp

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable lya
sudo systemctl restart lya

echo ""
log "Lya is running as a systemd service!"
echo ""
echo "  View logs:    sudo journalctl -u lya -f"
echo "  Status:       sudo systemctl status lya"
echo "  Stop:         sudo systemctl stop lya"
echo "  Restart:      sudo systemctl restart lya"
echo ""
echo "  The bot will auto-start on system reboot."
echo ""
