# Lya 🤖 - Professional Autonomous AGI Agent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/architecture-clean%20architecture-success)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-grade autonomous AGI agent built with **Clean Architecture** principles.

```
┌─────────────────────────────────────────────────────────────┐
│                    LYA ORCHESTRATOR                        │
│  One entry point • Multiple interfaces • Health checks     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (Single Command)

```bash
# Clone or navigate to project
cd lya

# Run the unified installer (creates venv, installs deps, configures)
python install.py

# Or skip installer and run directly
python run_lya.py
```

That's it! The installer will:
1. ✓ Create virtual environment
2. ✓ Install all dependencies
3. ✓ Create `.env` configuration
4. ✓ Check LLM connectivity
5. ✓ Start interactive setup

## 📁 Project Structure

```
lya/
├── 📄 run_lya.py                ← 🎯 MAIN ENTRY POINT (unified runner)
├── 📄 install.py            ← 🔧 Setup wizard (venv, deps, config)
├── 📄 pyproject.toml        ← Poetry dependencies
├── 📄 Makefile             ← Development commands
│
├── 📁 src/lya/              ← Clean Architecture source
│   ├── 📁 domain/          ← Business logic (entities, services)
│   ├── 📁 application/     ← Use cases (commands, queries)
│   ├── 📁 infrastructure/  ← Adapters (LLM, DB, config)
│   └── 📁 adapters/        ← Interfaces (CLI, API, Telegram)
│
├── 📁 .docker/            ← Docker setup
├── 📁 .kubernetes/        ← K8s manifests
├── 📁 docs/               ← Documentation
└── 📁 tests/              ← Test suite
```

## 🎮 Usage

### Interactive Mode (Recommended)

```bash
python run_lya.py
```

This launches the interactive orchestrator that:
1. Runs health checks (Python, deps, LLM, workspace)
2. Lets you configure interfaces (Web, Telegram)
3. Starts services
4. Opens interactive chat

### Installation Wizard

```bash
python install.py              # Full interactive setup
python install.py --quick      # Quick setup with defaults
python install.py --check      # Check installation only
```

### Development Commands

```bash
# After running install.py, activate venv:
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# Development
make dev              # Run with hot reload
make test             # Run tests
make lint             # Run linter
make format           # Format code
make type-check       # Run mypy
```

## 🏗️ Architecture

### Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  ADAPTERS        CLI │ Web API │ Telegram │ Voice           │  ← Interface Adapters
├─────────────────────────────────────────────────────────────┤
│  APPLICATION     Commands │ Queries │ Events                │  ← Use Cases
├─────────────────────────────────────────────────────────────┤
│  DOMAIN          Agent │ Goal │ Task │ Memory │ Events       │  ← Business Logic
│                  (No external dependencies)                 │
├─────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE  LLM (Ollama/OpenAI) │ Vector DB │ Tools    │  ← External Services
└─────────────────────────────────────────────────────────────┘
        ↑ Dependencies point INWARD
```

### Key Features

| Feature | Implementation |
|---------|---------------|
| **Domain Events** | 15+ events for loose coupling |
| **Repository Pattern** | Swappable storage (Chroma/Qdrant/File) |
| **Dependency Injection** | Protocol-based ports/adapters |
| **Configuration** | Pydantic settings with validation |
| **Observability** | Structured logging (structlog) |
| **Security** | Sandboxed execution |

## 🔌 Interfaces

### CLI (Always Available)
```bash
lya run --autonomous    # Run autonomously
lya chat               # Interactive chat
lya api                # Start API server
lya status             # Check status
```

### Web API (Port 8000)
```
GET  /health           # Health check
GET  /api/v1/agents    # List agents
POST /api/v1/goals     # Create goal
WS   /ws               # WebSocket for real-time
```

### Telegram Bot
```
Configure LYA_TELEGRAM_BOT_TOKEN in .env
Bot commands:
  /start - Start conversation
  /goals - List goals
  /status - System status
```

## ⚙️ Configuration

Edit `.env` file:

```bash
# LLM Provider
LYA_LLM_PROVIDER=ollama              # or openai, anthropic
LYA_LLM_MODEL=kimi-k2.5:cloud
LYA_LLM_BASE_URL=http://localhost:11434

# For OpenAI
LYA_OPENAI_API_KEY=sk-...

# Vector Database
LYA_MEMORY_VECTOR_DB=chroma            # or qdrant

# Telegram (optional)
LYA_TELEGRAM_BOT_TOKEN=...
LYA_TELEGRAM_ALLOWED_USERS=username1,username2
```

## 🧪 Testing

```bash
make test-unit        # Unit tests (fast)
make test-integration # Integration tests
make test-e2e         # End-to-end tests
make test-coverage    # Coverage report
```

## 🐳 Docker

```bash
# Development with all services
docker-compose up

# Access:
# - Lya API: http://localhost:8000
# - Chroma DB: http://localhost:8001
# - Prometheus: http://localhost:9091
# - Grafana: http://localhost:3000
```

## 📚 Documentation

- [Architecture Overview](docs/architecture/STRUCTURE.md)
- [Implementation Summary](docs/architecture/IMPLEMENTATION_SUMMARY.md)
- [Migration Guide](docs/architecture/MIGRATION_GUIDE.md)

## 🛠️ Development

### Prerequisites
- Python 3.10+
- Poetry (for dependency management)
- Ollama (for local LLM) or OpenAI API key

### Setup Steps

```bash
# 1. Install Python 3.10+ and Poetry

# 2. Run installer
python install.py

# 3. Activate environment
source .venv/bin/activate

# 4. Run
python run_lya.py
```

## 📦 Dependencies

| Category | Packages |
|----------|----------|
| **Core** | pydantic, structlog, python-dotenv |
| **LLM** | httpx, openai, anthropic |
| **Memory** | chromadb, qdrant-client, sentence-transformers |
| **API** | fastapi, uvicorn, websockets |
| **CLI** | rich, click, prompt-toolkit |
| **Testing** | pytest, pytest-asyncio, pytest-cov |

See `pyproject.toml` for full list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run `make ci` to check lint/tests
4. Submit a pull request

## 📝 License

MIT - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built with:
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [Pydantic](https://pydantic.dev) for configuration management
- [FastAPI](https://fastapi.tiangolo.com) for web interface
- [Chroma](https://www.trychroma.com) for vector storage
- [Ollama](https://ollama.ai) for local LLM inference

---

**Status**: ✅ Professional structure ready for implementation
**Next**: Add LLM adapters and command handlers in infrastructure layer
