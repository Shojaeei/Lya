"""Lya - Entry point for the Telegram bot with all features.

Features initialized:
- LLM via Ollama (OllamaAdapter)
- Long-term memory (JSON-backed)
- Voice messages (STT/TTS)
- Image understanding (vision models)
- Streaming responses
- Conversation summarization
- Task scheduler (reminders)
- RAG / Document Q&A
- Interactive buttons
- Preference learning
- Web dashboard (optional)
"""

import asyncio
import signal
import sys
import os
from pathlib import Path

# Fix Windows encoding — force UTF-8 for stdout/stderr
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def check_ollama(base_url: str) -> tuple[bool, str]:
    """Check if Ollama is running and the model is available."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return True, f"{len(models)} models available"
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def check_telegram(bot_token: str) -> tuple[bool, str]:
    """Check if the Telegram bot token is valid."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")
            data = r.json()
            if data.get("ok"):
                name = data["result"].get("username", "unknown")
                return True, f"@{name}"
            return False, data.get("description", "Invalid token")
    except Exception as e:
        return False, str(e)


async def start_dashboard(bot) -> None:
    """Start the web dashboard (FastAPI) alongside the bot."""
    try:
        import uvicorn
        from lya.infrastructure.config.settings import settings

        # Import and configure FastAPI app
        from lya.adapters.api.app import app

        config = uvicorn.Config(
            app,
            host=settings.api.host,
            port=settings.api.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()
    except ImportError:
        pass  # uvicorn/fastapi not installed
    except Exception as e:
        print(f"  Dashboard: Failed to start ({e})")


async def main():
    """Start Lya Telegram bot."""
    print("=" * 50)
    print("  LYA - Autonomous AI Assistant")
    print("=" * 50)
    print()

    # Load settings
    from lya.infrastructure.config.settings import settings

    # Health checks
    print("Checking systems...")

    # Python
    v = sys.version_info
    print(f"  Python:   {v.major}.{v.minor}.{v.micro} ({'OK' if v >= (3, 10) else 'FAIL'})")

    # Ollama
    ollama_ok, ollama_msg = await check_ollama(settings.llm.base_url)
    print(f"  Ollama:   {ollama_msg} ({'OK' if ollama_ok else 'WARN'})")

    if not ollama_ok:
        print(f"  [WARN] Ollama not reachable. Bot starts but LLM won't work.")

    # LLM model
    print(f"  Model:    {settings.llm.model}")
    print(f"  API Key:  {'set' if settings.llm.ollama_api_key else 'not set'}")

    # Telegram
    if not settings.telegram.bot_token:
        print("\n[ERROR] Telegram bot token not configured.")
        print("  Set LYA_TELEGRAM_BOT_TOKEN in .env")
        return

    tg_ok, tg_msg = await check_telegram(settings.telegram.bot_token)
    print(f"  Telegram: {tg_msg} ({'OK' if tg_ok else 'FAIL'})")

    if not tg_ok:
        print("\n[ERROR] Telegram bot token is invalid.")
        return

    # Workspace
    workspace = settings.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"  Workspace: {workspace}")

    # Initialize embedding service (Feature 4)
    embedding_service = None
    try:
        from lya.infrastructure.persistence.embedding_service import EmbeddingService
        embedding_service = EmbeddingService.create("ollama")
        print(f"  Embeddings: Ollama ({settings.memory.embedding_model})")
    except Exception as e:
        print(f"  Embeddings: hash fallback ({e})")

    # Initialize memory system
    memory_service = None
    try:
        from lya.infrastructure.memory.memory_adapter import create_memory_adapter
        memory_dir = str(workspace / "memory")
        memory_service = await create_memory_adapter(
            persist_directory=memory_dir,
            embedding_service=embedding_service,
        )
        mem_count = await memory_service.get_memory_count()
        print(f"  Memory:   JSON store ({mem_count} memories)")
    except Exception as e:
        print(f"  Memory:   Failed ({e})")

    # Voice check
    voice_status = "disabled"
    if settings.voice.enabled:
        try:
            from lya.infrastructure.voice.stt_adapter import create_stt_adapter
            stt = create_stt_adapter(language=settings.voice.stt_language)
            voice_status = "ready" if stt.is_available() else "STT unavailable (need ffmpeg)"
        except Exception as e:
            voice_status = f"error ({e})"
    print(f"  Voice:    {voice_status}")

    # Scheduler
    print(f"  Scheduler: enabled")

    # RAG
    print(f"  RAG:      enabled")

    # Streaming
    print(f"  Streaming: {'enabled' if settings.telegram.streaming else 'disabled'}")

    # Dashboard
    dashboard_enabled = False
    try:
        import uvicorn
        from lya.adapters.api.app import app
        dashboard_enabled = True
        print(f"  Dashboard: http://{settings.api.host}:{settings.api.port}")
    except ImportError:
        print(f"  Dashboard: not available (install uvicorn + fastapi)")

    print()
    print("Starting bot...")
    print()

    # Start the bot
    from lya.adapters.telegram.telegram_bot import TelegramBot

    bot = TelegramBot(
        bot_token=settings.telegram.bot_token,
        workspace=workspace,
        memory_service=memory_service,
    )

    print(f"  Bot:   {tg_msg}")
    if bot.llm:
        print(f"  LLM:   {bot.llm.model} via Ollama")
    print(f"  Press Ctrl+C to stop")
    print()

    # Graceful shutdown
    def handle_signal(sig, frame):
        print("\nShutting down...")
        asyncio.create_task(bot.stop())

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    try:
        # Start dashboard in background if available
        if dashboard_enabled:
            asyncio.create_task(start_dashboard(bot))

        await bot.start()
    except asyncio.CancelledError:
        pass
    finally:
        await bot.stop()
        print("Lya stopped. Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
