"""Telegram Bot for Lya.

Features:
- LLM-powered responses via Ollama
- Persistent chat history (survives restarts)
- Short-term memory (WorkingMemoryBuffer per user session)
- Long-term memory (JSON via MemoryService)
- Memory-augmented LLM context
- Voice message support (STT/TTS)
- Image understanding (vision models)
- Streaming responses
- User access control
- Typing indicators
- Markdown formatting with fallback
- Message splitting for long responses
"""

from __future__ import annotations

import sys
import os

# Fix Windows Unicode: force UTF-8 for stdout/stderr so Persian/Arabic text doesn't crash
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Add ffmpeg to PATH if installed via WinGet but not in current PATH
    from pathlib import Path as _Path
    _ffmpeg_dirs = [
        _Path.home() / "AppData/Local/Microsoft/WinGet/Links",
        _Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
    ]
    for _d in _ffmpeg_dirs:
        if _d.exists():
            # Check if ffmpeg is in this dir or any subdir
            for _f in _d.rglob("ffmpeg.exe"):
                _bin_dir = str(_f.parent)
                if _bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")
                break

import asyncio
import base64
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Protocol
from uuid import UUID

import httpx

from lya.domain.models.agent import Agent
from lya.domain.models.memory import Memory
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings
from lya.infrastructure.llm.ollama_adapter import OllamaAdapter
from lya.infrastructure.llm.tool_calling import (
    build_system_prompt_with_tools,
    run_tool_calling_loop,
)
from lya.infrastructure.memory.working_memory import WorkingMemoryBuffer
from lya.infrastructure.tools.tool_registry import get_tool_registry

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TelegramMessage:
    """Represents a Telegram message."""
    message_id: int
    chat_id: int
    text: str
    username: str
    first_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TelegramCommand:
    """Represents a Telegram command."""
    command: str
    args: str
    message: TelegramMessage


@dataclass
class UserSession:
    """Represents a user session."""
    chat_id: int
    username: str
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_interaction: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    interaction_count: int = 0

    def record_interaction(self) -> None:
        self.last_interaction = datetime.now(timezone.utc)
        self.interaction_count += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Protocols
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryService(Protocol):
    async def store_memory(self, content: str, agent_id: UUID | None,
                           memory_type: str, importance: str,
                           tags: list[str] | None = None,
                           metadata: dict[str, Any] | None = None) -> None: ...

    async def recall_memories(self, query: str, agent_id: UUID | None = None,
                              limit: int = 10) -> list[Memory]: ...


class LLMService(Protocol):
    async def generate(self, prompt: str, temperature: float | None = None,
                       max_tokens: int | None = None,
                       system_prompt: str | None = None) -> str: ...

    async def chat(self, messages: list[dict[str, str]],
                   temperature: float | None = None,
                   max_tokens: int | None = None) -> str: ...


# ═══════════════════════════════════════════════════════════════════════════════
# Chat History Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class ChatHistoryStore:
    """Persistent chat history saved to disk as JSON."""

    def __init__(self, path: Path):
        self._path = path / "chat_history"
        self._path.mkdir(parents=True, exist_ok=True)

    def _file_for(self, chat_id: int) -> Path:
        return self._path / f"{chat_id}.json"

    def load(self, chat_id: int) -> list[dict[str, Any]]:
        f = self._file_for(chat_id)
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def save(self, chat_id: int, history: list[dict[str, Any]]) -> None:
        f = self._file_for(chat_id)
        try:
            f.write_text(json.dumps(history, ensure_ascii=False, default=str), encoding="utf-8")
        except OSError as e:
            logger.error("failed_to_save_chat_history", chat_id=chat_id, error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Bot
# ═══════════════════════════════════════════════════════════════════════════════

class TelegramBot:
    """Telegram bot with LLM integration, persistent history, and access control."""

    def __init__(
        self,
        bot_token: str | None = None,
        workspace: Path | None = None,
        memory_service: MemoryService | None = None,
        agent: Agent | None = None,
        llm_adapter: OllamaAdapter | None = None,
    ):
        self.bot_token = bot_token or settings.telegram.bot_token
        if not self.bot_token:
            raise ValueError(
                "Telegram bot token not configured. "
                "Set LYA_TELEGRAM_BOT_TOKEN in .env"
            )

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.workspace = workspace or settings.workspace_path
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Access control
        self.allowed_users: set[str] = set(settings.telegram.allowed_users)

        # Components
        self.memory = memory_service
        self.agent = agent
        self.llm = llm_adapter

        # State
        self.offset = 0
        self.running = False
        self.user_sessions: dict[int, UserSession] = {}
        self._client: httpx.AsyncClient | None = None

        # Short-term memory (per user session)
        self.working_memory: dict[int, WorkingMemoryBuffer] = {}

        # Persistent chat history
        self.history_store = ChatHistoryStore(self.workspace)
        self.chat_history: dict[int, list[dict[str, Any]]] = {}

        # Auto-init LLM
        if self.llm is None:
            try:
                self.llm = OllamaAdapter()
                logger.info("llm_adapter_initialized", model=self.llm.model,
                            has_api_key=bool(self.llm.api_key))
            except Exception as e:
                logger.warning("llm_adapter_init_failed", error=str(e))

        # Scheduler (Feature 6)
        self._scheduler = None
        try:
            from lya.infrastructure.scheduler.task_scheduler import TaskScheduler
            self._scheduler = TaskScheduler(self.workspace)
        except Exception as e:
            logger.debug("scheduler_init_skipped", error=str(e))

        # RAG service (Feature 7)
        self._rag_service = None
        try:
            from lya.infrastructure.rag.rag_service import RAGService
            self._rag_service = RAGService(self.workspace)
        except Exception as e:
            logger.debug("rag_init_skipped", error=str(e))

        # Conversation summarizer (Feature 5)
        self._summarizer = None
        try:
            from lya.infrastructure.memory.conversation_summarizer import ConversationSummarizer
            self._summarizer = ConversationSummarizer(self.workspace)
        except Exception as e:
            logger.debug("summarizer_init_skipped", error=str(e))

        # Preference learner (Feature 14)
        try:
            from lya.infrastructure.memory.preference_learner import PreferenceLearner
            self._preference_learner = PreferenceLearner(self.workspace)
        except Exception as e:
            logger.debug("preference_learner_init_skipped", error=str(e))

        # Interaction counter for preference learning
        self._interaction_count: dict[int, int] = {}

        logger.info("telegram_bot_initialized", has_llm=self.llm is not None,
                     allowed_users=len(self.allowed_users))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start polling loop."""
        logger.info("lya_telegram_bot_starting")
        await self._get_bot_info()

        # Start scheduler if available
        if self._scheduler:
            async def _on_task_due(task):
                await self._send_message(
                    task.chat_id,
                    f"Reminder: {task.description}",
                )
            self._scheduler.set_callback(_on_task_due)
            await self._scheduler.start()
            logger.info("scheduler_started")

        self.running = True
        logger.info("bot_running")

        while self.running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    asyncio.create_task(self._process_update(update))
                await asyncio.sleep(1)
            except Exception as e:
                logger.error("polling_error", error=str(e))
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self.running = False
        # Stop scheduler
        if self._scheduler:
            try:
                await self._scheduler.stop()
            except Exception as e:
                logger.error("scheduler_stop_failed", error=str(e))
        # Save all chat histories
        for chat_id, history in self.chat_history.items():
            self.history_store.save(chat_id, history)
        # Save working memory buffers
        for chat_id, wmb in self.working_memory.items():
            try:
                wm_path = self.workspace / "memory" / f"wm_{chat_id}.json"
                wm_path.parent.mkdir(parents=True, exist_ok=True)
                wmb.save_to_file(str(wm_path))
            except Exception as e:
                logger.error("wm_save_failed", chat_id=chat_id, error=str(e))
        # Save preferences
        if hasattr(self, "_preference_learner"):
            for chat_id in self._interaction_count:
                pass  # preferences auto-saved on update
        # Persist long-term memory
        if self.memory and hasattr(self.memory, "persist"):
            try:
                await self.memory.persist()
            except Exception as e:
                logger.error("ltm_persist_failed", error=str(e))
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("bot_stopped")

    # ── Telegram API ──────────────────────────────────────────────────────

    async def _get_bot_info(self) -> None:
        try:
            client = await self._get_client()
            r = await client.get(f"{self.base_url}/getMe")
            data = r.json()
            if data.get("ok"):
                info = data["result"]
                logger.info("bot_info", username=info.get("username"), id=info.get("id"))
        except Exception as e:
            logger.error("failed_to_get_bot_info", error=str(e))

    async def _get_updates(self) -> list[dict[str, Any]]:
        try:
            client = await self._get_client()
            r = await client.get(f"{self.base_url}/getUpdates",
                                 params={"offset": self.offset, "limit": 10, "timeout": 30},
                                 timeout=35.0)
            data = r.json()
            if data.get("ok"):
                updates = data.get("result", [])
                if updates:
                    self.offset = updates[-1]["update_id"] + 1
                return updates
            return []
        except Exception as e:
            logger.error("failed_to_get_updates", error=str(e))
            return []

    async def _send_typing(self, chat_id: int) -> None:
        try:
            client = await self._get_client()
            await client.post(f"{self.base_url}/sendChatAction",
                              json={"chat_id": chat_id, "action": "typing"})
        except Exception:
            pass

    async def _send_message(
        self, chat_id: int, text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Send message with markdown, splitting, and optional inline keyboard."""
        try:
            from lya.adapters.telegram.keyboard_builder import parse_llm_buttons

            # Parse button syntax from LLM output
            text, keyboard = parse_llm_buttons(text)
            if keyboard and reply_markup is None:
                reply_markup = keyboard.to_markup()

            client = await self._get_client()
            max_len = 4096
            last_result = None

            # Split long messages
            chunks = []
            if len(text) <= max_len:
                chunks = [text]
            else:
                remaining = text
                while remaining:
                    if len(remaining) <= max_len:
                        chunks.append(remaining)
                        break
                    split_at = remaining[:max_len].rfind("\n\n")
                    if split_at < max_len // 2:
                        split_at = remaining[:max_len].rfind("\n")
                    if split_at < max_len // 4:
                        split_at = max_len
                    chunks.append(remaining[:split_at])
                    remaining = remaining[split_at:].lstrip("\n")

            for i, chunk in enumerate(chunks):
                payload: dict[str, Any] = {
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown",
                }
                # Only add keyboard to last chunk
                if reply_markup and i == len(chunks) - 1:
                    payload["reply_markup"] = reply_markup

                response = await client.post(f"{self.base_url}/sendMessage", json=payload)

                # Fallback to plain text if Markdown fails
                if response.status_code != 200:
                    payload.pop("parse_mode")
                    response = await client.post(f"{self.base_url}/sendMessage", json=payload)

                if response.status_code == 200:
                    logger.info("message_sent", chat_id=chat_id, length=len(chunk))
                    last_result = response.json().get("result", {})
                else:
                    logger.error("send_failed", status=response.status_code,
                                 body=response.text[:200])

            return last_result

        except Exception as e:
            logger.error("failed_to_send_message", error=str(e), chat_id=chat_id)
            return None

    async def _edit_message(
        self, chat_id: int, message_id: int, text: str,
    ) -> bool:
        """Edit an existing message (used for streaming)."""
        try:
            client = await self._get_client()
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": "Markdown",
            }
            response = await client.post(
                f"{self.base_url}/editMessageText", json=payload,
            )
            if response.status_code != 200:
                payload.pop("parse_mode")
                response = await client.post(
                    f"{self.base_url}/editMessageText", json=payload,
                )
            return response.status_code == 200
        except Exception:
            return False

    async def _send_document(self, chat_id: int, file_path: str,
                             caption: str | None = None) -> None:
        """Send a file as a Telegram document."""
        try:
            from pathlib import Path
            path = Path(file_path).expanduser()
            if not path.exists():
                logger.error("file_not_found_for_send", path=str(path))
                return

            client = await self._get_client()
            data = {"chat_id": str(chat_id)}
            if caption:
                data["caption"] = caption[:1024]

            with open(path, "rb") as f:
                files = {"document": (path.name, f)}
                response = await client.post(
                    f"{self.base_url}/sendDocument",
                    data=data,
                    files=files,
                    timeout=60.0,
                )

            if response.status_code == 200:
                logger.info("document_sent", chat_id=chat_id, file=path.name)
            else:
                logger.error("document_send_failed", status=response.status_code,
                             body=response.text[:200])
        except Exception as e:
            logger.error("failed_to_send_document", error=str(e), path=file_path)

    # ── Message Processing ────────────────────────────────────────────────

    async def _process_update(self, update: dict[str, Any]) -> None:
        try:
            await self._handle_update(update)
        except Exception as e:
            logger.error("update_processing_error", error=str(e),
                         error_type=type(e).__name__)

    async def _handle_update(self, update: dict[str, Any]) -> None:
        # Handle callback queries (inline buttons)
        if "callback_query" in update:
            await self._handle_callback_query(update["callback_query"])
            return

        if "message" not in update:
            return

        message = update["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        user = message["from"]
        user_id = str(user.get("id", ""))
        username = user.get("username", user.get("first_name", "Unknown"))
        message_id = message["message_id"]

        # Handle voice messages
        if message.get("voice") or message.get("audio"):
            voice_data = message.get("voice") or message.get("audio")
            file_id = voice_data["file_id"]
            # Access control first
            if self.allowed_users and user_id not in self.allowed_users:
                await self._send_message(chat_id, "You are not authorized to use this bot.")
                return
            session = self._get_or_create_session(chat_id, username)
            session.record_interaction()
            await self._handle_voice_message(chat_id, file_id, message, session, username)
            return

        # Handle photo messages
        if message.get("photo"):
            photos = message["photo"]
            file_id = photos[-1]["file_id"]  # Largest photo
            caption = message.get("caption", "What's in this image?")
            if self.allowed_users and user_id not in self.allowed_users:
                await self._send_message(chat_id, "You are not authorized to use this bot.")
                return
            session = self._get_or_create_session(chat_id, username)
            session.record_interaction()
            msg = TelegramMessage(
                message_id=message_id, chat_id=chat_id, text=caption,
                username=username, first_name=user.get("first_name", "Unknown"),
            )
            await self._handle_photo_message(chat_id, file_id, caption, msg, session)
            return

        # Handle document uploads (for RAG)
        if message.get("document"):
            doc = message["document"]
            mime = doc.get("mime_type", "")
            if mime.startswith("image/"):
                # Image sent as document
                if self.allowed_users and user_id not in self.allowed_users:
                    await self._send_message(chat_id, "You are not authorized to use this bot.")
                    return
                session = self._get_or_create_session(chat_id, username)
                session.record_interaction()
                caption = message.get("caption", "What's in this image?")
                msg = TelegramMessage(
                    message_id=message_id, chat_id=chat_id, text=caption,
                    username=username, first_name=user.get("first_name", "Unknown"),
                )
                await self._handle_photo_message(chat_id, doc["file_id"], caption, msg, session)
                return
            elif mime in ("application/pdf", "text/plain", "text/csv",
                          "application/json", "text/markdown"):
                if self.allowed_users and user_id not in self.allowed_users:
                    await self._send_message(chat_id, "You are not authorized to use this bot.")
                    return
                session = self._get_or_create_session(chat_id, username)
                session.record_interaction()
                await self._handle_document_upload(chat_id, doc, message, session, username)
                return

        if not text:
            return

        # Safe logging — ASCII-only text preview to avoid encoding errors
        safe_text = text[:50].encode("ascii", errors="replace").decode("ascii")
        logger.info("message_received", username=username, chat_id=chat_id,
                     text=safe_text)

        # Access control
        if self.allowed_users and user_id not in self.allowed_users:
            logger.warning("unauthorized_user", user_id=user_id, username=username)
            await self._send_message(chat_id, "You are not authorized to use this bot.")
            return

        msg = TelegramMessage(
            message_id=message_id, chat_id=chat_id, text=text,
            username=username, first_name=user.get("first_name", "Unknown"),
        )

        session = self._get_or_create_session(chat_id, username)
        session.record_interaction()

        # Load history from disk if not in memory
        if chat_id not in self.chat_history:
            self.chat_history[chat_id] = self.history_store.load(chat_id)

        # Sustained typing indicator (re-send every 4s while generating)
        typing_active = True

        async def keep_typing():
            while typing_active:
                await self._send_typing(chat_id)
                await asyncio.sleep(4)

        typing_task = asyncio.create_task(keep_typing())

        try:
            # Route to handler
            if text.startswith("/"):
                response = await self._handle_command(text, msg, session)
            else:
                response = await self._handle_conversation(text, msg, session)
        finally:
            typing_active = False
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # Save to history
        self.chat_history[chat_id].append({
            "text": text, "is_user": True, "username": username,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.chat_history[chat_id].append({
            "text": response, "is_user": False, "username": "Lya",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 100 messages
        self.chat_history[chat_id] = self.chat_history[chat_id][-100:]

        # Persist to disk
        self.history_store.save(chat_id, self.chat_history[chat_id])

        await self._send_message(chat_id, response)

        # Auto-send any files that were downloaded/uploaded during tool calls
        ctx = getattr(self, "_last_tool_context", None)
        if ctx and ctx.tool_calls:
            sent_paths = set()
            for tc in ctx.tool_calls:
                if tc.tool_name in ("upload_file", "download_file", "download_video") and tc.success:
                    file_path = tc.result.get("path", "")
                    caption = tc.result.get("caption") or tc.result.get("title") or tc.result.get("filename", "")
                    if file_path and file_path not in sent_paths:
                        await self._send_document(chat_id, file_path, caption=caption)
                        sent_paths.add(file_path)
            # Clear context to prevent duplicate sends on next message
            self._last_tool_context = None

        # Store to memory service if available
        if self.memory:
            await self._store_memory(username, text, response, chat_id)

    def _get_or_create_session(self, chat_id: int, username: str) -> UserSession:
        if chat_id not in self.user_sessions:
            self.user_sessions[chat_id] = UserSession(chat_id=chat_id, username=username)
            logger.info("new_user_session", username=username, chat_id=chat_id)
        return self.user_sessions[chat_id]

    def _get_working_memory(self, chat_id: int) -> WorkingMemoryBuffer:
        """Get or create a WorkingMemoryBuffer for a chat session."""
        if chat_id not in self.working_memory:
            wmb = WorkingMemoryBuffer(max_items=200, decay_interval_seconds=600)
            # Try to load from disk
            wm_path = self.workspace / "memory" / f"wm_{chat_id}.json"
            if wm_path.exists():
                try:
                    wmb.load_from_file(str(wm_path))
                    logger.debug("wm_loaded_from_disk", chat_id=chat_id,
                                 items=len(wmb))
                except Exception:
                    pass
            self.working_memory[chat_id] = wmb
        return self.working_memory[chat_id]

    # ── File Download ─────────────────────────────────────────────────────

    async def _download_file(self, file_id: str) -> Path | None:
        """Download a file from Telegram by file_id. Returns local path."""
        try:
            client = await self._get_client()
            # Step 1: Get file path from Telegram
            r = await client.get(
                f"{self.base_url}/getFile",
                params={"file_id": file_id},
            )
            data = r.json()
            if not data.get("ok"):
                logger.error("get_file_failed", response=data)
                return None

            file_path = data["result"]["file_path"]

            # Step 2: Download the file
            download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            r = await client.get(download_url, timeout=60.0)
            if r.status_code != 200:
                logger.error("file_download_failed", status=r.status_code)
                return None

            # Step 3: Save to temp dir
            suffix = Path(file_path).suffix or ".bin"
            temp_dir = self.workspace / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            local_path = temp_dir / f"tg_{file_id}{suffix}"
            local_path.write_bytes(r.content)

            logger.info("file_downloaded", path=str(local_path), size=len(r.content))
            return local_path

        except Exception as e:
            logger.error("file_download_error", error=str(e))
            return None

    # ── Voice Handling ─────────────────────────────────────────────────────

    async def _handle_voice_message(
        self, chat_id: int, file_id: str,
        message: dict[str, Any], session: UserSession,
        username: str,
    ) -> None:
        """Handle incoming voice/audio message: STT → respond → optionally TTS."""
        await self._send_typing(chat_id)

        # Lazy-init STT adapter
        if not hasattr(self, "_stt_adapter"):
            try:
                from lya.infrastructure.voice.stt_adapter import create_stt_adapter
                self._stt_adapter = create_stt_adapter(
                    language=settings.voice.stt_language,
                )
            except Exception as e:
                logger.error("stt_init_failed", error=str(e))
                await self._send_message(chat_id, "Voice processing is not available.")
                return

        if not self._stt_adapter.is_available():
            await self._send_message(
                chat_id,
                "Voice processing requires ffmpeg. Install it from: https://ffmpeg.org/download.html",
            )
            return

        # Download voice file
        audio_path = await self._download_file(file_id)
        if not audio_path:
            await self._send_message(chat_id, "Failed to download voice message.")
            return

        try:
            # Transcribe
            text = await self._stt_adapter.transcribe(audio_path)
            if not text or text.startswith("[Could not"):
                await self._send_message(chat_id, "Could not understand the audio. Please try again.")
                return

            logger.info("voice_transcribed", text=text[:50], chat_id=chat_id)

            # Process as regular text message
            msg = TelegramMessage(
                message_id=message["message_id"],
                chat_id=chat_id,
                text=text,
                username=username,
                first_name=message["from"].get("first_name", "Unknown"),
            )

            # Load history
            if chat_id not in self.chat_history:
                self.chat_history[chat_id] = self.history_store.load(chat_id)

            response = await self._handle_conversation(text, msg, session)

            # Save to history
            self.chat_history[chat_id].append({
                "text": f"[Voice] {text}", "is_user": True, "username": username,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            self.chat_history[chat_id].append({
                "text": response, "is_user": False, "username": "Lya",
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            self.chat_history[chat_id] = self.chat_history[chat_id][-100:]
            self.history_store.save(chat_id, self.chat_history[chat_id])

            # Send text response (with transcription prefix)
            await self._send_message(chat_id, f"*[Heard]:* _{text}_\n\n{response}")

            # Optionally reply with voice
            if settings.voice.voice_reply:
                await self._send_voice_reply(chat_id, response)

            # Store memory
            if self.memory:
                await self._store_memory(username, text, response, chat_id)

        except Exception as e:
            logger.error("voice_processing_failed", error=str(e))
            await self._send_message(chat_id, f"Voice processing error: {type(e).__name__}")
        finally:
            # Clean up temp file
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    pass

    async def _send_voice_reply(self, chat_id: int, text: str) -> None:
        """Send a voice reply using TTS."""
        if not hasattr(self, "_tts_adapter"):
            try:
                from lya.infrastructure.voice.tts_adapter import create_tts_adapter
                self._tts_adapter = create_tts_adapter()
            except Exception:
                return

        if not self._tts_adapter.is_available():
            return

        try:
            temp_dir = self.workspace / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            audio_path = temp_dir / f"reply_{chat_id}.wav"
            result_path = await self._tts_adapter.synthesize(text[:500], audio_path)

            if result_path and result_path.exists():
                client = await self._get_client()
                with open(result_path, "rb") as f:
                    files = {"audio": (result_path.name, f, "audio/wav")}
                    await client.post(
                        f"{self.base_url}/sendAudio",
                        data={"chat_id": str(chat_id)},
                        files=files,
                        timeout=60.0,
                    )
                result_path.unlink(missing_ok=True)
        except Exception as e:
            logger.debug("tts_reply_failed", error=str(e))

    # ── Photo/Image Handling ───────────────────────────────────────────────

    async def _handle_photo_message(
        self, chat_id: int, file_id: str, caption: str,
        message: TelegramMessage, session: UserSession,
    ) -> None:
        """Handle photo: download → base64 → send to vision model."""
        await self._send_typing(chat_id)

        # Download image
        image_path = await self._download_file(file_id)
        if not image_path:
            await self._send_message(chat_id, "Failed to download image.")
            return

        try:
            # Read and base64 encode
            image_bytes = image_path.read_bytes()
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Load history
            if chat_id not in self.chat_history:
                self.chat_history[chat_id] = self.history_store.load(chat_id)

            # Sustained typing
            typing_active = True

            async def keep_typing():
                while typing_active:
                    await self._send_typing(chat_id)
                    await asyncio.sleep(4)

            typing_task = asyncio.create_task(keep_typing())

            try:
                response = await self._generate_ai_response(
                    caption, message, session, images=[b64_image],
                )
            finally:
                typing_active = False
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

            # Save to history
            self.chat_history[chat_id].append({
                "text": f"[Image] {caption}", "is_user": True,
                "username": message.username,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            self.chat_history[chat_id].append({
                "text": response, "is_user": False, "username": "Lya",
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            self.chat_history[chat_id] = self.chat_history[chat_id][-100:]
            self.history_store.save(chat_id, self.chat_history[chat_id])

            await self._send_message(chat_id, response)

            if self.memory:
                await self._store_memory(message.username, caption, response, chat_id)

        except Exception as e:
            logger.error("photo_processing_failed", error=str(e))
            await self._send_message(chat_id, f"Image analysis error: {type(e).__name__}")
        finally:
            if image_path and image_path.exists():
                try:
                    image_path.unlink()
                except OSError:
                    pass

    # ── Document Upload (RAG) ──────────────────────────────────────────────

    async def _handle_document_upload(
        self, chat_id: int, doc: dict[str, Any],
        message: dict[str, Any], session: UserSession,
        username: str,
    ) -> None:
        """Handle document upload for RAG indexing."""
        await self._send_typing(chat_id)

        file_name = doc.get("file_name", "document")
        file_id = doc["file_id"]

        # Download document
        doc_path = await self._download_file(file_id)
        if not doc_path:
            await self._send_message(chat_id, "Failed to download document.")
            return

        try:
            # Try RAG ingestion if available
            if hasattr(self, "_rag_service") and self._rag_service:
                chunk_count = await self._rag_service.ingest_document(
                    doc_path, chat_id, metadata={"filename": file_name},
                )
                await self._send_message(
                    chat_id,
                    f"Document *{file_name}* indexed: {chunk_count} chunks.\n"
                    "You can now ask questions about it.",
                )
            else:
                # Basic: read text content and process as message
                content = doc_path.read_text(encoding="utf-8", errors="replace")
                caption = message.get("caption", f"Analyze this document: {file_name}")
                truncated = content[:3000]

                msg = TelegramMessage(
                    message_id=message["message_id"], chat_id=chat_id,
                    text=f"{caption}\n\n---\n{truncated}",
                    username=username,
                    first_name=message["from"].get("first_name", "Unknown"),
                )

                if chat_id not in self.chat_history:
                    self.chat_history[chat_id] = self.history_store.load(chat_id)

                response = await self._handle_conversation(msg.text, msg, session)
                await self._send_message(chat_id, response)

        except Exception as e:
            logger.error("document_upload_failed", error=str(e))
            await self._send_message(chat_id, f"Document processing error: {type(e).__name__}")
        finally:
            if doc_path and doc_path.exists():
                try:
                    doc_path.unlink()
                except OSError:
                    pass

    # ── Callback Query (Inline Buttons) ────────────────────────────────────

    async def _handle_callback_query(self, callback: dict[str, Any]) -> None:
        """Handle inline button callback."""
        callback_id = callback["id"]
        data = callback.get("data", "")
        chat_id = callback["message"]["chat"]["id"]
        user = callback["from"]
        user_id = str(user.get("id", ""))
        username = user.get("username", user.get("first_name", "Unknown"))

        # Access control
        if self.allowed_users and user_id not in self.allowed_users:
            return

        # Answer the callback (required by Telegram API)
        try:
            client = await self._get_client()
            await client.post(
                f"{self.base_url}/answerCallbackQuery",
                json={"callback_query_id": callback_id},
            )
        except Exception:
            pass

        # Process callback data as if user typed it
        session = self._get_or_create_session(chat_id, username)
        session.record_interaction()

        if chat_id not in self.chat_history:
            self.chat_history[chat_id] = self.history_store.load(chat_id)

        msg = TelegramMessage(
            message_id=callback["message"]["message_id"],
            chat_id=chat_id, text=data,
            username=username,
            first_name=user.get("first_name", "Unknown"),
        )

        typing_active = True

        async def keep_typing():
            while typing_active:
                await self._send_typing(chat_id)
                await asyncio.sleep(4)

        typing_task = asyncio.create_task(keep_typing())

        try:
            response = await self._handle_conversation(data, msg, session)
        finally:
            typing_active = False
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        self.chat_history[chat_id].append({
            "text": f"[Button] {data}", "is_user": True, "username": username,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.chat_history[chat_id].append({
            "text": response, "is_user": False, "username": "Lya",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.chat_history[chat_id] = self.chat_history[chat_id][-100:]
        self.history_store.save(chat_id, self.chat_history[chat_id])
        await self._send_message(chat_id, response)

    # ── Command Handlers ──────────────────────────────────────────────────

    async def _handle_command(self, text: str, message: TelegramMessage,
                              session: UserSession) -> str:
        parts = text.split(maxsplit=1)
        command = parts[0].lower().split("@")[0]  # Handle /cmd@botname
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/start": self._cmd_start,
            "/help": self._cmd_help,
            "/remember": self._cmd_remember,
            "/recall": self._cmd_recall,
            "/forget": self._cmd_forget,
            "/history": self._cmd_history,
            "/remind": self._cmd_remind,
            "/tasks": self._cmd_tasks,
            "/cancel": self._cmd_cancel,
            "/ask": self._cmd_ask,
            "/docs": self._cmd_docs,
            "/preferences": self._cmd_preferences,
            "/set_preference": self._cmd_set_preference,
            "/improve": self._cmd_improve,
            "/capabilities": self._cmd_capabilities,
            "/plugins": self._cmd_plugins,
            "/sandbox": self._cmd_sandbox,
        }

        handler = handlers.get(command)
        if handler:
            return await handler(args, message, session)
        return await self._handle_conversation(text, message, session)

    async def _cmd_start(self, args: str, message: TelegramMessage,
                         session: UserSession) -> str:
        prompt = (f"The user {message.username} just started a conversation. "
                  "Greet them and briefly introduce yourself and what you can do.")
        return await self._handle_conversation(prompt, message, session)

    async def _cmd_help(self, args: str, message: TelegramMessage,
                        session: UserSession) -> str:
        return (
            "**Lya Commands:**\n\n"
            "/start - Start conversation\n"
            "/help - Show this help\n"
            "/remember <text> - Save a note\n"
            "/recall <query> - Search memories\n"
            "/forget - Clear short-term memory\n"
            "/history - Chat history info\n"
            "/remind <time> <text> - Set reminder\n"
            "/tasks - List scheduled tasks\n"
            "/cancel <id> - Cancel a task\n"
            "/ask <question> - Query uploaded docs\n"
            "/docs - List uploaded documents\n"
            "/preferences - View learned preferences\n"
            "/set\\_preference <key> <value> - Set preference\n"
            "/improve <goal> - Self-improve: generate a new tool\n"
            "/capabilities - List registered capabilities\n"
            "/plugins - List loaded plugins/tools\n"
            "/sandbox - Show sandbox config & status\n\n"
            "**Media:**\n"
            "Send voice messages, photos, or documents.\n"
            "I can transcribe voice, analyze images, and index documents."
        )

    async def _cmd_remember(self, args: str, message: TelegramMessage,
                            session: UserSession) -> str:
        text = args.strip()
        if not text:
            return "Usage: /remember <text>"

        # Store in short-term memory (high importance so it persists)
        wmb = self._get_working_memory(message.chat_id)
        wmb.add(
            content=text,
            category="user_note",
            importance=0.9,
            source=message.username,
            metadata={"type": "explicit_remember"},
        )

        # Store in long-term memory
        if self.memory:
            try:
                await self.memory.store_memory(
                    content=text,
                    agent_id=self.agent.id if self.agent else None,
                    memory_type="SEMANTIC", importance="HIGH",
                    tags=["telegram", message.username, "user_note"],
                    metadata={"chat_id": message.chat_id},
                )
            except Exception as e:
                logger.error("memory_store_failed", error=str(e))

        prompt = f'The user asked you to remember: "{text}". Confirm briefly.'
        return await self._handle_conversation(prompt, message, session)

    async def _cmd_recall(self, args: str, message: TelegramMessage,
                          session: UserSession) -> str:
        query = args.strip()
        if not query:
            return "Usage: /recall <query>"

        results_parts = []

        # Search long-term memory
        if self.memory:
            try:
                memories = await self.memory.recall_memories(
                    query=query, limit=5,
                )
                if memories:
                    results_parts.append("**Long-term memories:**")
                    for i, mem in enumerate(memories, 1):
                        similarity = ""
                        if mem.context and mem.context.metadata:
                            sim = mem.context.metadata.get("similarity")
                            if sim:
                                similarity = f" ({sim:.0%})"
                        results_parts.append(
                            f"{i}. {mem.content[:200]}{similarity}"
                        )
            except Exception as e:
                logger.error("recall_failed", error=str(e))

        # Search short-term memory
        wmb = self._get_working_memory(message.chat_id)
        stm_items = wmb.retrieve(query=query, limit=3)
        if stm_items:
            results_parts.append("\n**Short-term memories:**")
            for item in stm_items:
                results_parts.append(f"- {item.content[:200]}")

        if not results_parts:
            return f"No memories found for: {query}"

        return "\n".join(results_parts)

    async def _cmd_forget(self, args: str, message: TelegramMessage,
                          session: UserSession) -> str:
        # Clear short-term memory for this session
        wmb = self._get_working_memory(message.chat_id)
        count = wmb.clear()
        return f"Cleared {count} items from short-term memory."

    async def _cmd_history(self, args: str, message: TelegramMessage,
                           session: UserSession) -> str:
        count = session.interaction_count
        since = session.first_seen.strftime("%Y-%m-%d")
        prompt = (f"The user asked about chat history. You've had {count} "
                  f"interactions since {since}. Summarize briefly.")
        return await self._handle_conversation(prompt, message, session)

    async def _cmd_remind(self, args: str, message: TelegramMessage,
                          session: UserSession) -> str:
        """Set a reminder: /remind in 5 minutes call mom"""
        text = args.strip()
        if not text:
            return "Usage: /remind <time> <description>\nExamples:\n/remind in 5 minutes call mom\n/remind at 17:00 meeting\n/remind tomorrow check email"

        if not hasattr(self, "_scheduler") or not self._scheduler:
            return "Scheduler not available."

        from lya.infrastructure.scheduler.time_parser import (
            parse_time_expression,
            format_task_time,
        )

        trigger_time, recurring = parse_time_expression(text)
        if not trigger_time:
            # Try to use LLM to parse
            return "Could not parse time. Try: /remind in 5 minutes <text>"

        task = await self._scheduler.schedule(
            chat_id=message.chat_id,
            description=text,
            trigger_time=trigger_time,
            recurring=recurring,
        )

        time_str = format_task_time(task.trigger_time)
        rec_str = f" (recurring: {recurring})" if recurring else ""
        return f"Reminder set {time_str}{rec_str}\nID: `{task.id}`"

    async def _cmd_tasks(self, args: str, message: TelegramMessage,
                         session: UserSession) -> str:
        """List scheduled tasks."""
        if not hasattr(self, "_scheduler") or not self._scheduler:
            return "Scheduler not available."

        from lya.infrastructure.scheduler.time_parser import format_task_time

        tasks = await self._scheduler.list_tasks(chat_id=message.chat_id)
        if not tasks:
            return "No pending tasks."

        lines = ["**Scheduled Tasks:**\n"]
        for task in tasks:
            time_str = format_task_time(task.trigger_time)
            rec = f" [{task.recurring}]" if task.recurring else ""
            lines.append(f"- `{task.id}` {task.description[:80]} ({time_str}){rec}")
        return "\n".join(lines)

    async def _cmd_cancel(self, args: str, message: TelegramMessage,
                          session: UserSession) -> str:
        """Cancel a scheduled task."""
        task_id = args.strip()
        if not task_id:
            return "Usage: /cancel <task_id>"

        if not hasattr(self, "_scheduler") or not self._scheduler:
            return "Scheduler not available."

        if await self._scheduler.cancel(task_id):
            return f"Task `{task_id}` cancelled."
        return f"Task `{task_id}` not found."

    async def _cmd_ask(self, args: str, message: TelegramMessage,
                       session: UserSession) -> str:
        """Query uploaded documents."""
        query = args.strip()
        if not query:
            return "Usage: /ask <question about your documents>"

        if not hasattr(self, "_rag_service") or not self._rag_service:
            return "Document Q&A not available. Upload a document first."

        results = await self._rag_service.query(query, message.chat_id, top_k=5)
        if not results:
            return "No relevant content found in your documents."

        # Build context from retrieved chunks
        context_parts = []
        for r in results:
            fname = r.get("metadata", {}).get("filename", "doc")
            context_parts.append(f"[{fname}] {r['content'][:300]}")

        doc_context = "\n---\n".join(context_parts)

        prompt = (
            f"The user asks about their uploaded documents.\n\n"
            f"## Retrieved Document Context\n{doc_context}\n\n"
            f"Question: {query}\n\n"
            "Answer based on the document context above."
        )
        return await self._handle_conversation(prompt, message, session)

    async def _cmd_docs(self, args: str, message: TelegramMessage,
                        session: UserSession) -> str:
        """List uploaded documents."""
        if not hasattr(self, "_rag_service") or not self._rag_service:
            return "No documents uploaded yet."

        docs = await self._rag_service.list_documents(message.chat_id)
        if not docs:
            return "No documents uploaded. Send a PDF or text file to index it."

        lines = ["**Indexed Documents:**\n"]
        for doc in docs:
            lines.append(
                f"- {doc['filename']} ({doc['chunks']} chunks)"
            )
        return "\n".join(lines)

    async def _cmd_preferences(self, args: str, message: TelegramMessage,
                               session: UserSession) -> str:
        """Show learned user preferences."""
        if not hasattr(self, "_preference_learner"):
            return "Preference learning not available."

        prefs = self._preference_learner.load_preferences(message.chat_id)
        context = prefs.to_prompt_context()
        if not context:
            return "No preferences learned yet. Just keep chatting!"
        return context

    async def _cmd_set_preference(self, args: str, message: TelegramMessage,
                                  session: UserSession) -> str:
        """Manually set a preference: /set_preference language Persian"""
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /set_preference <key> <value>\nKeys: language, timezone, verbosity, code_language"

        key, value = parts[0], parts[1]
        if not hasattr(self, "_preference_learner"):
            return "Preference learning not available."

        prefs = self._preference_learner.update_from_signal(
            message.chat_id, {key: value},
        )
        return f"Preference set: {key} = {value}"

    # ── Feature 9: Self-Improvement ──────────────────────────────────────

    async def _cmd_improve(self, args: str, message: TelegramMessage,
                           session: UserSession) -> str:
        """Self-improve: generate a new tool from a description."""
        goal = args.strip()
        if not goal:
            return "Usage: /improve <goal>\nExample: /improve create a unit converter tool"

        if not self.llm:
            return "LLM not connected. Cannot generate tools."

        try:
            from lya.infrastructure.self_improvement.self_improvement_service import (
                SelfImprovementService,
            )

            service = SelfImprovementService(
                workspace=self.workspace,
                llm_interface=self.llm,
                tool_registry=get_tool_registry(),
            )

            await self._send_message(
                message.chat_id, "Generating tool... This may take a moment.",
            )

            result = await service.improve(goal=goal)

            if result.get("success"):
                return (
                    f"Tool generated: **{result.get('tool_name', 'unknown')}**\n"
                    f"Path: `{result.get('tool_path', 'N/A')}`\n"
                    f"{result.get('message', '')}"
                )
            return f"Generation failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error("self_improvement_failed", error=str(e))
            return f"Self-improvement failed: {e}"

    async def _cmd_capabilities(self, args: str, message: TelegramMessage,
                                session: UserSession) -> str:
        """List registered capabilities."""
        try:
            from lya.infrastructure.tools.capability_registry import get_registry

            registry = get_registry()
            caps = registry.list_capabilities()
            funcs = registry.list_functions()

            if not caps and not funcs:
                # Fall back to tool registry
                tool_reg = get_tool_registry()
                tools = tool_reg.list_tools()
                if tools:
                    tool_list = "\n".join(f"- `{t}`" for t in tools[:20])
                    return f"**Available Tools** ({len(tools)}):\n{tool_list}"
                return "No capabilities or tools registered yet.\nUse /improve to generate new ones."

            parts = [f"**Capabilities** ({len(caps)}):"]
            for cap_id in caps[:15]:
                parts.append(f"- `{cap_id}`")

            if funcs:
                parts.append(f"\n**Functions** ({len(funcs)}):")
                for func_name in funcs[:15]:
                    parts.append(f"- `{func_name}`")

            return "\n".join(parts)

        except Exception as e:
            logger.error("capabilities_list_failed", error=str(e))
            return f"Error listing capabilities: {e}"

    # ── Feature 11: Plugin System ────────────────────────────────────────

    async def _cmd_plugins(self, args: str, message: TelegramMessage,
                           session: UserSession) -> str:
        """List loaded plugins/generated tools."""
        try:
            from lya.infrastructure.self_improvement.tool_loader import ToolLoader

            loader = ToolLoader(self.workspace)
            tools = loader.list_tools()

            if not tools:
                return "No plugins installed.\nUse /improve <goal> to generate tools."

            lines = [f"**Installed Plugins** ({len(tools)}):"]
            for t in tools[:20]:
                loaded = loader.is_loaded(t) if hasattr(loader, "is_loaded") else False
                status = "loaded" if loaded else "available"
                lines.append(f"- `{t}` ({status})")

            return "\n".join(lines)

        except Exception as e:
            logger.error("plugins_list_failed", error=str(e))
            return f"Error listing plugins: {e}"

    # ── Feature 12: Sandbox Status ───────────────────────────────────────

    async def _cmd_sandbox(self, args: str, message: TelegramMessage,
                           session: UserSession) -> str:
        """Show sandbox configuration and status."""
        sec = settings.security
        import shutil

        docker_available = shutil.which("docker") is not None

        lines = [
            "**Sandbox Configuration:**",
            f"- Enabled: `{sec.sandbox_enabled}`",
            f"- Type: `{sec.sandbox_type}`",
            f"- Docker image: `{sec.docker_image}`",
            f"- Docker available: `{docker_available}`",
            f"- File write: `{sec.allow_file_write}`",
            f"- Network access: `{sec.allow_network_access}`",
            f"- Blocked commands: `{', '.join(sec.blocked_commands[:5])}`",
        ]

        if sec.sandbox_type == "docker" and not docker_available:
            lines.append("\nDocker not found. Falling back to process sandbox.")

        return "\n".join(lines)

    # ── LLM Integration ──────────────────────────────────────────────────

    async def _handle_conversation(self, text: str, message: TelegramMessage,
                                   session: UserSession) -> str:
        if self.llm:
            try:
                return await self._generate_ai_response(text, message, session)
            except ConnectionError as e:
                logger.error("llm_connection_failed", error=str(e))
                return (
                    "\u26a0\ufe0f LLM service is unreachable. "
                    "Ollama might not be running.\n\n"
                    "Start it with: `ollama serve`"
                )
            except Exception as e:
                logger.error("llm_generation_failed", error=str(e),
                             error_type=type(e).__name__)
                return f"\u26a0\ufe0f Something went wrong: {type(e).__name__}"
        return "\u26a0\ufe0f LLM not connected. Check the configuration."

    async def generate_response(self, text: str, username: str) -> str:
        """Public API for generating responses outside of Telegram context."""
        if not self.llm:
            return "LLM not connected."
        try:
            session = UserSession(chat_id=0, username=username)
            msg = TelegramMessage(message_id=0, chat_id=0, text=text,
                                  username=username, first_name=username)
            return await self._generate_ai_response(text, msg, session)
        except Exception as e:
            logger.error("generate_response_failed", error=str(e))
            return "Error generating response."

    async def _generate_ai_response(
        self, text: str, message: TelegramMessage,
        session: UserSession, images: list[str] | None = None,
    ) -> str:
        chat_context = self._build_chat_context(message.chat_id)

        base_prompt = (
            f"You are Lya, a senior software engineer and AI coding agent created by Shoji. "
            f"You are working with {message.username} via Telegram.\n\n"
            "You are an expert in Python, JavaScript/TypeScript, Dart/Flutter, web frameworks, "
            "databases, DevOps, and system design. You build, debug, and ship real software.\n\n"
            "## Workflow\n"
            "1. **Explore**: Use `project_tree`, `file_read`, `grep`, `find_files` to understand code.\n"
            "2. **Implement**: `file_write` for new files, `file_edit` for changes. Multiple tool_call blocks OK.\n"
            "3. **Test**: `execute_command` to run tests/builds. Fix and re-run on failure.\n"
            "4. **Download**: `download_file` for URLs, `download_video` for YouTube/Instagram/Twitter/TikTok, "
            "`web_scrape` for news (Persian/Farsi supported).\n"
            "5. **Deliver**: `upload_file` to send files to the user.\n"
            "6. **Version**: `git_add` + `git_commit` + `git_push`.\n"
            "7. **Schedule**: `schedule_task` for recurring jobs (daily news, reminders).\n\n"
            "## Rules\n"
            "- **NEVER paste code into chat.** Always use `file_write` or `file_edit`.\n"
            "- Read existing code BEFORE modifying it.\n"
            "- For recurring tasks, use `schedule_task` — do NOT write cron/scheduler code.\n"
            "- Be direct. No filler. No disclaimers.\n"
            "- When asked to write code, ALWAYS use file_write — never output code in your message.\n"
            "- **NEVER tell the user to run commands themselves.** YOU have the tools — USE them.\n"
            "- When asked to download a video/audio, ALWAYS use `download_video` tool directly. "
            "Do NOT say 'ffmpeg is not installed' or give bash commands. Just call the tool.\n"
            "- When asked to download a file, ALWAYS use `download_file` tool. Do NOT give URLs for the user to visit.\n"
            "- After downloading, ALWAYS use `upload_file` to send the file to the user.\n"
            "- You are an autonomous agent. DO things, don't tell the user how to do things."
        )

        # ── Memory augmentation ──
        memory_context = await self._build_memory_context(
            text, message.chat_id, message.username,
        )
        if memory_context:
            base_prompt += f"\n\n{memory_context}"

        # ── Preference context ──
        if hasattr(self, "_preference_learner"):
            try:
                prefs = self._preference_learner.load_preferences(message.chat_id)
                pref_context = prefs.to_prompt_context()
                if pref_context:
                    base_prompt += f"\n\n{pref_context}"

                # Detect explicit preference signals
                signal = self._preference_learner.detect_preference_signal(text)
                if signal:
                    self._preference_learner.update_from_signal(
                        message.chat_id, signal,
                    )
            except Exception:
                pass

        # ── RAG context ──
        if self._rag_service:
            try:
                rag_results = await self._rag_service.query(
                    text, message.chat_id, top_k=3,
                )
                if rag_results:
                    rag_parts = [r["content"][:300] for r in rag_results]
                    base_prompt += (
                        "\n\n## Relevant Document Context\n"
                        + "\n---\n".join(rag_parts)
                    )
            except Exception:
                pass

        # Augment with tool descriptions
        registry = get_tool_registry()

        # Inject scheduler so LLM can use schedule_task / list_scheduled_tasks / cancel_task
        if self._scheduler and not registry._scheduler:
            registry.set_scheduler(self._scheduler, message.chat_id)
        elif self._scheduler:
            # Update chat_id for this request
            registry._scheduler_chat_id = message.chat_id

        system_prompt = build_system_prompt_with_tools(base_prompt, registry)

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation summary if available
        if self._summarizer:
            summary = self._summarizer.load_summary(message.chat_id)
            if summary:
                messages.append({
                    "role": "system",
                    "content": f"Summary of earlier conversation: {summary}",
                })

        if chat_context:
            messages.extend(chat_context[-20:])

        messages.append({"role": "user", "content": text})

        # Use vision model for image requests
        llm_to_use = self.llm
        if images and self.llm:
            vision_model = settings.llm.vision_model
            if vision_model != self.llm.model:
                llm_to_use = OllamaAdapter(
                    base_url=self.llm.base_url,
                    model=vision_model,
                    api_key=self.llm.api_key,
                )
        elif settings.llm.routing_enabled and self.llm and not images:
            # Multi-model routing (Feature 13)
            try:
                from lya.infrastructure.llm.model_router import ModelRouter
                router = ModelRouter()
                routed_model = router.get_model_for_task(text)
                if routed_model and routed_model != self.llm.model:
                    llm_to_use = self.llm.with_model(routed_model)
                    logger.info("model_routed", task=text[:50], model=routed_model)
            except Exception:
                pass

        response, ctx = await run_tool_calling_loop(
            llm=llm_to_use,
            messages=messages,
            registry=registry,
            max_iterations=15,
            temperature=0.7,
            max_tokens=settings.llm.max_tokens,
            images=images,
        )

        if ctx.tool_calls:
            logger.info(
                "tool_calls_used",
                count=len(ctx.tool_calls),
                tools=[tc.tool_name for tc in ctx.tool_calls],
            )

        self._last_tool_context = ctx

        # ── Store in short-term memory ──
        wmb = self._get_working_memory(message.chat_id)
        wmb.add(
            content=f"{message.username}: {text}",
            category="conversation",
            importance=0.5,
            source="user",
        )
        wmb.add(
            content=f"Lya: {response[:500]}",
            category="conversation",
            importance=0.4,
            source="assistant",
        )

        # ── Background tasks ──

        # Conversation summarization (Feature 5)
        if self._summarizer and self.llm:
            history = self.chat_history.get(message.chat_id, [])
            if self._summarizer.should_summarize(history):
                asyncio.create_task(
                    self._summarize_old_messages(message.chat_id)
                )

        # Periodic preference learning (Feature 14)
        if hasattr(self, "_preference_learner") and self.llm:
            count = self._interaction_count.get(message.chat_id, 0) + 1
            self._interaction_count[message.chat_id] = count
            interval = settings.telegram.preference_learning_interval
            if count % interval == 0:
                asyncio.create_task(
                    self._learn_preferences(message.chat_id)
                )

        return response

    async def _summarize_old_messages(self, chat_id: int) -> None:
        """Background task: summarize old messages."""
        try:
            history = self.chat_history.get(chat_id, [])
            if len(history) <= 40:
                return

            # Summarize the oldest 20 messages
            old_messages = history[:20]
            existing_summary = self._summarizer.load_summary(chat_id)

            summary = await self._summarizer.summarize(
                self.llm, old_messages, existing_summary,
            )

            if summary:
                self._summarizer.save_summary(chat_id, summary)
                # Keep only recent messages
                self.chat_history[chat_id] = history[20:]
                self.history_store.save(chat_id, self.chat_history[chat_id])
                logger.info("conversation_summarized", chat_id=chat_id)

        except Exception as e:
            logger.debug("summarization_background_failed", error=str(e))

    async def _learn_preferences(self, chat_id: int) -> None:
        """Background task: analyze conversation for preferences."""
        try:
            history = self.chat_history.get(chat_id, [])
            if not history:
                return

            prefs = self._preference_learner.load_preferences(chat_id)
            updated = await self._preference_learner.analyze_conversation(
                self.llm, history, prefs,
            )
            self._preference_learner.save_preferences(chat_id, updated)
            logger.info("preferences_updated", chat_id=chat_id)

        except Exception as e:
            logger.debug("preference_learning_failed", error=str(e))

    async def _build_memory_context(
        self, query: str, chat_id: int, username: str,
    ) -> str:
        """Build memory context to inject into the LLM system prompt.

        Combines:
        - Short-term memory (WorkingMemoryBuffer): recent facts, user notes
        - Long-term memory (ChromaDB): semantically relevant past memories
        """
        parts: list[str] = []

        # 1. Short-term memory: recent user notes and important facts
        wmb = self._get_working_memory(chat_id)
        stm_items = wmb.retrieve(
            category="user_note",
            min_importance=0.5,
            limit=5,
        )
        if stm_items:
            notes = [f"- {item.content[:200]}" for item in stm_items]
            parts.append("Things the user asked you to remember:\n" + "\n".join(notes))

        # 2. Long-term memory: recall relevant memories
        if self.memory:
            try:
                memories = await self.memory.recall_memories(
                    query=query,
                    agent_id=self.agent.id if self.agent else None,
                    limit=3,
                )
                if memories:
                    ltm_items = [f"- {m.content[:200]}" for m in memories]
                    parts.append(
                        "Relevant past memories:\n" + "\n".join(ltm_items)
                    )
            except Exception as e:
                logger.debug("ltm_recall_skipped", error=str(e))

        if not parts:
            return ""

        return "## Your Memory\n" + "\n\n".join(parts)

    def _build_chat_context(self, chat_id: int) -> list[dict[str, str]]:
        context = []
        history = self.chat_history.get(chat_id, [])
        for msg in history[-20:]:
            role = "user" if msg.get("is_user") else "assistant"
            context.append({"role": role, "content": msg.get("text", "")})
        return context

    # ── Memory ────────────────────────────────────────────────────────────

    async def _store_memory(self, username: str, user_text: str,
                            bot_response: str, chat_id: int) -> None:
        if not self.memory:
            return

        # Determine importance based on content
        importance = "LOW"
        text_lower = user_text.lower()
        if any(kw in text_lower for kw in [
            "remember", "important", "note", "save", "don't forget",
            "my name", "i am", "i like", "i hate", "i prefer",
        ]):
            importance = "MEDIUM"
        if len(user_text) > 200:
            importance = "MEDIUM"

        try:
            await self.memory.store_memory(
                content=f"@{username}: {user_text} | Lya: {bot_response[:300]}",
                agent_id=self.agent.id if self.agent else None,
                memory_type="CONVERSATION", importance=importance,
                tags=["telegram", username],
                metadata={"chat_id": chat_id},
            )
        except Exception as e:
            logger.error("failed_to_store_memory", error=str(e))

        # Periodically consolidate short-term → long-term
        wmb = self._get_working_memory(chat_id)
        to_consolidate = wmb.consolidate_to_long_term(
            min_importance=0.7, min_access_count=2,
        )
        for item_data in to_consolidate:
            try:
                await self.memory.store_memory(
                    content=item_data["content"],
                    agent_id=self.agent.id if self.agent else None,
                    memory_type="SEMANTIC", importance="MEDIUM",
                    tags=["telegram", username, "consolidated"],
                    metadata={"chat_id": chat_id, "source": "short_term"},
                )
            except Exception as e:
                logger.debug("consolidation_failed", error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Runner and Factory
# ═══════════════════════════════════════════════════════════════════════════════

class TelegramBotRunner:
    def __init__(self, bot: TelegramBot | None = None) -> None:
        self.bot = bot
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.bot is None:
            self.bot = TelegramBot()
        self._task = asyncio.create_task(self.bot.start())

    async def stop(self) -> None:
        if self.bot:
            await self.bot.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class TelegramBotAdapter:
    def __init__(self, bot_token: str | None = None,
                 memory_service: MemoryService | None = None,
                 llm_service: LLMService | None = None,
                 agent: Agent | None = None):
        self.bot = TelegramBot(bot_token=bot_token, memory_service=memory_service,
                               llm_adapter=llm_service, agent=agent)
        self.runner = TelegramBotRunner(self.bot)

    async def start(self) -> None:
        await self.runner.start()

    async def stop(self) -> None:
        await self.runner.stop()


_bot_instance: TelegramBot | None = None


def get_telegram_bot(
    bot_token: str | None = None,
    workspace: Path | None = None,
    memory_service: MemoryService | None = None,
    llm_service: LLMService | None = None,
    agent: Agent | None = None,
) -> TelegramBot:
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TelegramBot(
            bot_token=bot_token, workspace=workspace,
            memory_service=memory_service, llm_adapter=llm_service, agent=agent,
        )
    return _bot_instance


def reset_bot_instance() -> None:
    global _bot_instance
    _bot_instance = None


__all__ = [
    "TelegramBot", "TelegramBotRunner", "TelegramBotAdapter",
    "TelegramMessage", "TelegramCommand", "UserSession",
    "MemoryService", "LLMService",
    "get_telegram_bot", "reset_bot_instance",
]
