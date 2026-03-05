"""Telegram bot adapter for Lya.

Provides a Telegram interface for interacting with the Lya agent.
Features include command handling and memory integration.

Example:
    >>> from lya.adapters.telegram import TelegramBotAdapter
    >>> adapter = TelegramBotAdapter()
    >>> await adapter.start()
"""

from .telegram_bot import (
    LLMService,
    MemoryService,
    TelegramBot,
    TelegramBotAdapter,
    TelegramBotRunner,
    TelegramCommand,
    TelegramMessage,
    UserSession,
    get_telegram_bot,
    reset_bot_instance,
)

__all__ = [
    # Core bot classes
    "TelegramBot",
    "TelegramBotRunner",
    "TelegramBotAdapter",
    # Data models
    "TelegramMessage",
    "TelegramCommand",
    "UserSession",
    # Services
    "MemoryService",
    "LLMService",
    # Factory functions
    "get_telegram_bot",
    "reset_bot_instance",
]
