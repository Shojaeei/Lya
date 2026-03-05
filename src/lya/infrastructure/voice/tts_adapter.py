"""Text-to-Speech Adapter.

Pure Python 3.14 compatible TTS using pyttsx3.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class TTSAdapter(Protocol):
    """Text-to-speech protocol."""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """Synthesize text to audio file. Returns path to audio file."""
        ...

    def is_available(self) -> bool:
        """Check if TTS engine is operational."""
        ...


class Pyttsx3TTSAdapter:
    """TTS using pyttsx3 (offline, pure Python).

    Generates WAV audio files from text.
    """

    def __init__(self, rate: int = 150, volume: float = 0.9):
        self.rate = rate
        self.volume = volume
        self._engine = None

    def _get_engine(self):
        """Lazy-load pyttsx3 engine."""
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)
            except Exception as e:
                raise RuntimeError(f"pyttsx3 init failed: {e}") from e
        return self._engine

    def is_available(self) -> bool:
        try:
            self._get_engine()
            return True
        except (RuntimeError, ImportError):
            return False

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """Generate audio file from text.

        pyttsx3 saves as WAV. For Telegram sendVoice we'd need OGG,
        but Telegram also accepts WAV via sendAudio.
        """
        engine = self._get_engine()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = output_path.with_suffix(".wav")

        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()

        if wav_path.exists():
            logger.info("tts_synthesized", path=str(wav_path), text_len=len(text))
            return wav_path

        raise RuntimeError("TTS failed to generate audio file")


def create_tts_adapter(
    engine: str | None = None,
) -> TTSAdapter:
    """Factory to create a TTS adapter based on settings."""
    engine = engine or settings.voice.tts_engine

    if engine == "pyttsx3":
        return Pyttsx3TTSAdapter()
    else:
        return Pyttsx3TTSAdapter()
