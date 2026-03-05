"""Speech-to-Text Adapter.

Pure Python 3.14 compatible STT using SpeechRecognition library.
Supports Google Web Speech API (free, no API key) with OGG→WAV conversion.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Protocol

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class STTAdapter(Protocol):
    """Speech-to-text protocol."""

    async def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file to text."""
        ...

    def is_available(self) -> bool:
        """Check if the STT engine is operational."""
        ...


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _convert_ogg_to_wav(ogg_path: Path, wav_path: Path) -> bool:
    """Convert OGG/Opus to WAV using ffmpeg subprocess."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(ogg_path),
                "-ar", "16000", "-ac", "1", "-f", "wav",
                str(wav_path),
            ],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0 and wav_path.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error("ffmpeg_conversion_failed", error=str(e))
        return False


class GoogleSTTAdapter:
    """STT using Google Web Speech API via SpeechRecognition library.

    Free, no API key required. Requires ffmpeg for OGG→WAV conversion.
    """

    def __init__(self, language: str = "en-US"):
        self.language = language
        self._has_ffmpeg = _check_ffmpeg()
        self._recognizer = None

        if not self._has_ffmpeg:
            logger.warning(
                "ffmpeg_not_found",
                message="Voice transcription requires ffmpeg. "
                        "Install it: https://ffmpeg.org/download.html",
            )

    def _get_recognizer(self):
        """Lazy-load speech_recognition."""
        if self._recognizer is None:
            try:
                import speech_recognition as sr
                self._recognizer = sr.Recognizer()
            except ImportError:
                raise RuntimeError(
                    "SpeechRecognition not installed. "
                    "Run: pip install SpeechRecognition"
                )
        return self._recognizer

    def is_available(self) -> bool:
        try:
            self._get_recognizer()
            return self._has_ffmpeg
        except RuntimeError:
            return False

    async def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file to text.

        Accepts OGG (Telegram voice) or WAV files.
        Converts OGG→WAV via ffmpeg, then uses Google STT.
        """
        import speech_recognition as sr

        recognizer = self._get_recognizer()
        wav_path = audio_path

        # Convert OGG to WAV if needed
        tmp_wav = None
        if audio_path.suffix.lower() in (".ogg", ".oga", ".opus"):
            if not self._has_ffmpeg:
                raise RuntimeError(
                    "Cannot transcribe OGG voice: ffmpeg not installed. "
                    "Install ffmpeg: https://ffmpeg.org/download.html"
                )
            tmp_wav = audio_path.with_suffix(".wav")
            if not _convert_ogg_to_wav(audio_path, tmp_wav):
                raise RuntimeError("Failed to convert OGG to WAV via ffmpeg")
            wav_path = tmp_wav

        try:
            with sr.AudioFile(str(wav_path)) as source:
                audio_data = recognizer.record(source)

            text = recognizer.recognize_google(
                audio_data, language=self.language,
            )
            logger.info("stt_transcribed", length=len(text), language=self.language)
            return text

        except sr.UnknownValueError:
            return "[Could not understand the audio]"
        except sr.RequestError as e:
            raise RuntimeError(f"Google STT API error: {e}") from e
        finally:
            # Clean up temp WAV
            if tmp_wav and tmp_wav.exists():
                try:
                    tmp_wav.unlink()
                except OSError:
                    pass


class OllamaWhisperSTTAdapter:
    """STT using Ollama's Whisper endpoint (if available).

    Pure HTTP — no external binary dependencies.
    Note: Ollama does not natively support audio models yet,
    so this is a placeholder for future Ollama audio support.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.llm.base_url

    def is_available(self) -> bool:
        return False  # Ollama doesn't support audio yet

    async def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError("Ollama Whisper not yet supported")


def create_stt_adapter(
    engine: str | None = None,
    language: str = "en-US",
) -> STTAdapter:
    """Factory to create an STT adapter based on settings."""
    engine = engine or settings.voice.stt_engine

    if engine == "google":
        return GoogleSTTAdapter(language=language)
    elif engine == "whisper":
        # Try Google as fallback since faster-whisper may not work on 3.14
        adapter = GoogleSTTAdapter(language=language)
        if adapter.is_available():
            return adapter
        logger.warning("whisper_unavailable_using_google_fallback")
        return adapter
    else:
        return GoogleSTTAdapter(language=language)
