"""Application settings using Pydantic."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Get project root for .env file location
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
_ENV_FILE_PATH = str(_PROJECT_ROOT / ".env")


class LLMSettings(BaseSettings):
    """LLM provider settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_LLM_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    model: str = "kimi-k2.5:cloud"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=60, gt=0)

    # API keys for cloud providers
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_api_key: str | None = None

    # Vision model for image understanding
    vision_model: str = "llava:latest"

    # Multi-model routing
    routing_enabled: bool = False
    model_routes: dict[str, str] = Field(default_factory=dict)


class MemorySettings(BaseSettings):
    """Memory and vector DB settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_MEMORY_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vector_db: Literal["chroma", "qdrant", "pinecone"] = "chroma"
    db_path: Path = Path("~/.lya/memory")
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    # Qdrant specific
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None

    @validator("db_path")
    @classmethod
    def expand_db_path(cls, v: Path) -> Path:
        return v.expanduser()


class SecuritySettings(BaseSettings):
    """Security settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_SECURITY_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sandbox_enabled: bool = True
    sandbox_type: Literal["process", "docker"] = "process"
    docker_image: str = "python:3.11-slim"
    allow_file_write: bool = True
    allow_network_access: bool = True
    allowed_directories: list[str] = Field(default_factory=lambda: ["~/.lya/workspace"])
    blocked_commands: list[str] = Field(default_factory=lambda: ["rm -rf", "dd", "format", "mkfs"])

    @validator("allowed_directories", each_item=True)
    @classmethod
    def expand_allowed_dirs(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @validator("blocked_commands", pre=True)
    @classmethod
    def parse_blocked_commands(cls, v):
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v

    @validator("allowed_directories", pre=True)
    @classmethod
    def parse_allowed_directories(cls, v):
        if isinstance(v, str):
            return [d.strip() for d in v.split(",") if d.strip()]
        return v


class APISettings(BaseSettings):
    """API server settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_API_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = True
    workers: int = Field(default=1, ge=1)


class TelegramSettings(BaseSettings):
    """Telegram bot settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_TELEGRAM_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str | None = None
    allowed_users: list[str] = Field(default_factory=list)
    streaming: bool = True
    stream_edit_interval: float = 2.0
    preference_learning_interval: int = 10

    @validator("allowed_users", pre=True)
    @classmethod
    def parse_allowed_users(cls, v):
        """Parse allowed_users from env var (comma-separated or single value)."""
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        if isinstance(v, (int, float)):
            return [str(int(v))]
        return v


class VoiceSettings(BaseSettings):
    """Voice interface settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_VOICE_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = True
    stt_engine: Literal["whisper", "google", "local"] = "google"
    tts_engine: Literal["pyttsx3", "elevenlabs", "openai"] = "pyttsx3"
    whisper_model: str = "base"
    voice_reply: bool = False
    stt_language: str = "en-US"


class MonitoringSettings(BaseSettings):
    """Monitoring settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_MONITORING_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    metrics_enabled: bool = True
    metrics_port: int = 9090
    tracing_enabled: bool = False
    otel_exporter_endpoint: str | None = None


class DistributedSettings(BaseSettings):
    """Distributed mode settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_DISTRIBUTED_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    node_id: str = "node-1"
    coordinator_url: str | None = None


class SelfImprovementSettings(BaseSettings):
    """Self-improvement settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_SELF_IMPROVEMENT_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = True
    interval: int = 3600  # seconds
    code_generation_enabled: bool = False


class Settings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_prefix="LYA_",
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Workspace
    workspace_path: Path = Path("~/.lya")

    # Sub-settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api: APISettings = Field(default_factory=APISettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)
    self_improvement: SelfImprovementSettings = Field(default_factory=SelfImprovementSettings)

    @validator("workspace_path")
    @classmethod
    def expand_workspace_path(cls, v: Path) -> Path:
        path = v.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
