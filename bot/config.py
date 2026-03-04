"""Bot configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BotConfig:
    """Telegram bot settings."""
    token: str
    admin_id: int


@dataclass
class AssemblyAIConfig:
    """AssemblyAI API settings."""
    api_key: str


# Default fallback chain of free models
DEFAULT_MODELS = [
    "google/gemma-3-12b-it:free",
    "meta-llama/llama-4-scout:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-27b-it:free",
]


@dataclass
class OpenRouterConfig:
    """OpenRouter API settings."""
    api_key: str
    models: list[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())


@dataclass
class AppConfig:
    """Application settings."""
    max_audio_duration: int  # seconds
    max_file_size: int       # bytes
    log_level: str


@dataclass
class Config:
    """Root configuration container."""
    bot: BotConfig
    assemblyai: AssemblyAIConfig
    openrouter: OpenRouterConfig
    app: AppConfig


def _parse_models(models_str: str) -> list[str]:
    """Parse comma-separated model names, fall back to defaults if empty."""
    if not models_str.strip():
        return DEFAULT_MODELS.copy()
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    return models if models else DEFAULT_MODELS.copy()


def load_config() -> Config:
    """Load configuration from environment variables.

    Raises:
        ValueError: If required environment variables are not set.
    """
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise ValueError("BOT_TOKEN environment variable is required")

    admin_id_str = os.getenv("ADMIN_ID")
    if not admin_id_str:
        raise ValueError("ADMIN_ID environment variable is required")

    assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    return Config(
        bot=BotConfig(
            token=bot_token,
            admin_id=int(admin_id_str),
        ),
        assemblyai=AssemblyAIConfig(
            api_key=assemblyai_key,
        ),
        openrouter=OpenRouterConfig(
            api_key=openrouter_key,
            models=_parse_models(os.getenv("OPENROUTER_MODELS", "")),
        ),
        app=AppConfig(
            max_audio_duration=int(os.getenv("MAX_AUDIO_DURATION", "7200")),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", "20971520")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        ),
    )
