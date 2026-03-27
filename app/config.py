from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # auth
    gateway_api_key: str

    # provider keys
    anthropic_api_key: str
    gemini_api_key: str
    ollama_base_url: str = "http://localhost:11434"

    # model routing table: model name → provider name
    model_routes: dict[str, str] = {
        "claude-opus-4-6":   "anthropic",
        "claude-sonnet-4-6": "anthropic",
        "gemini-2.0-flash":   "gemini",
        "gemini-2.5-flash":   "gemini",
        "gemini-2.5-pro":     "gemini",
        "llama3.2":          "ollama",
        "mistral":           "ollama",
        "qwen3.5":           "ollama",
    }

    model_aliases: dict[str, str] = {
        "fast":  "gemini-2.0-flash",
        "smart": "claude-opus-4-6",
        "local": "llama3.2",
    }

    # per-provider timeouts in seconds
    anthropic_timeout: float = 60.0
    gemini_timeout: float = 60.0
    ollama_timeout: float = 120.0

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
