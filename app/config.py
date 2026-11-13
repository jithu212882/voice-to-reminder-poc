import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    STT_PROVIDER: Literal["groq", "sarvam", "local_whisper", "deepgram"] = "groq"
    PARSER_PROVIDER: Literal["groq", "gemini", "local_rules"] = "groq"
    ENABLE_FAILOVER: bool = True

    GROQ_API_KEY: str = ""
    SARVAM_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""
    LOCAL_WHISPER_SIZE: str = "base"

    DATABASE_PATH: str = "reminders.db"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

