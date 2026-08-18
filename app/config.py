"""Central settings, loaded from environment variables (.env in local dev)."""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    mongo_uri: str | None = os.getenv("MONGO_URI")
    redis_url: str | None = os.getenv("REDIS_URL")


settings = Settings()
