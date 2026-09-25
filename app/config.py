"""Central settings, loaded from environment variables (.env in local dev)."""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # `or None` on each of these so an explicitly-empty env var (e.g. a
    # subprocess env that sets MONGO_URI="" to force the mock fallback --
    # see app/mcp_client.py) is treated the same as an unset one, rather
    # than being passed to the real client and failing to connect.
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    mongo_uri: str | None = os.getenv("MONGO_URI") or None
    redis_url: str | None = os.getenv("REDIS_URL") or None
    neo4j_uri: str | None = os.getenv("NEO4J_URI") or None
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str | None = os.getenv("NEO4J_PASSWORD")
    # Optional -- if unset, memory_store.py persists to ./chroma_data in the
    # project root. No mock/real split needed here the way mongo_uri/redis_url/
    # neo4j_uri have one: Chroma is local either way, so this is just a path
    # override, not a "use the real thing" switch.
    chroma_persist_dir: str | None = os.getenv("CHROMA_PERSIST_DIR") or None
    # Explicit override for classification_agent.py's provider auto-detection
    # ("local" / "anthropic" / "heuristic"). Unset means auto-detect: prefer
    # local Qwen via Ollama whenever reachable, then real Anthropic, then the
    # heuristic fallback. Tests pin this directly via monkeypatch rather than
    # relying on whatever happens to be running on the machine executing them
    # -- see tests/conftest.py's _force_heuristic_classification fixture.
    llm_provider: str | None = os.getenv("LLM_PROVIDER") or None
    # A local model served via Ollama (e.g. an Unsloth-tuned Qwen model
    # imported into Ollama) -- preferred automatically over Anthropic
    # whenever reachable; see classification_agent.py's _resolve_provider().
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str | None = os.getenv("OLLAMA_MODEL") or None


settings = Settings()
