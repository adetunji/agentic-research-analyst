from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.1  # Low for analytical consistency

    # --- Vector DB ---
    chroma_persist_dir: str = "data/chroma_db"
    collection_name: str = "research_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval ---
    rag_top_k: int = 5          # Docs retrieved on first attempt
    rag_retry_top_k: int = 10   # Wider net on retry

    # --- Self-RAG ---
    max_critique_retries: int = 2
    critique_pass_threshold: int = 7  # Score >= 7 → PASS

    # --- Ingestion ---
    chunk_size: int = 1000
    chunk_overlap: int = 200


settings = Settings()
