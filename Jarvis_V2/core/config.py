# --- UPDATED FILE: core/config.py ---
import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:123456@127.0.0.1:5432/jarvis_v2")

    SRC_LANG: str = os.getenv("SRC_LANG", "en")
    TGT_LANG: str = os.getenv("TGT_LANG", "pt")

    # Embeddings local (sentence-transformers)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
    EMBEDDINGS_BATCH_SIZE: int = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "8"))

    # Limiar/heurísticas
    LENGTH_RATIO_PENALTY: float = float(os.getenv("LENGTH_RATIO_PENALTY", "0.10"))
    MAP_SIM_WARN_MIN: float = float(os.getenv("MAP_SIM_WARN_MIN", "0.60"))
    MAP_LEN_RATIO_WARN: float = float(os.getenv("MAP_LEN_RATIO_WARN", "0.50"))
    MAP_CHAR_RATIO_WARN: float = float(os.getenv("MAP_CHAR_RATIO_WARN", "0.50"))

    # OLLAMA
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    OLLAMA_KEEP_ALIVE_LLM: str = os.getenv("OLLAMA_KEEP_ALIVE_LLM", "10m")
    OLLAMA_KEEP_ALIVE_EMB: str = os.getenv("OLLAMA_KEEP_ALIVE_EMB", "0s")

    # Se 0 => sempre ST local; se 1 => tenta Ollama primeiro
    USE_OLLAMA_EMBEDDINGS: bool = bool(int(os.getenv("USE_OLLAMA_EMBEDDINGS", "0")))
    AUTO_UNLOAD_LLM_AFTER_ENDPOINT: bool = bool(int(os.getenv("AUTO_UNLOAD_LLM_AFTER_ENDPOINT", "1")))
    AUTO_UNLOAD_EMB_AFTER_CALL: bool = bool(int(os.getenv("AUTO_UNLOAD_EMB_AFTER_CALL", "1")))
    AUTO_UNLOAD_EMB_BEFORE_LLM: bool = bool(int(os.getenv("AUTO_UNLOAD_EMB_BEFORE_LLM", "1")))
    LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", "8"))
    OLLAMA_LLM_FALLBACKS: str = os.getenv("OLLAMA_LLM_FALLBACKS", "qwen2.5:14b-instruct-q4_K_M")
    LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "1024"))

    # Logs com/em sem emoji (evita Unicode no Windows)
    LOG_USE_EMOJI: bool = bool(int(os.getenv("LOG_USE_EMOJI", "0")))

    # Logging do LLM (controle do que imprimir)
    LOG_LLM_REQUEST_MAX_CHARS: int = int(os.getenv("LOG_LLM_REQUEST_MAX_CHARS", "800"))  # trecho EN logado
    LOG_LLM_CAND_EXCERPT_CHARS: int = int(
        os.getenv("LOG_LLM_CAND_EXCERPT_CHARS", "220"))  # excerpt de cada candidato PT
    LOG_LLM_RESPONSE_MAX_CHARS: int = int(os.getenv("LOG_LLM_RESPONSE_MAX_CHARS", "800"))  # resposta crua do LLM
    LOG_LEVEL_LLM: str = os.getenv("LOG_LEVEL_LLM", "INFO")  # INFO/DEBUG

    # Verificação de evidências (gates)
    EVIDENCE_MIN_QUOTES: int = int(os.getenv("EVIDENCE_MIN_QUOTES", "2"))
    ANCHOR_MIN_HITS: int = int(os.getenv("ANCHOR_MIN_HITS", "2"))
    QUOTE_MIN_CHARS: int = int(os.getenv("QUOTE_MIN_CHARS", "20"))

    CACHE_TTL_TAGS_SECONDS: int = int(os.getenv("CACHE_TTL_TAGS_SECONDS", "3600"))
    CACHE_TTL_SHOW_SECONDS: int = int(os.getenv("CACHE_TTL_SHOW_SECONDS", "43200"))
    OLLAMA_MAX_CONCURRENCY: int = int(os.getenv("OLLAMA_MAX_CONCURRENCY", "1"))


settings = Settings()
