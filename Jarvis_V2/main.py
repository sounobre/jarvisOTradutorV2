# --- FILE: main.py ---
"""
Ponto de entrada do FastAPI.

Conceitos:
- `FastAPI(...)`: cria a aplicação.
- Evento `@app.on_event("startup")`: roda no início — aqui criamos as tabelas automaticamente (dev).
- `include_router(...)`: registra rotas no app.
"""
import os
import sys

from api import epub_import, chapter_window_map, epub_bulk_import, translation_api

try:
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import logging

from fastapi import FastAPI

from core.logging_setup import setup_logging
from db.models import Base
from db.session import engine


from sqlalchemy import text



setup_logging()

# Configuração simples de logging (INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

# Instancia o app com título e versão (aparecem no Swagger)
app = FastAPI(title="JarvisTradutor v2 — EPUB Mapping + LLM Alignment", version="0.2.0")


@app.on_event("startup")
async def startup():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        import traceback
        print("ERRO AO CONECTAR/CRIAR TABELAS:", e)
        traceback.print_exc()
        raise


@app.get("/health/db")
async def health_db():
    try:
        async with engine.begin() as conn:
            await conn.execute(text("select 1"))
        return {"db": "ok"}
    except Exception as e:
        return {"db": "error", "detail": str(e)}


# Registra o router principal (todas as rotas /epub)

app.include_router(epub_import.router)

app.include_router(chapter_window_map.router)  # <-- ADICIONE ESTA LINHA
app.include_router(epub_bulk_import.router)
app.include_router(translation_api.router)