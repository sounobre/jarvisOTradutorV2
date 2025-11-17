# --- FILE: db/session.py ---
"""
Cria a **conexão assíncrona** com o PostgreSQL via SQLAlchemy 2.0.

Conceitos (Python + FastAPI + SQLAlchemy):
- `async def`: define função assíncrona — permite `await` e IO sem travar o servidor.
- `create_async_engine(...)`: cria um *engine* (conexão de baixo nível) para o banco.
- `sessionmaker(...)`: fábrica de **sessões** (objetos que fazem as queries/commits).
- `yield` em `get_db()`: padrão de **dependência** no FastAPI que abre/fecha a sessão por requisição.
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings

# Cria o engine assíncrono usando a URL do .env
engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)

# Cria a fábrica de sessões assíncronas. `expire_on_commit=False` mantém objetos usáveis após commit.
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

async def get_db():
    """Dependência do FastAPI que injeta `db: AsyncSession` nos endpoints.
    O `yield` devolve a sessão para o endpoint e, ao terminar, o FastAPI cuida de fechar a sessão.
    """
    async with AsyncSessionLocal() as session:
        yield session