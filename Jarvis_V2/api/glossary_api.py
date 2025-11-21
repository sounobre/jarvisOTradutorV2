# --- NOVO ARQUIVO: api/glossary_api.py ---
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, delete
from typing import List

from db.session import get_db
from db import models

router = APIRouter(prefix="/epub/glossary", tags=["epub-glossary"])


class GlossaryItem(BaseModel):
    term_source: str
    term_target: str


class BulkGlossaryRequest(BaseModel):
    import_id: int
    terms: List[GlossaryItem]


@router.post("/add-terms")
async def add_glossary_terms(
        req: BulkGlossaryRequest,
        db: AsyncSession = Depends(get_db)
):
    """
    Adiciona uma lista de termos ao glossário de um livro.
    Ex: "High Lord" -> "Grão-Senhor"
    """
    try:
        # Prepara os dados para inserção
        values = [
            {
                "import_id": req.import_id,
                "term_source": item.term_source,
                "term_target": item.term_target
            }
            for item in req.terms
        ]

        # Usa UPSERT (se já existir, atualiza a tradução)
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(models.TmGlossary).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=['import_id', 'term_source'],
            set_={'term_target': stmt.excluded.term_target}
        )

        await db.execute(stmt)
        await db.commit()

        return {"message": f"{len(values)} termos adicionados/atualizados no glossário."}

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_glossary(import_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(models.TmGlossary).where(models.TmGlossary.import_id == import_id)
    result = (await db.execute(stmt)).scalars().all()
    return result