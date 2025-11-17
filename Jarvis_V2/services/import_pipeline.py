# --- NEW FILE: services/import_pipeline.py ---
"""
Service que, após o upload, lê os dois EPUBs e persiste:
- tm_book (um por idioma)
- tm_chapter_index (file_href, título, contagens)
- tm_chapter_text (texto consolidado de cada doc/capítulo)

Explicando a sintaxe:
- `async def` define função assíncrona (roda com await).
- `await db.execute(...)` envia SQL de forma não bloqueante.
- `insert(Model).values(...).returning(Model.id)` insere e volta com o id.
"""
from __future__ import annotations

import logging
from typing import Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert, delete
from db import models
from utils.epub import read_epub_docs

logger = logging.getLogger(__name__)

async def persist_books_and_chapters(
    db: AsyncSession,
    import_id: int,
    path_en: str,
    path_pt: str,
) -> Tuple[int, int]:
    """
    Retorna (book_en_id, book_pt_id).
    """
    # 1) Ler docs dos EPUBs (cada item tem href/title/text/para_count/char_count)
    docs_en = read_epub_docs(path_en)
    docs_pt = read_epub_docs(path_pt)

    # Heurísticas simples para title/author se não vierem do EPUB (melhorar depois).
    title_en = (docs_en[0]["title"] if docs_en and docs_en[0]["title"] else None)
    title_pt = (docs_pt[0]["title"] if docs_pt and docs_pt[0]["title"] else None)
    author_en = None
    author_pt = None

    # 2) Criar registros tm_book (um por idioma)
    res_en = await db.execute(
        insert(models.Book)
        .values(import_id=import_id, lang="en",
                title=title_en, author=author_en, spine_count=len(docs_en))
        .returning(models.Book.id)
    )
    book_en_id = res_en.scalar_one()

    res_pt = await db.execute(
        insert(models.Book)
        .values(import_id=import_id, lang="pt",
                title=title_pt, author=author_pt, spine_count=len(docs_pt))
        .returning(models.Book.id)
    )
    book_pt_id = res_pt.scalar_one()

    # 3) Regravar índice/texto idempotente (apaga antes se reprocessar)
    await db.execute(delete(models.ChapterIndex).where(models.ChapterIndex.import_id == import_id))
    await db.execute(delete(models.ChapterText).where(models.ChapterText.import_id == import_id))

    # 4) Persistir índice e texto EN
    for i, d in enumerate(docs_en):
        await db.execute(
            insert(models.ChapterIndex).values(
                import_id=import_id, lang="en", ch_idx=d["title"],
                file_href=d["href"], title=d["title"],
                para_count=d["para_count"], char_count=d["char_count"]
            )
        )
        logger.info("import_id=%s, lang=%s, ch_idx=%s, char_count=%s",import_id,"en", d["title"], d["char_count"])
        await db.execute(
            insert(models.ChapterText).values(
                import_id=import_id, lang="en", ch_idx=d["title"], text=d["text"], char_count=d["char_count"]
            )
        )

    # 5) Persistir índice e texto PT
    for j, d in enumerate(docs_pt):
        await db.execute(
            insert(models.ChapterIndex).values(
                import_id=import_id, lang="pt", ch_idx=d["title"],
                file_href=d["href"], title=d["title"],
                para_count=d["para_count"], char_count=d["char_count"]
            )
        )
        logger.info("import_id=%s, lang=%s, ch_idx=%s, file_href=%s, title=%s,para_count=%s, char_count=%s",import_id,"pt",d["title"],d["href"],d["title"],d["para_count"],d["char_count"])
        await db.execute(
            insert(models.ChapterText).values(
                import_id=import_id, lang="pt", ch_idx=d["title"], text=d["text"], char_count=d["char_count"]
            )
        )

    # 6) Commit em lote
    await db.commit()
    return book_en_id, book_pt_id
