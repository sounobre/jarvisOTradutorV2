# --- ARQUIVO ATUALIZADO: services/import_pipeline.py ---
"""
Service que, após o upload, lê os dois EPUBs e persiste:
*** ATUALIZADO: 'ch_idx' agora é 'd["href"]' (a chave única) ***
*** 'title' agora é 'd["title"]' (o título "esperto") ***
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
        path_pt: str = None,  # <-- Agora é opcional
) -> Tuple[int, int | None]:
    # 1) Ler docs EN (Sempre)
    docs_en = read_epub_docs(path_en)

    # Ler docs PT (Só se existir)
    docs_pt = []
    if path_pt:
        docs_pt = read_epub_docs(path_pt)

    title_en = (docs_en[0]["title"] if docs_en and docs_en[0]["title"] else None)
    # Título PT só se tiver docs
    title_pt = (docs_pt[0]["title"] if docs_pt and docs_pt[0]["title"] else None)

    # 2) Criar Book EN
    res_en = await db.execute(
        insert(models.Book)
        .values(import_id=import_id, lang="en",
                title=title_en, author=None, spine_count=len(docs_en))
        .returning(models.Book.id)
    )
    book_en_id = res_en.scalar_one()

    # Criar Book PT (Só se existir)
    book_pt_id = None
    if path_pt:
        res_pt = await db.execute(
            insert(models.Book)
            .values(import_id=import_id, lang="pt",
                    title=title_pt, author=None, spine_count=len(docs_pt))
            .returning(models.Book.id)
        )
        book_pt_id = res_pt.scalar_one()

    # 3) Limpeza (Igual)
    await db.execute(delete(models.ChapterIndex).where(models.ChapterIndex.import_id == import_id))
    await db.execute(delete(models.ChapterText).where(models.ChapterText.import_id == import_id))

    # 4) Persistir EN (Igual)
    for i, d in enumerate(docs_en):
        await db.execute(
            insert(models.ChapterIndex).values(
                import_id=import_id, lang="en", ch_idx=d["href"],
                file_href=d["href"], title=d["title"],
                para_count=d["para_count"], char_count=d["char_count"]
            )
        )
        await db.execute(
            insert(models.ChapterText).values(
                import_id=import_id, lang="en", ch_idx=d["href"],
                text=d["text"], char_count=d["char_count"]
            )
        )

    # 5) Persistir PT (Só se existir path_pt)
    if path_pt:
        for j, d in enumerate(docs_pt):
            await db.execute(
                insert(models.ChapterIndex).values(
                    import_id=import_id, lang="pt", ch_idx=d["href"],
                    file_href=d["href"], title=d["title"],
                    para_count=d["para_count"], char_count=d["char_count"]
                )
            )
            await db.execute(
                insert(models.ChapterText).values(
                    import_id=import_id, lang="pt", ch_idx=d["href"],
                    text=d["text"], char_count=d["char_count"]
                )
            )

    await db.commit()
    return book_en_id, book_pt_id