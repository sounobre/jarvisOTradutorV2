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
        path_pt: str,
) -> Tuple[int, int]:
    # 1) Ler docs (agora 'title' é o título "esperto")
    docs_en = read_epub_docs(path_en)
    docs_pt = read_epub_docs(path_pt)

    # (Lógica de Book fica igual)
    title_en = (docs_en[0]["title"] if docs_en and docs_en[0]["title"] else None)
    title_pt = (docs_pt[0]["title"] if docs_pt and docs_pt[0]["title"] else None)
    res_en = await db.execute(
        insert(models.Book)
        .values(import_id=import_id, lang="en",
                title=title_en, author=None, spine_count=len(docs_en))
        .returning(models.Book.id)
    )
    book_en_id = res_en.scalar_one()
    res_pt = await db.execute(
        insert(models.Book)
        .values(import_id=import_id, lang="pt",
                title=title_pt, author=None, spine_count=len(docs_pt))
        .returning(models.Book.id)
    )
    book_pt_id = res_pt.scalar_one()

    # 3) Regravar (igual)
    await db.execute(delete(models.ChapterIndex).where(models.ChapterIndex.import_id == import_id))
    await db.execute(delete(models.ChapterText).where(models.ChapterText.import_id == import_id))

    # --- 4) Persistir EN (COM A MUDANÇA) ---
    for i, d in enumerate(docs_en):
        await db.execute(
            insert(models.ChapterIndex).values(
                import_id=import_id, lang="en",
                ch_idx=d["href"],  # <--- MUDANÇA (A CHAVE É O HREF)
                file_href=d["href"],
                title=d["title"],  # <--- MUDANÇA (O TÍTULO É O TÍTULO)
                para_count=d["para_count"],
                char_count=d["char_count"]
            )
        )
        await db.execute(
            insert(models.ChapterText).values(
                import_id=import_id, lang="en",
                ch_idx=d["href"],  # <--- MUDANÇA (A CHAVE É O HREF)
                text=d["text"],
                char_count=d["char_count"]
            )
        )

    # --- 5) Persistir PT (COM A MUDANÇA) ---
    for j, d in enumerate(docs_pt):
        await db.execute(
            insert(models.ChapterIndex).values(
                import_id=import_id, lang="pt",
                ch_idx=d["href"],  # <--- MUDANÇA (A CHAVE É O HREF)
                file_href=d["href"],
                title=d["title"],  # <--- MUDANÇA (O TÍTULO É O TÍTULO)
                para_count=d["para_count"],
                char_count=d["char_count"]
            )
        )
        await db.execute(
            insert(models.ChapterText).values(
                import_id=import_id, lang="pt",
                ch_idx=d["href"],  # <--- MUDANÇA (A CHAVE É O HREF)
                text=d["text"],
                char_count=d["char_count"]
            )
        )

    await db.commit()
    return book_en_id, book_pt_id