# --- UPDATED FILE: api/epub_import.py ---
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from db.session import get_db
from db import models
from services.import_pipeline import persist_books_and_chapters
import uuid, os, shutil
import logging

router = APIRouter(prefix="/epub", tags=["epub"])
logger = logging.getLogger(__name__)

UPLOAD_BASE = "data/uploads"


@router.post("/import")
async def create_import(
        name: str = Form(...),
        file_en: UploadFile = File(...),
        file_pt: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
):
    """
    Fluxo:
    1) Salva os 2 arquivos em data/uploads/<uuid>/
    2) Cria tm_import
    3) Lê os EPUBs e persiste:
       - tm_book (EN/PT)
       - tm_chapter_index (índice de capítulos)
       - tm_chapter_text (texto por capítulo)
    4) Retorna IDs e contagens
    """
    # 1) salvar arquivos fisicamente
    bucket = str(uuid.uuid4())
    outdir = os.path.join(UPLOAD_BASE, bucket)
    os.makedirs(outdir, exist_ok=True)

    en_path = os.path.join(outdir, f"en_{file_en.filename}")
    pt_path = os.path.join(outdir, f"pt_{file_pt.filename}")

    with open(en_path, "wb") as f:
        shutil.copyfileobj(file_en.file, f)
    with open(pt_path, "wb") as f:
        shutil.copyfileobj(file_pt.file, f)

    # Normaliza separador Windows -> Unix
    en_path = en_path.replace("\\", "/")
    pt_path = pt_path.replace("\\", "/")

    # 2) cria tm_import
    res = await db.execute(
        insert(models.Import)
        .values(name=name, file_en=en_path, file_pt=pt_path)
        .returning(models.Import.id)
    )
    import_id = res.scalar_one()
    await db.commit()

    logger.info("[import] salvo | id=%s | name=%s", import_id, name)

    # 3) persiste livros + capítulos
    try:
        book_en_id, book_pt_id = await persist_books_and_chapters(
            db=db, import_id=import_id, path_en=en_path, path_pt=pt_path
        )
    except Exception as e:
        # se der erro, registra e retorna 500 amigável
        logger.exception("[import] falha ao persistir livros/capitulos | import_id=%s | erro=%s", import_id, e)
        raise HTTPException(status_code=500, detail=f"import ok, mas falhou ao persistir livros/capitulos: {e}")

    # 4) resposta “rica”
    return {
        "id": import_id,
        "name": name,
        "file_en": en_path,
        "file_pt": pt_path,
        "books": {
            "en": {"book_id": book_en_id},
            "pt": {"book_id": book_pt_id},
        },
        "message": "Import + livros + capítulos persistidos com sucesso.",
    }
