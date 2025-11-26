# --- ARQUIVO ATUALIZADO: api/epub_bulk_import.py ---
"""
Este é o "Botão de Automação" COM LOGS NO BANCO + PROTEÇÃO CONTRA DUPLICATAS.
"""
from __future__ import annotations
import logging
import os
import shutil
import uuid
import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel

from db.session import AsyncSessionLocal
# ADICIONEI 'select' AQUI
from sqlalchemy import insert, select
from db import models
from services.import_pipeline import persist_books_and_chapters

router = APIRouter(prefix="/epub/bulk", tags=["epub-bulk-import"])
logger = logging.getLogger(__name__)

UPLOAD_BASE = "data/uploads"


# --- Helper de Log no Banco ---
async def _log_to_db(batch_id: str, row_idx: int, title: str, status: str, message: str):
    if status == 'ERROR':
        logger.error(f"[BulkImport] {title}: {message}")
    elif status == 'SKIPPED':
        logger.warning(f"[BulkImport] {title}: {message}")
    else:
        logger.info(f"[BulkImport] {title}: {message}")

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                insert(models.TmBulkImportLog).values(
                    batch_id=batch_id,
                    row_index=row_idx,
                    book_title=title[:255] if title else "Unknown",
                    status=status,
                    message=str(message)
                )
            )
            await session.commit()
        except Exception as e:
            logger.error(f"FATAL: Falha ao salvar log no banco: {e}")


# --- O "Miolo" (A Lógica em Background) ---

async def _run_bulk_import(spreadsheet_path: str, batch_id: str):
    await _log_to_db(batch_id, 0, "SYSTEM", "INFO", f"Iniciando processamento: {spreadsheet_path}")

    try:
        if spreadsheet_path.endswith('.xlsx'):
            df = pd.read_excel(spreadsheet_path)
        else:
            df = pd.read_csv(spreadsheet_path, sep=None, engine='python')
    except Exception as e:
        await _log_to_db(batch_id, 0, "SYSTEM", "ERROR", f"Falha fatal ao ler arquivo: {e}")
        return

    total_rows = len(df)
    await _log_to_db(batch_id, 0, "SYSTEM", "INFO", f"Planilha carregada. {total_rows} linhas.")

    # Loop principal
    for index, row in df.iterrows():
        row_num = index + 2

        name = str(row.get('TituloOriginalEN', '')).strip(' "')
        en_path = str(row.get('caminhoIngles', '')).strip(' "')
        pt_path = str(row.get('CaminhoPortugues', '')).strip(' "')

        # 1. Validação Básica
        if not name or not en_path:
            await _log_to_db(batch_id, row_num, name or "SEM TITULO", "SKIPPED", "Dados faltando.")
            continue

        # 2. Validação de Arquivos
        if not os.path.exists(en_path):
            await _log_to_db(batch_id, row_num, name, "ERROR", f"Arquivo EN não encontrado: {en_path}")
            continue
        if pt_path and pt_path.lower() != 'nan':
            if not os.path.exists(pt_path):
                await _log_to_db(batch_id, row_num, name, "ERROR", f"Arquivo PT não encontrado: {pt_path}")
                continue
        else:
            pt_path = None

        async with AsyncSessionLocal() as session:
            try:
                # --- TRAVA DE DUPLICIDADE (NOVO) ---
                # Verifica se já existe um import com esse NOME EXATO
                stmt_check = select(models.Import.id).where(models.Import.name == name)
                existing_id = await session.scalar(stmt_check)

                if existing_id:
                    # Se já existe, loga como SKIPPED e PULA
                    await _log_to_db(batch_id, row_num, name, "SKIPPED",
                                     f"Livro já importado anteriormente (ID: {existing_id}).")
                    continue
                # -----------------------------------

                # Cria o registro na tm_import
                res = await session.execute(
                    insert(models.Import)
                    .values(name=name, file_en=en_path, file_pt=pt_path)
                    .returning(models.Import.id)
                )
                import_id = res.scalar_one()
                await session.commit()

                await persist_books_and_chapters(
                    db=session, import_id=import_id,
                    path_en=en_path, path_pt=pt_path
                )

                await _log_to_db(batch_id, row_num, name, "SUCCESS", f"Importado com sucesso. ID={import_id}")

            except Exception as e:
                error_msg = str(e)
                await _log_to_db(batch_id, row_num, name, "ERROR", f"Falha no processamento: {error_msg}")

    await _log_to_db(batch_id, 0, "SYSTEM", "INFO", "Job Finalizado.")


# --- O Endpoint ---
@router.post("/import-spreadsheet")
async def create_bulk_import(
        background_tasks: BackgroundTasks,
        spreadsheet: UploadFile = File(...),
):
    try:
        batch_id = str(uuid.uuid4())[:8]
        temp_dir = os.path.join(UPLOAD_BASE, "spreadsheets")
        os.makedirs(temp_dir, exist_ok=True)
        file_name = f"{batch_id}_{spreadsheet.filename}"
        spreadsheet_path = os.path.join(temp_dir, file_name)
        with open(spreadsheet_path, "wb") as f:
            shutil.copyfileobj(spreadsheet.file, f)
        spreadsheet_path = spreadsheet_path.replace("\\", "/")

        background_tasks.add_task(_run_bulk_import, spreadsheet_path, batch_id)

        return {
            "status": "bulk_import_scheduled",
            "batch_id": batch_id,
            "message": "Importação iniciada. Duplicatas serão puladas automaticamente."
        }
    except Exception as e:
        logger.exception(f"Erro no upload: {e}")
        raise HTTPException(status_code=500, detail=f"Erro: {e}")