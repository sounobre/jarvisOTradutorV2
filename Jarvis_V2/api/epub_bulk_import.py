# --- NOVO ARQUIVO: api/epub_bulk_import.py ---
"""
Este é o "Botão de Automação".

Ele fornece um endpoint para importar em LOTE a partir
de uma planilha (CSV/Excel) que contém os caminhos locais
para os arquivos EPUB.
"""
from __future__ import annotations
import logging
import os
import shutil
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List

from db.session import get_db, AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from db import models

# Importa a "lógica de importação" que JÁ EXISTE
from services.import_pipeline import persist_books_and_chapters

router = APIRouter(prefix="/epub/bulk", tags=["epub-bulk-import"])
logger = logging.getLogger(__name__)

UPLOAD_BASE = "data/uploads"  # Onde vamos salvar a planilha


# --- Modelos de Resposta ---
class BulkImportResult(BaseModel):
    import_id: int
    name: str
    status: str
    error: Optional[str] = None


class BulkImportResponse(BaseModel):
    spreadsheet_path: str
    total_rows: int
    imports_processed: List[BulkImportResult]


# --- O "Miolo" (A Lógica em Background) ---

async def _run_bulk_import(spreadsheet_path: str):
    """
    Esta é a função que roda em background.
    Ela lê a planilha e processa cada linha.
    """
    logger.info(f"[BulkImport] Job em background iniciado. Lendo: {spreadsheet_path}")

    try:
        # Detecta se é Excel ou CSV e lê
        if spreadsheet_path.endswith('.xlsx'):
            df = pd.read_excel(spreadsheet_path)
        else:
            # Tenta detectar o separador (vírgula ou ponto-e-vírgula)
            df = pd.read_csv(spreadsheet_path, sep=None, engine='python')

    except Exception as e:
        logger.error(f"[BulkImport] Falha ao ler a planilha: {e}")
        return  # Aborta o job

    total_rows = len(df)
    logger.info(f"[BulkImport] Planilha lida. {total_rows} linhas encontradas.")

    # Loop principal (linha por linha)
    for index, row in df.iterrows():

        # --- A CORREÇÃO ESTÁ AQUI ---
        # Converte para string (para o caso de a célula estar vazia/nula)
        # e usa .strip(' "') para remover espaços E aspas
        # do começo e do fim da string.

        name = str(row.get('TituloOriginalEN', '')).strip(' "')
        en_path = str(row.get('caminhoIngles', '')).strip(' "')
        pt_path = str(row.get('CaminhoPortugues', '')).strip(' "')

        # --- FIM DA CORREÇÃO ---

        # --- Validação da Linha ---
        if not all([name, en_path, pt_path]):
            logger.warning(f"[BulkImport] Pulando linha {index + 2}: dados faltando (Título, EN ou PT).")
            continue

        # Checa se os arquivos *existem* no caminho local (que o servidor pode ler)
        if not os.path.exists(en_path):
            logger.error(f"[BulkImport] ERRO na linha {index + 2}: Arquivo EN não encontrado em: {en_path}")
            continue
        if not os.path.exists(pt_path):
            logger.error(f"[BulkImport] ERRO na linha {index + 2}: Arquivo PT não encontrado em: {pt_path}")
            continue

        logger.info(f"[BulkImport] Processando {index + 1}/{total_rows}: {name}")

        # --- Lógica (copiada do seu /epub/import) ---
        async with AsyncSessionLocal() as session:
            try:
                # 1. Cria o registro tm_import
                # (Não salvamos os arquivos, só apontamos para os caminhos que você deu)
                res = await session.execute(
                    insert(models.Import)
                    .values(name=name, file_en=en_path, file_pt=pt_path)
                    .returning(models.Import.id)
                )
                import_id = res.scalar_one()
                await session.commit()

                # 2. Persiste os livros e capítulos (a lógica que já existe)
                await persist_books_and_chapters(
                    db=session, import_id=import_id,
                    path_en=en_path, path_pt=pt_path
                )

                logger.info(f"[BulkImport] SUCESSO: import_id={import_id} | name={name}")

            except Exception as e:
                await session.rollback()
                logger.exception(f"[BulkImport] FALHA ao persistir '{name}': {e}")
                # (Continua para a próxima linha)

    logger.info(f"[BulkImport] Job em background concluído.")


# --- O Endpoint (O Botão de Upload) ---
@router.post("/import-spreadsheet")
async def create_bulk_import(
        background_tasks: BackgroundTasks,
        spreadsheet: UploadFile = File(...),
):
    """
    Endpoint para "automatização".
    1. Recebe UMA planilha (Excel ou CSV).
    2. Salva a planilha em data/uploads.
    3. Dispara um job em BACKGROUND para ler a planilha
       e importar cada par de livros (usando os caminhos locais).
    4. Retorna uma resposta imediata.
    """
    try:
        # 1. Salvar a planilha
        temp_dir = os.path.join(UPLOAD_BASE, "spreadsheets")
        os.makedirs(temp_dir, exist_ok=True)

        # Garante um nome de arquivo único
        file_name = f"{uuid.uuid4()}_{spreadsheet.filename}"
        spreadsheet_path = os.path.join(temp_dir, file_name)

        with open(spreadsheet_path, "wb") as f:
            shutil.copyfileobj(spreadsheet.file, f)

        spreadsheet_path = spreadsheet_path.replace("\\", "/")

        # 2. Agendar o job em background
        background_tasks.add_task(_run_bulk_import, spreadsheet_path=spreadsheet_path)

        logger.info(f"[BulkImport] Job agendado. Planilha salva em: {spreadsheet_path}")

        # 3. Resposta imediata
        return {
            "status": "bulk_import_scheduled",
            "spreadsheet_path": spreadsheet_path,
            "message": "Upload da planilha OK. Importação em lote iniciada em background."
        }
    except Exception as e:
        logger.exception(f"Falha ao fazer upload da planilha: {e}")
        raise HTTPException(status_code=500, detail=f"Falha ao salvar planilha: {e}")