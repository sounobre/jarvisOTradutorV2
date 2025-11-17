# --- ARQUIVO ATUALIZADO (O "PAINEL DE CONTROLE"): api/chapter_window_map.py ---
"""
Este é o "Painel de Controle" da API (Arquitetura "Embeddings-Only").
Agora inclui o botão "Processar Tudo".
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import TmWinMapping

# Importa as NOVAS funções do "Motor"
from services.corpus_builder_service import (
    schedule_macro_map,
    run_sentence_alignment,
    export_corpus_to_tsv,
    get_corpus_status,
    run_alignment_for_all_pending, run_corpus_validation  # <-- A NOVA FUNÇÃO
)

router = APIRouter(prefix="/epub/corpus_v2", tags=["corpus-pipeline-v2"])
logger = logging.getLogger(__name__)


# -------- Modelos de Request (Payloads) --------
class JobRequest(BaseModel):
    import_id: int = Field(..., description="ID do livro em tm_import")


class AlignOneChapterRequest(BaseModel):
    import_id: int = Field(..., description="ID do livro em tm_import")
    ch_src: str = Field(..., description="O TÍTULO do capítulo EN para processar (ex: 'Chapter 1')")


# -------- Endpoints (Os Botões) --------

@router.post("/1-schedule-macro-map")
async def schedule_macro_map_endpoint(
        req: JobRequest,
        background_tasks: BackgroundTasks,
):
    """
    Botão 1: Roda a Fase 1 (Mapa Macro).
    Salva na 'TmWinMapping' com status 'pending'.
    """
    try:
        background_tasks.add_task(schedule_macro_map, req.import_id)
        logger.info(f"Endpoint 1: Job de Agendamento (Macro-Map) iniciado para import_id={req.import_id}.")
        return {
            "status": "macro_map_job_scheduled",
            "import_id": req.import_id,
            "message": "Fase 1 (Mapeamento de Capítulos) iniciada em background."
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar job 1 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar: {e}")


@router.post("/2-align-next-pending-chapter")
async def align_next_pending_chapter_endpoint(
        req: JobRequest,
        db: AsyncSession = Depends(get_db)
):
    """
    Botão 2 (Automático): Processa o PRÓXIMO capítulo 'pending'.
    Este endpoint ESPERA o alinhamento (15-30s) terminar.
    """
    try:
        # 1. Encontra o próximo capítulo "pending"
        stmt = select(TmWinMapping.ch_src).where(
            TmWinMapping.import_id == req.import_id,
            TmWinMapping.micro_status == "pending"
        ).order_by(TmWinMapping.ch_src).limit(1)

        proximo_capitulo = (await db.execute(stmt)).scalar_one_or_none()

        if not proximo_capitulo:
            logger.info(f"Endpoint 2: Nenhum capítulo 'pending' encontrado para import_id={req.import_id}.")
            return {"message": "Nenhum capítulo pendente encontrado. Alinhamento concluído!",
                    "import_id": req.import_id}

        ch_src_str = str(proximo_capitulo)

        # 2. CHAMA E ESPERA
        logger.info(f"Endpoint 2: INICIANDO alinhamento síncrono para: '{ch_src_str}'.")
        result_json = await run_sentence_alignment(req.import_id, ch_src_str)
        logger.info(f"Endpoint 2: CONCLUÍDO alinhamento para: '{ch_src_str}'.")

        if "error" in result_json:
            raise HTTPException(status_code=500, detail=result_json)
        return result_json

    except Exception as e:
        logger.exception(f"Falha ao agendar job 2 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar: {e}")


# --- NOVO ENDPOINT (O que você pediu) ---
@router.post("/2c-align-all-pending")
async def align_all_pending_endpoint(
        req: JobRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
):
    """
    Botão 2-C (Processar Tudo): Processa TODOS os capítulos 'pending'.

    Este endpoint é em BACKGROUND (não espera), pois pode
    levar muitos minutos (ex: 50 caps * 30s = 25 min).
    """
    try:
        # 1. Checagem rápida para ver se há trabalho a fazer
        stmt = select(TmWinMapping.ch_src).where(
            TmWinMapping.import_id == req.import_id,
            TmWinMapping.micro_status == "pending"
        ).limit(1)

        if not (await db.execute(stmt)).scalar_one_or_none():
            logger.info(f"Endpoint 2C: Nenhum capítulo 'pending' encontrado.")
            return {"message": "Nenhum capítulo pendente encontrado."}

        # 2. Agenda o "loop" completo em background
        background_tasks.add_task(run_alignment_for_all_pending, req.import_id)

        logger.info(f"Endpoint 2C: Job de Alinhamento COMPLETO (Fase 2) iniciado para import_id={req.import_id}.")
        return {
            "status": "full_alignment_job_scheduled",
            "import_id": req.import_id,
            "message": "Fase 2 (Loop de Alinhamento de Todos os Capítulos) iniciada em background."
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar job 2C para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar: {e}")


# --- FIM DO NOVO ENDPOINT ---

@router.post("/2b-align-specific-chapter")
async def align_specific_chapter_endpoint(
        req: AlignOneChapterRequest,
):
    """
    Botão 2 (Manual): Processa um capítulo específico.
    Também espera o trabalho terminar (síncrono).
    """
    try:
        logger.info(f"Endpoint 2b: INICIANDO alinhamento síncrono para: '{req.ch_src}'.")
        result_json = await run_sentence_alignment(req.import_id, req.ch_src)
        logger.info(f"Endpoint 2b: CONCLUÍDO alinhamento para: '{req.ch_src}'.")

        if "error" in result_json:
            raise HTTPException(status_code=500, detail=result_json)
        return result_json

    except Exception as e:
        logger.exception(f"Falha ao agendar job 2b para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar: {e}")


@router.post("/3-export-corpus")
async def export_corpus_endpoint(
        req: JobRequest,
        background_tasks: BackgroundTasks,
):
    """
    Botão 3: Roda a Fase 3 (Exportação Final).
    """
    try:
        background_tasks.add_task(export_corpus_to_tsv, req.import_id)
        logger.info(f"Endpoint 3: Job de Exportação (Fase 3) iniciado para import_id={req.import_id}.")
        return {
            "status": "export_job_scheduled",
            "import_id": req.import_id,
            "message": "Fase 3 (Exportação para .tsv) iniciada em background."
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar job 3 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar exportação: {e}")


@router.get("/get-status")
async def get_status_endpoint(
        import_id: int,
        db: AsyncSession = Depends(get_db)
):
    """
    (Tela de Status) Checa o progresso do Mapeamento de Capítulos.
    """
    try:
        stats = await get_corpus_status(import_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar status: {e}")


# --- (NOVO) BOTÃO 3: VALIDAÇÃO ---
@router.post("/3-validate-corpus")
async def validate_corpus_endpoint(
        req: JobRequest,
        background_tasks: BackgroundTasks,
):
    """
    Botão 3: Roda a Fase 3 (Validação de Sanidade).

    Processa TODOS os pares alinhados (status 'pending') e
    calcula as métricas extras (len_ratio, number_mismatch).
    Roda em BACKGROUND.
    """
    try:
        background_tasks.add_task(run_corpus_validation, req.import_id)
        logger.info(f"Endpoint 3: Job de Validação (Fase 3) iniciado para import_id={req.import_id}.")
        return {
            "status": "validation_job_scheduled",
            "import_id": req.import_id,
            "message": "Fase 3 (Validação de Pares) iniciada em background."
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar job 3 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar validação: {e}")
