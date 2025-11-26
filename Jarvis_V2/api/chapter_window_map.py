# --- ARQUIVO ATUALIZADO (O "PAINEL DE CONTROLE"): api/chapter_window_map.py ---
"""
Este é o "Painel de Controle" da API (Arquitetura "Embeddings-Only").
*** ATUALIZADO: O "Botão 1" agora tem o "Modo Automático" ***
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional  # <-- Importa o 'Optional'

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
    run_alignment_for_all_pending,
    run_corpus_validation,
    schedule_macro_map_for_all_pending, run_master_alignment_job,
    run_master_validation_job, run_chapter_audit, run_master_chapter_audit_job  # <-- A NOVA "FUNÇÃO MESTRA"
)

router = APIRouter(prefix="/epub/corpus_v2", tags=["corpus-pipeline-v2"])
logger = logging.getLogger(__name__)


# -------- Modelos de Request (Payloads) --------

# --- MUDANÇA 1: 'import_id' agora é OPCIONAL ---
class JobRequest(BaseModel):
    import_id: Optional[int] = Field(None,
                                     description="ID do livro em tm_import. Se omitido, processa TODOS os pendentes.")


class AlignOneChapterRequest(BaseModel):
    import_id: int = Field(..., description="ID do livro em tm_import")
    ch_src: str = Field(..., description="O TÍTULO do capítulo EN para processar (ex: 'Chapter 1')")


class ExportRequest(BaseModel):
    import_id: int = Field(..., description="ID do livro em tm_import")
    min_score: Optional[float] = Field(0.7, description="Score de similaridade mínimo (0.0 a 1.0)")
    max_len_ratio: Optional[float] = Field(3.0,
                                           description="Ratio máximo (PT/EN). 3.0 = PT pode ter até 3x o tamanho do EN.")
    require_number_match: Optional[bool] = Field(True,
                                                 description="Se True, descarta pares onde os números (dígitos) não batem.")


# -------- Endpoints (Os Botões) --------

# --- MUDANÇA 2: Lógica "Inteligente" no Botão 1 ---
@router.post("/1-schedule-macro-map")
async def schedule_macro_map_endpoint(
        req: JobRequest,  # <-- Usa o novo payload
        background_tasks: BackgroundTasks,
):
    """
    Botão 1: Roda a Fase 1 (Mapa Macro).
    - Se 'import_id' for fornecido: Roda a Fase 1 para esse ID.
    - Se 'import_id' for omitido (corpo vazio {}): Roda a Fase 1
      para TODOS os 'import_id' que ainda não foram processados.
    """
    try:
        if req.import_id:
            # --- Modo 1: Processamento Único ---
            background_tasks.add_task(schedule_macro_map, req.import_id)
            logger.info(f"Endpoint 1: Job de Agendamento (Macro-Map) iniciado para import_id={req.import_id}.")
            return {
                "status": "macro_map_job_scheduled",
                "import_id": req.import_id,
                "message": "Fase 1 (Mapeamento de Capítulos) iniciada em background."
            }
        else:
            # --- Modo 2: "Modo Automático" ---
            background_tasks.add_task(schedule_macro_map_for_all_pending)
            logger.info(f"Endpoint 1: Job MESTRE (Macro-Map) iniciado para TODOS os imports pendentes.")
            return {
                "status": "master_macro_map_job_scheduled",
                "message": "Fase 1 (Mapeamento de Capítulos) iniciada em background para TODOS os imports pendentes."
            }

    except Exception as e:
        logger.exception(f"Falha ao agendar job 1 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar: {e}")


# --- FIM DA MUDANÇA ---


@router.post("/2-align-next-pending-chapter")
async def align_next_pending_chapter_endpoint(
        req: JobRequest,
        db: AsyncSession = Depends(get_db)
):
    """
    Botão 2 (Automático): Processa o PRÓXIMO capítulo 'pending'.
    Este endpoint ESPERA o alinhamento (15-30s) terminar.
    """
    # (Este endpoint precisa de um ID, então a lógica não muda)
    if not req.import_id:
        raise HTTPException(status_code=400, detail="import_id é obrigatório para este endpoint.")

    try:
        stmt = select(TmWinMapping.ch_src).where(
            TmWinMapping.import_id == req.import_id,
            TmWinMapping.micro_status == "pending"
        ).order_by(TmWinMapping.ch_src).limit(1)

        proximo_capitulo = (await db.execute(stmt)).scalar_one_or_none()

        if not proximo_capitulo:
            return {"message": "Nenhum capítulo pendente encontrado. Alinhamento concluído!",
                    "import_id": req.import_id}

        ch_src_str = str(proximo_capitulo)

        logger.info(f"Endpoint 2: INICIANDO alinhamento síncrono para: '{ch_src_str}'.")
        result_json = await run_sentence_alignment(req.import_id, ch_src_str)
        logger.info(f"Endpoint 2: CONCLUÍDO alinhamento para: '{ch_src_str}'.")

        if "error" in result_json:
            raise HTTPException(status_code=500, detail=result_json)
        return result_json

    except Exception as e:
        logger.exception(f"Falha ao agendar job 2 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar: {e}")


@router.post("/2c-align-all-pending")
async def align_all_pending_endpoint(
        req: JobRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
):
    """
    Botão 2-C (Processar Tudo):
    - Se 'import_id' for enviado: Processa todos os capítulos daquele livro.
    - Se 'import_id' for vazio: Processa TODOS os livros pendentes no banco.
    """
    try:
        if req.import_id:
            # --- MODO 1: Um Livro Específico ---
            stmt = select(TmWinMapping.ch_src).where(
                TmWinMapping.import_id == req.import_id,
                TmWinMapping.micro_status == "pending"
            ).limit(1)

            if not (await db.execute(stmt)).scalar_one_or_none():
                logger.info(f"Endpoint 2C: Nenhum capítulo 'pending' encontrado para import_id={req.import_id}.")
                return {"message": "Nenhum capítulo pendente encontrado para este livro."}

            background_tasks.add_task(run_alignment_for_all_pending, req.import_id)
            logger.info(f"Endpoint 2C: Job iniciado para import_id={req.import_id}.")
            return {
                "status": "full_alignment_job_scheduled",
                "import_id": req.import_id,
                "message": "Fase 2 (Alinhamento do Livro) iniciada em background."
            }

        else:
            # --- MODO 2: Automático (Todos os Livros) ---
            background_tasks.add_task(run_master_alignment_job)
            logger.info(f"Endpoint 2C: Job MESTRE iniciado para TODOS os livros pendentes.")
            return {
                "status": "master_alignment_job_scheduled",
                "message": "Fase 2 (Job Mestre) iniciada em background para TODOS os livros pendentes."
            }

    except Exception as e:
        logger.exception(f"Falha ao agendar job 2C: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar: {e}")


@router.post("/2b-align-specific-chapter")
async def align_specific_chapter_endpoint(
        req: AlignOneChapterRequest,
):
    """
    Botão 2 (Manual): Processa um capítulo específico (Síncrono).
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


@router.post("/3-validate-corpus")
async def validate_corpus_endpoint(
    req: JobRequest, # 'import_id' é Optional[int] neste modelo
    background_tasks: BackgroundTasks,
):
    """
    Botão 3: Roda a Fase 3 (Validação de Sanidade).
    - Com ID: Valida o livro específico.
    - Sem ID: Valida TODOS os livros com pares alinhados.
    """
    try:
        if req.import_id:
            # MODO 1: Processamento Único
            background_tasks.add_task(run_corpus_validation, req.import_id)
            message = f"Validação iniciada para import_id={req.import_id}."
        else:
            # MODO 2: Master (Todos os livros que têm validação pendente)
            background_tasks.add_task(run_master_validation_job)
            message = "Fase 3 (Job Mestre) iniciada em background para TODOS os imports pendentes."

        logger.info(f"Endpoint 3: Job de Validação (Fase 3) iniciado. {message}")
        return {
            "status": "validation_job_scheduled",
            "import_id": req.import_id if req.import_id else "ALL",
            "message": message
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar job 3 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar validação: {e}")


@router.post("/4-export-corpus")
async def export_corpus_endpoint(
        req: ExportRequest,  # <-- Usa o payload de filtros
        background_tasks: BackgroundTasks,
):
    """
    Botão 4: Roda a Fase 4 (Exportação Final).
    """
    try:
        background_tasks.add_task(
            export_corpus_to_tsv,
            req.import_id,
            req.min_score,
            req.max_len_ratio,
            req.require_number_match
        )
        logger.info(f"Endpoint 4: Job de Exportação (Fase 4) iniciado para import_id={req.import_id}.")
        return {
            "status": "export_job_scheduled",
            "import_id": req.import_id,
            "message": "Fase 4 (Exportação Premium para .tsv) iniciada em background."
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar job 4 para import_id={req.import_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao agendar exportação: {e}")


# --- ARQUIVO: api/chapter_window_map.py ---

@router.get("/get-status")
async def get_status_endpoint(
    import_id: Optional[int] = None, # <-- Agora é Opcional (pode ser vazio)
    db: AsyncSession = Depends(get_db)
):
    """
    (Tela de Status)
    - Com ID: Mostra status do livro.
    - Sem ID: Lista o status de TODOS os livros processados.
    """
    try:
        # A lógica agora está toda dentro do serviço
        stats = await get_corpus_status(import_id)
        return stats
    except Exception as e:
        logger.exception(f"Erro ao buscar status: {e}") # Loga o erro completo no console
        raise HTTPException(status_code=500, detail=f"Erro ao buscar status: {e}")


@router.post("/3b-audit-chapters")
async def audit_chapters_endpoint(
    req: JobRequest,
    background_tasks: BackgroundTasks,
):
    """
    Botão 3-B (Auditoria):
    - Com ID: Audita o livro específico.
    - Sem ID (vazio): Audita TODOS os livros que têm capítulos alinhados ('aligned').
    """
    try:
        if req.import_id:
            # MODO 1: Único
            background_tasks.add_task(run_chapter_audit, req.import_id, True) # True = auto_fix
            msg = f"Auditoria iniciada para import_id={req.import_id}."
        else:
            # MODO 2: Automático (Todos)
            background_tasks.add_task(run_master_chapter_audit_job)
            msg = "Job Mestre de Auditoria iniciado para TODOS os livros pendentes."

        logger.info(f"Endpoint 3B: {msg}")
        return {
            "status": "audit_job_scheduled",
            "import_id": req.import_id if req.import_id else "ALL",
            "message": msg
        }

    except Exception as e:
        logger.exception(f"Falha ao agendar auditoria: {e}")
        raise HTTPException(status_code=500, detail=f"Erro: {e}")