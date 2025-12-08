# --- ARQUIVO: api/chapter_window_map.py ---
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import TmWinMapping

# --- IMPORTS DOS NOVOS SERVIÇOS MODULARES ---
from services.s1_macro_map import (
    schedule_macro_map,
    schedule_macro_map_for_all_pending,
    manual_fix_chapter_map, reset_stuck_or_error_chapters  # <--- NOVO
)
from services.s2_sentence_alignment import (
    run_sentence_alignment,
    run_alignment_for_all_pending,
    run_master_alignment_job
)
from services.s3_master_validation_job import (
    run_corpus_validation,
    run_master_validation_job,
    run_chapter_audit,
    run_master_chapter_audit_job
)
from services.s4_corpus_to_tsv import export_corpus_to_tsv
from services.sall_get_corpus_status import get_corpus_status

router = APIRouter(prefix="/epub/corpus_v2", tags=["corpus-pipeline-v2"])
logger = logging.getLogger(__name__)


# -------- Modelos de Request --------
class JobRequest(BaseModel):
    import_id: Optional[int] = Field(None, description="ID do livro. Se omitido, processa todos.")


class AlignOneChapterRequest(BaseModel):
    import_id: int = Field(...)
    ch_src: str = Field(...)


class ExportRequest(BaseModel):
    # MUDANÇA: Agora é Optional[int] = None
    import_id: Optional[int] = Field(None, description="ID do livro. Se omitido, usa TODO o banco.")

    min_score: Optional[float] = Field(0.7, description="Score do Bi-Encoder (LaBSE)")
    min_cross_score: Optional[float] = Field(0.0, description="Score do Cross-Encoder")
    max_len_ratio: Optional[float] = Field(3.0, description="Ratio máximo")
    require_number_match: Optional[bool] = Field(True, description="Checagem de números")
    limit: Optional[int] = Field(None, description="Limite de linhas (Pilot)")
    only_new: Optional[bool] = Field(False, description="Apenas novos")
    context_mode: Optional[bool] = Field(False, description="Se True, relaxa filtros para manter a continuidade do texto (ideal para treino de blocos).")


# --- NOVO MODELO PARA CORREÇÃO MANUAL ---
class ManualFixRequest(BaseModel):
    import_id: int
    ch_src: str = Field(..., description="O HREF do capítulo EN (que está errado)")
    correct_ch_tgt: str = Field(..., description="O HREF correto do capítulo PT")


# ==============================================================================
# FASE 1: MACRO MAP
# ==============================================================================

@router.post("/1-schedule-macro-map")
async def schedule_macro_map_endpoint(req: JobRequest, background_tasks: BackgroundTasks):
    try:
        if req.import_id:
            background_tasks.add_task(schedule_macro_map, req.import_id)
            msg = f"Fase 1 iniciada para import_id={req.import_id}."
        else:
            background_tasks.add_task(schedule_macro_map_for_all_pending)
            msg = "Fase 1 (Mestre) iniciada para TODOS os pendentes."
        return {"status": "scheduled", "message": msg}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, detail=str(e))


@router.post("/1b-manual-map-fix")
async def manual_map_fix_endpoint(req: ManualFixRequest):
    """
    Corrige manualmente um par errado e reseta para 'pending'.
    """
    try:
        result = await manual_fix_chapter_map(req.import_id, req.ch_src, req.correct_ch_tgt)
        if "error" in result:
            raise HTTPException(400, detail=result["error"])
        return result
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, detail=str(e))


# ==============================================================================
# FASE 2: MICRO ALIGNMENT
# ==============================================================================

@router.post("/2-align-next-pending-chapter")
async def align_next_pending_chapter_endpoint(req: JobRequest, db: AsyncSession = Depends(get_db)):
    if not req.import_id: raise HTTPException(400, "import_id obrigatório")
    try:
        stmt = select(TmWinMapping.ch_src).where(TmWinMapping.import_id == req.import_id, TmWinMapping.micro_status == "pending").limit(1)
        proximo = (await db.execute(stmt)).scalar_one_or_none()
        if not proximo: return {"message": "Nada pendente."}

        logger.info(f"Alinhando síncrono: {proximo}")
        result = await run_sentence_alignment(req.import_id, str(proximo))
        if "error" in result: raise HTTPException(500, detail=result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/2b-align-specific-chapter")
async def align_specific_chapter_endpoint(req: AlignOneChapterRequest):
    try:
        result = await run_sentence_alignment(req.import_id, req.ch_src)
        if "error" in result: raise HTTPException(500, detail=result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/2c-align-all-pending")
async def align_all_pending_endpoint(req: JobRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    try:
        if req.import_id:
            # Verifica se tem trabalho
            stmt = select(TmWinMapping.ch_src).where(TmWinMapping.import_id == req.import_id, TmWinMapping.micro_status == "pending").limit(1)
            if not (await db.execute(stmt)).scalar_one_or_none(): return {"message": "Nada pendente."}
            background_tasks.add_task(run_alignment_for_all_pending, req.import_id)
            msg = f"Fase 2 (Loop) iniciada para ID {req.import_id}."
        else:
            background_tasks.add_task(run_master_alignment_job)
            msg = "Fase 2 (Mestre) iniciada para TODOS."
        return {"status": "scheduled", "message": msg}
    except Exception as e:
        raise HTTPException(500, str(e))


# ==============================================================================
# FASE 3: VALIDAÇÃO E AUDITORIA
# ==============================================================================

@router.post("/3-validate-corpus")
async def validate_corpus_endpoint(req: JobRequest, background_tasks: BackgroundTasks):
    """Validação Fina (Cross-Encoder nas frases)"""
    try:
        if req.import_id:
            background_tasks.add_task(run_corpus_validation, req.import_id)
            msg = f"Validação (Fase 3) iniciada para ID {req.import_id}."
        else:
            background_tasks.add_task(run_master_validation_job)
            msg = "Validação Mestre iniciada."
        return {"status": "scheduled", "message": msg}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/3b-audit-chapters")
async def audit_chapters_endpoint(req: JobRequest, background_tasks: BackgroundTasks):
    """Auditoria Grossa (LLM nos capítulos)"""
    try:
        if req.import_id:
            background_tasks.add_task(run_chapter_audit, req.import_id, True)
            msg = f"Auditoria de Capítulos iniciada para ID {req.import_id}."
        else:
            background_tasks.add_task(run_master_chapter_audit_job)
            msg = "Auditoria Mestre iniciada."
        return {"status": "scheduled", "message": msg}
    except Exception as e:
        raise HTTPException(500, str(e))


# ==============================================================================
# FASE 4 & STATUS
# ==============================================================================

@router.post("/4-export-corpus")
async def export_corpus_endpoint(
        req: ExportRequest,
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
            req.require_number_match,
            req.min_cross_score,  # Passando o novo
            req.limit,  # Passando o limite
            req.only_new,  # Passando o incremental
            req.context_mode
        )

        mode = "PILOT" if req.limit else "FULL"
        return {
            "status": "export_job_scheduled",
            "import_id": req.import_id,
            "mode": mode,
            "message": f"Exportação {mode} iniciada em background."
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar job 4: {e}")
        raise HTTPException(status_code=500, detail=f"Erro: {e}")


@router.get("/get-status")
async def get_status_endpoint(import_id: Optional[int] = None):
    try:
        return await get_corpus_status(import_id)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/admin/reset-status")
async def reset_status_endpoint(req: JobRequest):
    """
    Reseta status 'processing' ou 'error' para 'pending'.
    Útil se o servidor caiu no meio de um job ou para tentar novamente.
    """
    try:
        return await reset_stuck_or_error_chapters(req.import_id)
    except Exception as e:
        raise HTTPException(500, str(e))