# --- ARQUIVO: services/s3_master_validation_job.py ---
import logging
import re
from typing import List, Optional
from sqlalchemy import select, distinct, delete
from db.models import TmAlignedSentences, TmWinMapping, ChapterText
from db.session import AsyncSessionLocal
from llama_cpp import Llama
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODELO_CROSS_ENCODER = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
# Constantes para o Auditor (Qwen)
REPO_ID = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q5_K_M.gguf"


def _get_numbers_from_text(text: str) -> set[str]:
    return set(re.findall(r'\d+', text))


async def _get_full_text_by_href(db, import_id, lang, href) -> Optional[str]:
    return await db.scalar(select(ChapterText.text).where(ChapterText.import_id == import_id, ChapterText.lang == lang, ChapterText.ch_idx == href))


# --- 3A: Validação Fina (Cross Encoder) ---
async def run_corpus_validation(import_id: int, batch_size: int = 200):
    logger.info(f"--- FASE 3A (CROSS ENCODER) ID: {import_id} ---")
    try:
        cross_model = CrossEncoder(MODELO_CROSS_ENCODER, device='cuda')
    except Exception as e:
        return {"error": str(e)}

    async with AsyncSessionLocal() as session:
        while True:
            # Busca Pendentes OU Sem Score
            stmt = select(TmAlignedSentences).where(
                TmAlignedSentences.import_id == import_id,
                (TmAlignedSentences.validation_status == "pending") |
                (TmAlignedSentences.cross_encoder_score.is_(None))
            ).limit(batch_size)

            pares = (await session.execute(stmt)).scalars().all()
            if not pares: break

            inputs = [[p.source_text, p.target_text] for p in pares]
            scores = cross_model.predict(inputs)

            for i, par in enumerate(pares):
                par.len_ratio = len(par.target_text) / max(1, len(par.source_text))
                par.number_mismatch = (_get_numbers_from_text(par.source_text) != _get_numbers_from_text(par.target_text))
                par.cross_encoder_score = float(scores[i])
                par.validation_status = "validated"

            await session.commit()
    return {"ok": True}


async def _find_imports_with_pending_validation() -> List[int]:
    async with AsyncSessionLocal() as session:
        stmt = select(distinct(TmAlignedSentences.import_id)).where((TmAlignedSentences.validation_status == "pending") | (TmAlignedSentences.cross_encoder_score.is_(None))).order_by(TmAlignedSentences.import_id)
        return list((await session.execute(stmt)).scalars().all())


async def run_master_validation_job():
    ids = await _find_imports_with_pending_validation()
    for i in ids: await run_corpus_validation(i)


# --- 3B: Auditoria de Capítulos (Qwen Judge) ---
async def run_chapter_audit(import_id: int, auto_fix: bool = True):
    logger.info(f"--- FASE 3B (AUDITORIA) ID: {import_id} ---")
    llm = None
    try:
        llm = Llama.from_pretrained(repo_id=REPO_ID, filename=MODEL_FILE, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    except:
        pass

    async with AsyncSessionLocal() as session:
        stmt = select(TmWinMapping).where(
            TmWinMapping.import_id == import_id,
            TmWinMapping.micro_status.in_(['aligned', 'pending']),
            TmWinMapping.ch_tgt.is_not(None)
        ).order_by(TmWinMapping.ch_src)

        for mapa in (await session.execute(stmt)).scalars().all():
            txt_en = await _get_full_text_by_href(session, import_id, 'en', mapa.ch_src)
            txt_pt = await _get_full_text_by_href(session, import_id, 'pt', str(mapa.ch_tgt))
            if not txt_en or not txt_pt: continue

            ratio = len(txt_pt) / max(1, len(txt_en))
            fail = (ratio > 4.0) or (ratio < 0.25)

            if not fail and llm:
                prompt = f"<|im_start|>system\nCompare:\nEN: {txt_en[:600]}\nPT: {txt_pt[:600]}\nSame scene? YES/NO<|im_end|>\n<|im_start|>assistant\n"
                out = llm(prompt, max_tokens=5, temperature=0.0)
                if "NO" in out['choices'][0]['text'].upper(): fail = True

            if fail:
                logger.warning(f"Auditoria falhou: {mapa.ch_src}. Limpando...")
                mapa.micro_status = "error_audit"
                if auto_fix:
                    await session.execute(delete(TmAlignedSentences).where(TmAlignedSentences.import_id == import_id, TmAlignedSentences.ch_src == mapa.ch_src))
            elif mapa.micro_status == 'aligned':
                mapa.micro_status = "audit_pass"

            await session.commit()


# Em services/s3_master_validation_job.py

async def _find_imports_pending_chapter_audit() -> List[int]:
    """
    Busca imports que acabaram de ser alinhados e precisam de auditoria.
    Status alvo: 'aligned' (principal) e 'pending' (opcional, se quiser pré-auditar).
    """
    async with AsyncSessionLocal() as session:
        stmt = select(distinct(TmWinMapping.import_id)).where(
            # Pegamos 'aligned' (sucesso da fase 2) e 'pending' (caso queira auditar antes)
            TmWinMapping.micro_status.in_(['aligned', 'pending']),
            TmWinMapping.llm_verdict == True
        ).order_by(TmWinMapping.import_id)

        resultado = (await session.execute(stmt)).scalars().all()
        return list(resultado)

async def run_master_chapter_audit_job():
    ids = await _find_imports_pending_chapter_audit()
    for i in ids: await run_chapter_audit(i)