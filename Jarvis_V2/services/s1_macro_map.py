# --- ARQUIVO: services/s1_macro_map.py ---
import logging
from typing import Dict, List, Any, Optional
from sqlalchemy import select, func, distinct, update, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from db.session import AsyncSessionLocal
from db import models
from db.models import TmWinMapping, ChapterText, TmMacroMapLog, TmAlignedSentences

logger = logging.getLogger(__name__)

MODELO_EMBEDDING = 'sentence-transformers/LaBSE'
MACRO_SIMILARITY_THRESHOLD = 0.5
GPU_BATCH_SIZE = 64


# --- Helpers Locais ---
async def _log_macro_db(import_id: int, step: str, status: str, message: str):
    if status == 'ERROR':
        logger.error(f"[MacroMap] ID {import_id} - {step}: {message}")
    else:
        logger.info(f"[MacroMap] ID {import_id} - {step}: {message}")
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(insert(TmMacroMapLog).values(import_id=import_id, step=step, status=status, message=str(message)))
            await session.commit()
        except:
            pass


async def _get_corpus_preview(db: AsyncSession, import_id: int, lang: str) -> Dict[str, str]:
    # 1. Monta a Query (Pergunta ao Banco)
    logger.info(f"DEBUG QUERY: import_id={import_id} (type: {type(import_id)}), lang='{lang}'")

    stmt = select(
        ChapterText.ch_idx,
        func.substr(ChapterText.text, 1, 4000)
    ).where(
        ChapterText.import_id == import_id,
        ChapterText.lang == lang,
        # Removemos o is_not(None) se usarmos o char_count > 50, pois implica que tem texto
        # Mas por segurança, deixe o char_count que é INDEXADO
        ChapterText.char_count > 50
    ).order_by(ChapterText.ch_idx)

    # (Opcional) Imprime a query compilada para ver se o SQL está certo
    # from sqlalchemy.dialects import postgresql
    # print(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

    resultado = await db.execute(stmt)

    # Transforma as linhas do banco em um Dicionário Python:
    # { "OEBPS/cap1.xhtml": "Era uma vez um reino distante..." }
    return {str(r[0]): str(r[1]) for r in resultado.all()}


async def _upsert_many_mappings(db: AsyncSession, values_list: List[Dict[str, Any]]):
    if not values_list: return
    stmt = pg_insert(TmWinMapping).values(values_list)
    await db.execute(stmt.on_conflict_do_update(index_elements=[TmWinMapping.import_id, TmWinMapping.ch_src], set_={k: stmt.excluded[k] for k in values_list[0] if k not in ("import_id", "ch_src")}))


# --- Função Mestra Fase 1 ---
async def schedule_macro_map(import_id: int):
    await _log_macro_db(import_id, "START", "INFO", f"Iniciando Fase 1 ({MODELO_EMBEDDING})")
    async with AsyncSessionLocal() as session:
        try:
            en_map = await _get_corpus_preview(session, import_id, 'en')
            pt_map = await _get_corpus_preview(session, import_id, 'pt')
            if not en_map or not pt_map: raise Exception("Faltam dados EN ou PT.")

            en_hrefs, en_texts = list(en_map.keys()), list(en_map.values())
            pt_hrefs, pt_texts = list(pt_map.keys()), list(pt_map.values())

            await _log_macro_db(import_id, "EMBEDDING", "INFO", "Gerando embeddings...")
            model = SentenceTransformer(MODELO_EMBEDDING, device='cuda')
            en_emb = model.encode(en_texts, batch_size=GPU_BATCH_SIZE, show_progress_bar=False)
            pt_emb = model.encode(pt_texts, batch_size=GPU_BATCH_SIZE, show_progress_bar=False)

            sim_matrix = cosine_similarity(en_emb, pt_emb)
            linhas = []
            matches = 0

            for i in range(len(en_hrefs)):
                best_j = sim_matrix[i].argmax()
                score = float(sim_matrix[i][best_j])

                # Pré-filtro de tamanho (Ratio)
                len_en = len(en_texts[i])
                len_pt = len(pt_texts[best_j])
                ratio = len_pt / max(1, len_en)
                is_size_ok = 0.25 <= ratio <= 4.0

                match = (score >= MACRO_SIMILARITY_THRESHOLD) and is_size_ok
                status = "pending" if match else ("skipped_size" if not is_size_ok else "skipped_low_score")
                if match: matches += 1

                linhas.append({
                    "import_id": import_id, "ch_src": en_hrefs[i], "ch_tgt": pt_hrefs[best_j] if match else None,
                    "sim_cosine": score, "len_src": len_en, "len_tgt": len_pt,
                    "method": "labse_content_v2", "llm_score": score, "llm_verdict": match,
                    "llm_reason": f"LaBSE: {score:.4f} | R: {ratio:.2f}", "score": score, "micro_status": status
                })

            await _upsert_many_mappings(session, linhas)
            await session.commit()
            await _log_macro_db(import_id, "FINISH", "SUCCESS", f"Concluído. {matches} pares.")
        except Exception as e:
            await session.rollback()
            await _log_macro_db(import_id, "FATAL", "ERROR", str(e))


# --- Job Mestre ---
async def _find_pending_macro_map_imports() -> List[int]:
    async with AsyncSessionLocal() as session:
        sub_q = select(distinct(TmWinMapping.import_id)).subquery()
        stmt = select(models.Import.id).where(models.Import.id.notin_(select(sub_q))).order_by(models.Import.id)
        return list((await session.execute(stmt)).scalars().all())


async def schedule_macro_map_for_all_pending():
    try:
        ids = await _find_pending_macro_map_imports()
        for imp_id in ids: await schedule_macro_map(imp_id)
    except:
        pass


# --- CORREÇÃO MANUAL (O que você pediu) ---
async def manual_fix_chapter_map(import_id: int, ch_src: str, correct_ch_tgt: str):
    logger.info(f"MANUAL FIX: {ch_src} -> {correct_ch_tgt} (ID {import_id})")
    async with AsyncSessionLocal() as session:
        try:
            # 1. Verifica se target existe
            stmt_check = select(ChapterText.id).where(ChapterText.import_id == import_id, ChapterText.lang == 'pt', ChapterText.ch_idx == correct_ch_tgt)
            if not (await session.execute(stmt_check)).scalar_one_or_none():
                return {"error": "Target não encontrado na ChapterText."}

            # 2. Atualiza e RESETA status para 'pending'
            stmt = update(TmWinMapping).where(TmWinMapping.import_id == import_id, TmWinMapping.ch_src == ch_src).values(
                ch_tgt=correct_ch_tgt, llm_verdict=True, llm_reason="Manual Fix", micro_status="pending", sim_cosine=1.0
            )
            res = await session.execute(stmt)

            # Se não existia, cria
            if res.rowcount == 0:
                session.add(TmWinMapping(import_id=import_id, ch_src=ch_src, ch_tgt=correct_ch_tgt, llm_verdict=True, llm_reason="Manual Fix (New)", micro_status="pending", sim_cosine=1.0, len_src=0, len_tgt=0))

            # 3. Limpa lixo antigo da Fase 2
            await session.execute(models.TmAlignedSentences.__table__.delete().where(models.TmAlignedSentences.import_id == import_id, models.TmAlignedSentences.ch_src == ch_src))

            await session.commit()
            return {"message": "Par corrigido. Status resetado para 'pending'."}
        except Exception as e:
            return {"error": str(e)}


# Adicione imports necessários: update, etc.

async def reset_stuck_or_error_chapters(import_id: Optional[int] = None):
    """
    Reseta capítulos travados ('processing') ou com erro ('error...')
    de volta para 'pending', para que a Fase 2 tente novamente.
    """
    logger.info(f"RESET: Destravando capítulos {'do ID ' + str(import_id) if import_id else 'de TODOS'}.")
    async with AsyncSessionLocal() as session:
        # Query base
        stmt = update(TmWinMapping).where(
            TmWinMapping.micro_status.in_([
                'processing',  # Travou no meio (crash)
                'error',  # Erro genérico (ex: Stanza falhou)
                'error_size_mismatch',   # CUIDADO: Se falhou por tamanho, vai falhar de novo. Melhor corrigir manual.
                'error_content_mismatch' # CUIDADO: Idem.
            ])
        ).values(micro_status='pending')

        if import_id:
            stmt = stmt.where(TmWinMapping.import_id == import_id)

        result = await session.execute(stmt)
        await session.commit()

        logger.info(f"RESET: {result.rowcount} capítulos resetados para 'pending'.")
        return {"reset_count": result.rowcount}