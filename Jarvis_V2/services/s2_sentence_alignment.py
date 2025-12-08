# --- ARQUIVO: services/s2_sentence_alignment.py ---
import logging
import os
import uuid
import asyncio
import subprocess
import shutil
import stanza
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select, insert, distinct
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.session import AsyncSessionLocal
from db.models import TmWinMapping, ChapterText, TmAlignedSentences, TmAlignmentLog

logger = logging.getLogger(__name__)

# CONFIGS
SENTALIGN_PYTHON_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\SentAlign\.venv\Scripts\python.exe"
SENTALIGN_ROOT_FOLDER = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\SentAlign"
TEMP_PROCESSING_DIR = os.path.join(os.getcwd(), "temp_processing")
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
MICRO_SIMILARITY_THRESHOLD = 0.7
GPU_BATCH_SIZE = 256


# --- Helpers ---
async def _get_full_text_by_href(db, import_id, lang, href):
    return await db.scalar(select(ChapterText.text).where(ChapterText.import_id == import_id, ChapterText.lang == lang, ChapterText.ch_idx == href))


async def _log_align_db(import_id, ch_src, step, status, message):
    if status == 'ERROR':
        logger.error(f"[Align] {ch_src}: {message}")
    else:
        logger.info(f"[Align] {ch_src}: {message}")
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(insert(TmAlignmentLog).values(import_id=import_id, ch_src=ch_src, step=step, status=status, message=str(message)))
            await session.commit()
        except:
            pass


async def _run_sentalign_subprocess(job_folder: str, file_name: str) -> str:
    TIMEOUT = 300
    logger.info("Subprocess: Chamando files2align...")
    cmd1 = [SENTALIGN_PYTHON_PATH, os.path.join(SENTALIGN_ROOT_FOLDER, "files2align.py"), "-dir", job_folder, "--source-language", "eng"]
    proc1 = await asyncio.create_subprocess_exec(*cmd1, cwd=SENTALIGN_ROOT_FOLDER, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    await proc1.communicate()

    logger.info("Subprocess: Chamando sentAlign...")
    cmd2 = [SENTALIGN_PYTHON_PATH, os.path.join(SENTALIGN_ROOT_FOLDER, "sentAlign.py"), "-dir", job_folder, "-sl", "eng", "-tl", "por", "--proc-device", "cuda"]
    proc2 = await asyncio.create_subprocess_exec(*cmd2, cwd=SENTALIGN_ROOT_FOLDER, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        out, err = await asyncio.wait_for(proc2.communicate(), timeout=TIMEOUT)
        if proc2.returncode != 0: raise Exception(f"SentAlign Error: {err.decode('utf-8', errors='ignore')}")
        return os.path.join(job_folder, "output", f"{file_name}.aligned")
    except asyncio.TimeoutError:
        try:
            proc2.kill()
        except:
            pass
        raise Exception("Timeout SentAlign")


# --- Miolo ---
async def _align_one_chapter(db, nlp_en, nlp_pt, import_id, ch_src_href):
    job_folder = ""
    await _log_align_db(import_id, ch_src_href, "START", "INFO", "Iniciando...")
    try:
        mapa = await db.get(TmWinMapping, (import_id, ch_src_href))
        if not mapa: raise Exception("Mapa não encontrado")
        mapa.micro_status = "processing";
        await db.commit()

        txt_en = await _get_full_text_by_href(db, import_id, 'en', ch_src_href)
        txt_pt = await _get_full_text_by_href(db, import_id, 'pt', str(mapa.ch_tgt))
        if not txt_en or not txt_pt: raise Exception("Texto não encontrado")

        await _log_align_db(import_id, ch_src_href, "SEGMENTATION", "INFO", "Rodando Stanza...")
        sents_en = [s.text.strip().replace("\n", " ") for s in nlp_en(txt_en).sentences if s.text.strip()]
        sents_pt = [s.text.strip().replace("\n", " ") for s in nlp_pt(txt_pt).sentences if s.text.strip()]

        job_id = str(uuid.uuid4());
        job_folder = os.path.join(TEMP_PROCESSING_DIR, job_id)
        os.makedirs(os.path.join(job_folder, "eng"), exist_ok=True)
        os.makedirs(os.path.join(job_folder, "por"), exist_ok=True)
        with open(os.path.join(job_folder, "eng", "chapter.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(sents_en))
        with open(os.path.join(job_folder, "por", "chapter.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(sents_pt))

        await _log_align_db(import_id, ch_src_href, "SUBPROCESS", "INFO", "SentAlign...")
        out_path = await _run_sentalign_subprocess(job_folder, "chapter.txt")

        pares = []
        with open(out_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    try:
                        sc = float(parts[2])
                        if sc >= MICRO_SIMILARITY_THRESHOLD:
                            pares.append({"import_id": import_id, "ch_src": ch_src_href, "source_text": parts[0], "target_text": parts[1], "similarity_score": sc})
                    except:
                        pass

        if pares:
            await db.execute(pg_insert(TmAlignedSentences).values(pares).on_conflict_do_nothing())

        mapa.micro_status = "aligned";
        await db.commit()
        await _log_align_db(import_id, ch_src_href, "FINISHED", "SUCCESS", f"{len(pares)} pares.")
        return {"ok": True}

    except Exception as e:
        await db.rollback()
        await _log_align_db(import_id, ch_src_href, "FATAL", "ERROR", str(e))
        try:
            mapa = await db.get(TmWinMapping, (import_id, ch_src_href))
            if mapa: mapa.micro_status = "error"; await db.commit()
        except:
            pass
        return {"error": str(e)}
    finally:
        if os.path.exists(job_folder): shutil.rmtree(job_folder, ignore_errors=True)


# --- Public ---
async def run_sentence_alignment(import_id: int, ch_src: str):
    # Stanza configs
    try:
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
    except:
        return {"error": "Stanza fail"}
    async with AsyncSessionLocal() as session:
        return await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch_src)


async def run_alignment_for_all_pending(import_id: int):
    try:
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
    except:
        return
    async with AsyncSessionLocal() as session:
        stmt = select(TmWinMapping.ch_src).where(TmWinMapping.import_id == import_id, TmWinMapping.micro_status == "pending").order_by(TmWinMapping.ch_src)
        for ch in (await session.execute(stmt)).scalars().all(): await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch)


async def _find_imports_with_pending_alignment() -> List[int]:
    """
    Busca imports que têm capítulos prontos para alinhar.
    Status alvo: 'pending'
    """
    async with AsyncSessionLocal() as session:
        stmt = select(distinct(TmWinMapping.import_id)).where(
            TmWinMapping.micro_status == "pending"
            # Não incluímos 'processing' (está rodando) nem 'error' (precisa de reset)
        ).order_by(TmWinMapping.import_id)
        return list((await session.execute(stmt)).scalars().all())

async def run_master_alignment_job():
    ids = await _find_imports_with_pending_alignment()
    for i in ids: await run_alignment_for_all_pending(i)