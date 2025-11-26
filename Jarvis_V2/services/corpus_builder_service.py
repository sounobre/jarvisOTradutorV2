# --- ARQUIVO COMPLETO: services/corpus_builder_service.py ---

"""
Este é o "Motor" (Engine) do Pipeline "Embeddings-Only".

ARQUITETURA:
- Fase 1 (Macro): SentenceTransformer (Embeddings de Conteúdo) -> TmWinMapping (com Logs no Banco)
- Fase 2 (Micro): Stanza (Segmentação) + SentAlign (Subprocess) -> TmAlignedSentences
- Fase 3 (Valid): Métricas (Ratio, Numbers) -> TmAlignedSentences (Status)
- Fase 4 (Export): Filtros SQL -> TSV
"""

import logging
import os
import re
import asyncio
import subprocess
import shutil
import uuid
import stanza  # Para segmentação
from typing import Dict, List, Any, Optional
import io
import numpy as np
import torch
from llama_cpp import Llama
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

# --- Importações do Projeto ---
from db.session import AsyncSessionLocal
from db.models import TmWinMapping, ChapterText, ChapterIndex, TmAlignedSentences, TmMacroMapLog, TmAlignmentLog
from db import models
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select, func, update, bindparam, cast, String, distinct, insert, delete

# --- Importações de ML/Processamento ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# --- ☢️☢️☢️ CONFIGURAÇÃO CRÍTICA (SentAlign) ☢️☢️☢️ ---
# Ajuste estes caminhos para o SEU ambiente de teste onde o SentAlign funciona
SENTALIGN_PYTHON_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\.venv\Scripts\python.exe"
SENTALIGN_ROOT_FOLDER = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\SentAlign"
TEMP_PROCESSING_DIR = os.path.join(os.getcwd(), "temp_processing")
# --- FIM DA CONFIGURAÇÃO ---

REPO_ID = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q5_K_M.gguf"

# --- Constantes ---
MODELO_EMBEDDING = 'sentence-transformers/LaBSE'
MACRO_SIMILARITY_THRESHOLD = 0.5
MICRO_SIMILARITY_THRESHOLD = 0.7
OUTPUT_DIR = 'data'
GPU_BATCH_SIZE = 256

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER: LOGS NO BANCO (FASE 1)
# ==============================================================================

async def _log_macro_db(import_id: int, step: str, status: str, message: str):
    """Salva log da Fase 1 no banco (Transação isolada)."""
    if status == 'ERROR':
        logger.error(f"[MacroMap] ID {import_id} - {step}: {message}")
    else:
        logger.info(f"[MacroMap] ID {import_id} - {step}: {message}")

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                insert(TmMacroMapLog).values(
                    import_id=import_id,
                    step=step,
                    status=status,
                    message=str(message)
                )
            )
            await session.commit()
        except Exception as e:
            logger.error(f"FATAL: Falha ao salvar log Macro no banco: {e}")


# ==============================================================================
# HELPERS DE BUSCA E BANCO DE DADOS
# ==============================================================================

async def _get_corpus_preview(db: AsyncSession, import_id: int, lang: str) -> Dict[str, str]:
    """Busca os primeiros 2000 caracteres de cada capítulo para o Macro Map."""
    stmt = select(
        ChapterText.ch_idx,
        func.substr(ChapterText.text, 1, 2000)
    ).where(
        ChapterText.import_id == import_id,
        ChapterText.lang == lang,
        ChapterText.text.is_not(None),
        func.length(ChapterText.text) > 50
    ).order_by(ChapterText.ch_idx)

    resultado = await db.execute(stmt)
    # Retorna {href: texto_parcial}
    return {str(row[0]): str(row[1]) for row in resultado.all()}


async def _get_full_text_by_href(db: AsyncSession, import_id: int, lang: str, ch_idx_href: str) -> Optional[str]:
    """Busca o texto completo usando o HREF."""
    stmt = select(ChapterText.text).where(
        ChapterText.import_id == import_id,
        ChapterText.lang == lang,
        ChapterText.ch_idx == ch_idx_href
    )
    txt = await db.scalar(stmt)
    return txt or None


async def _upsert_many_mappings(db: AsyncSession, values_list: List[Dict[str, Any]]):
    """Salva/Atualiza o mapa de capítulos."""
    if not values_list: return
    tabela = TmWinMapping.__table__
    ins = pg_insert(tabela).values(values_list)
    colunas_para_atualizar = {
        key: ins.excluded[key] for key in values_list[0] if key not in ("import_id", "ch_src")
    }
    upsert_stmt = ins.on_conflict_do_update(
        index_elements=[tabela.c.import_id, tabela.c.ch_src],
        set_=colunas_para_atualizar,
    )
    await db.execute(upsert_stmt)


def _get_numbers_from_text(text: str) -> set[str]:
    """Extrai números para validação."""
    return set(re.findall(r'\d+', text))


# ==============================================================================
# FASE 1: MACRO MAP (AGENDADOR)
# ==============================================================================

async def schedule_macro_map(import_id: int):
    await _log_macro_db(import_id, "START", "INFO", "Iniciando Fase 1 (Macro Map).")

    async with AsyncSessionLocal() as session:
        try:
            # 1. Busca Conteúdo (Preview)
            await _log_macro_db(import_id, "FETCH", "INFO", "Buscando textos EN e PT...")
            en_map = await _get_corpus_preview(session, import_id, 'en')
            pt_map = await _get_corpus_preview(session, import_id, 'pt')

            if not en_map:
                msg = "Nenhum capítulo EN encontrado."
                await _log_macro_db(import_id, "FETCH", "ERROR", msg)
                raise Exception(msg)
            if not pt_map:
                msg = "Nenhum capítulo PT encontrado."
                await _log_macro_db(import_id, "FETCH", "ERROR", msg)
                raise Exception(msg)

            await _log_macro_db(import_id, "FETCH", "SUCCESS",
                                f"Carregados: {len(en_map)} caps EN, {len(pt_map)} caps PT.")

            en_hrefs: List[str] = list(en_map.keys())
            en_texts: List[str] = list(en_map.values())
            pt_hrefs: List[str] = list(pt_map.keys())
            pt_texts: List[str] = list(pt_map.values())

            # 2. Embeddings
            await _log_macro_db(import_id, "EMBEDDING", "INFO", "Carregando modelo e gerando embeddings na GPU...")
            model = SentenceTransformer(MODELO_EMBEDDING, device='cuda')

            en_embeddings = model.encode(en_texts, show_progress_bar=False, batch_size=GPU_BATCH_SIZE)
            pt_embeddings = model.encode(pt_texts, show_progress_bar=False, batch_size=GPU_BATCH_SIZE)
            await _log_macro_db(import_id, "EMBEDDING", "SUCCESS", "Embeddings gerados com sucesso.")

            # 3. Similaridade
            await _log_macro_db(import_id, "MATCHING", "INFO", "Calculando matriz de similaridade...")
            sim_matrix = cosine_similarity(en_embeddings, pt_embeddings)

            linhas_para_salvar: List[Dict[str, Any]] = []
            matches_count = 0

            for i in range(len(en_hrefs)):
                en_href = en_hrefs[i]
                best_pt_j = sim_matrix[i].argmax()
                best_score = float(sim_matrix[i][best_pt_j])
                pt_href = pt_hrefs[best_pt_j]

                foi_match = best_score >= MACRO_SIMILARITY_THRESHOLD
                if foi_match: matches_count += 1

                linhas_para_salvar.append({
                    "import_id": import_id,
                    "ch_src": en_href,
                    "ch_tgt": pt_href if foi_match else None,
                    "sim_cosine": best_score,
                    "len_src": len(en_texts[i]),
                    "len_tgt": len(pt_texts[best_pt_j]),
                    "method": "embedding_content_preview",
                    "llm_score": best_score,
                    "llm_verdict": foi_match,
                    "llm_reason": f"Content Similarity: {best_score:.4f}",
                    "score": best_score,
                    "micro_status": "pending" if foi_match else "skipped",
                })

            # 4. Salvar
            await _log_macro_db(import_id, "SAVING", "INFO",
                                f"Salvando {len(linhas_para_salvar)} mapeamentos no banco...")
            await _upsert_many_mappings(session, linhas_para_salvar)
            await session.commit()

            await _log_macro_db(import_id, "FINISH", "SUCCESS",
                                f"Processo concluído. {matches_count} pares encontrados.")
            return {"message": "Mapeamento concluído.", "total": len(linhas_para_salvar), "matches": matches_count}

        except Exception as e:
            await session.rollback()
            await _log_macro_db(import_id, "FATAL", "ERROR", str(e))
            return {"error": str(e)}


# --- JOB MESTRE FASE 1 ---
async def _find_pending_macro_map_imports() -> List[int]:
    logger.info("Fase 1 (Master): Buscando imports pendentes...")
    async with AsyncSessionLocal() as session:
        sub_q = select(distinct(TmWinMapping.import_id)).subquery()
        stmt = select(models.Import.id).where(models.Import.id.notin_(select(sub_q))).order_by(models.Import.id)
        return list((await session.execute(stmt)).scalars().all())


async def schedule_macro_map_for_all_pending():
    logger.info("JOB MESTRE (FASE 1): Iniciando.")
    try:
        ids = await _find_pending_macro_map_imports()
        if not ids:
            logger.info("JOB MESTRE (FASE 1): Nada pendente.")
            return

        for i, imp_id in enumerate(ids):
            logger.info(f"JOB MESTRE (FASE 1): {i + 1}/{len(ids)} - ID {imp_id}")
            await schedule_macro_map(imp_id)
    except Exception as e:
        logger.error(f"Erro Mestre Fase 1: {e}")


# ==============================================================================
# FASE 2: MICRO ALIGNMENT (STANZA + SENTALIGN)
# ==============================================================================

# --- SUBSTITUA ESTA FUNÇÃO EM: services/corpus_builder_service.py ---

async def _run_sentalign_subprocess(job_folder: str, file_name: str) -> str:
    """
    Função helper que chama o sentAlign.py externo.
    *** ATUALIZADO: Com Timeout de 5 minutos para evitar travamentos ***
    """
    # Tempo máximo que aceitamos esperar pelo SentAlign (em segundos)
    # 300 segundos = 5 minutos (geralmente leva 30s)
    TIMEOUT_SECONDS = 300

    logger.info("Fase 2 (Subprocess): Chamando 'files2align.py'...")
    cmd_files2align = [SENTALIGN_PYTHON_PATH, os.path.join(SENTALIGN_ROOT_FOLDER, "files2align.py"), "-dir", job_folder,
                       "--source-language", "eng"]

    # O files2align é muito rápido, não precisa de timeout complexo
    process1 = await asyncio.create_subprocess_exec(*cmd_files2align, cwd=SENTALIGN_ROOT_FOLDER, stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE)
    stdout1, stderr1 = await process1.communicate()
    if process1.returncode != 0: raise Exception(
        f"Falha no 'files2align.py': {stderr1.decode('utf-8', errors='ignore')}")

    logger.info("Fase 2 (Subprocess): Chamando 'sentAlign.py'...")
    cmd_sentalign = [SENTALIGN_PYTHON_PATH, os.path.join(SENTALIGN_ROOT_FOLDER, "sentAlign.py"), "-dir", job_folder,
                     "-sl", "eng", "-tl", "por", "--proc-device", "cuda"]

    process2 = await asyncio.create_subprocess_exec(
        *cmd_sentalign,
        cwd=SENTALIGN_ROOT_FOLDER,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        # --- A MÁGICA ESTÁ AQUI: wait_for ---
        # Espera até TIMEOUT_SECONDS. Se passar, lança erro.
        stdout2_bytes, stderr2_bytes = await asyncio.wait_for(process2.communicate(), timeout=TIMEOUT_SECONDS)

        if process2.returncode != 0:
            raise Exception(f"Falha no 'sentAlign.py': {stderr2_bytes.decode('utf-8', errors='ignore')}")

        # Se chegou aqui, deu certo
        return os.path.join(job_folder, "output", f"{file_name}.aligned")

    except asyncio.TimeoutError:
        # SE TRAVAR: Mata o processo e lança erro
        logger.error(
            f"Fase 2 (Subprocess): TIMEOUT! O SentAlign travou por mais de {TIMEOUT_SECONDS}s. Matando processo...")
        try:
            process2.kill()  # Mata o processo travado na memória
        except:
            pass  # Ignora se já morreu
        raise Exception(f"Timeout no SentAlign ({TIMEOUT_SECONDS}s excedidos).")


async def _align_one_chapter(
        db: AsyncSession,
        nlp_en: stanza.Pipeline,
        nlp_pt: stanza.Pipeline,
        import_id: int,
        ch_src_href: str
) -> Dict[str, Any]:
    job_folder = ""
    await _log_align_db(import_id, ch_src_href, "START", "INFO", "Iniciando alinhamento do capítulo.")

    try:
        # 1. Busca e Trava
        mapa = await db.get(TmWinMapping, (import_id, ch_src_href))
        if not mapa or not mapa.llm_verdict or mapa.ch_tgt is None:
            msg = "Capítulo inválido ou sem par no mapa."
            await _log_align_db(import_id, ch_src_href, "START", "ERROR", msg)
            raise Exception(msg)

        mapa.micro_status = "processing"
        await db.commit()
        ch_tgt_href = str(mapa.ch_tgt)

        # 2. Texto
        texto_en = await _get_full_text_by_href(db, import_id, 'en', ch_src_href)
        texto_pt = await _get_full_text_by_href(db, import_id, 'pt', ch_tgt_href)
        if not texto_en or not texto_pt:
            msg = "Texto EN ou PT não encontrado no banco."
            await _log_align_db(import_id, ch_src_href, "FETCH", "ERROR", msg)
            raise Exception(msg)

        # 3. Segmentação (Stanza)
        await _log_align_db(import_id, ch_src_href, "SEGMENTATION", "INFO", "Rodando Stanza (CPU/GPU)...")

        sentencas_en = [s.text.strip().replace("\n", " ") for s in nlp_en(texto_en).sentences if s.text.strip()]
        sentencas_pt = [s.text.strip().replace("\n", " ") for s in nlp_pt(texto_pt).sentences if s.text.strip()]

        if not sentencas_en or not sentencas_pt:
            msg = f"Segmentação gerou listas vazias (EN={len(sentencas_en)}, PT={len(sentencas_pt)})."
            await _log_align_db(import_id, ch_src_href, "SEGMENTATION", "ERROR", msg)
            raise Exception(msg)

        await _log_align_db(import_id, ch_src_href, "SEGMENTATION", "SUCCESS",
                            f"Segmentado: {len(sentencas_en)} EN vs {len(sentencas_pt)} PT.")

        # 4. Arquivos Temp e Subprocesso
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(TEMP_PROCESSING_DIR, job_id)
        os.makedirs(os.path.join(job_folder, "eng"), exist_ok=True)
        os.makedirs(os.path.join(job_folder, "por"), exist_ok=True)

        with open(os.path.join(job_folder, "eng", "chapter.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(sentencas_en))
        with open(os.path.join(job_folder, "por", "chapter.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(sentencas_pt))

        await _log_align_db(import_id, ch_src_href, "SUBPROCESS", "INFO", "Chamando SentAlign externo...")
        output_path = await _run_sentalign_subprocess(job_folder, "chapter.txt")

        # 5. Ler e Filtrar
        pares_salvar = []
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        try:
                            if float(parts[2]) >= MICRO_SIMILARITY_THRESHOLD:
                                pares_salvar.append({
                                    "import_id": import_id, "ch_src": ch_src_href,
                                    "source_text": parts[0], "target_text": parts[1],
                                    "similarity_score": float(parts[2])
                                })
                        except:
                            pass
        except FileNotFoundError:
            msg = "Arquivo de saída do SentAlign não foi encontrado."
            await _log_align_db(import_id, ch_src_href, "SUBPROCESS", "ERROR", msg)
            raise Exception(msg)

        # 6. Salvar
        if pares_salvar:
            stmt = pg_insert(TmAlignedSentences).values(pares_salvar)
            await db.execute(
                stmt.on_conflict_do_nothing(index_elements=["import_id", "ch_src", "source_text", "target_text"]))

        mapa.micro_status = "aligned"
        await db.commit()

        await _log_align_db(import_id, ch_src_href, "FINISHED", "SUCCESS",
                            f"Capítulo concluído. {len(pares_salvar)} pares salvos.")
        return {"ok": True, "count": len(pares_salvar)}

    except Exception as e:
        await db.rollback()
        # Loga o erro fatal com o traceback resumido
        await _log_align_db(import_id, ch_src_href, "FATAL", "ERROR", str(e))
        try:
            mapa = await db.get(TmWinMapping, (import_id, ch_src_href))
            if mapa: mapa.micro_status = "error"; await db.commit()
        except:
            pass
        return {"error": str(e)}
    finally:
        if os.path.exists(job_folder): shutil.rmtree(job_folder, ignore_errors=True)


async def run_sentence_alignment(import_id: int, ch_src: str):
    """Roda alinhamento para 1 capítulo específico."""
    try:
        # Download silencioso dos modelos Stanza
        # stanza.download('en', logging_level='WARN', processors='tokenize')
        # stanza.download('pt', logging_level='WARN', processors='tokenize')
        # nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True)
        # nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True)
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
    except Exception as e:
        return {"error": str(e)}

    async with AsyncSessionLocal() as session:
        return await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch_src)


async def run_alignment_for_all_pending(import_id: int):
    """Roda alinhamento para TODOS os capítulos pendentes de um livro."""
    logger.info(f"JOB FASE 2: Iniciando loop para ID {import_id}")
    try:
        # Download silencioso dos modelos Stanza só se precisar muito
        #stanza.download('en', logging_level='WARN', processors='tokenize')
        #stanza.download('pt', logging_level='WARN', processors='tokenize')
        # nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True)
        # nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True)
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True, download_method=None)
    except Exception as e:
        logger.exception(f"Fase 2: Falha Stanza: {e}")
        return

    async with AsyncSessionLocal() as session:
        stmt = select(TmWinMapping.ch_src).where(TmWinMapping.import_id == import_id,
                                                 TmWinMapping.micro_status == "pending").order_by(TmWinMapping.ch_src)
        lista = (await session.execute(stmt)).scalars().all()

        for i, ch_src in enumerate(lista):
            logger.info(f"Fase 2: {i + 1}/{len(lista)} - '{ch_src}'...")
            # Reusamos a session e os modelos para performance
            await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch_src)


# --- JOB MESTRE FASE 2 ---
async def _find_imports_with_pending_alignment() -> List[int]:
    async with AsyncSessionLocal() as session:
        stmt = select(distinct(TmWinMapping.import_id)).where(TmWinMapping.micro_status == "pending").order_by(
            TmWinMapping.import_id)
        return list((await session.execute(stmt)).scalars().all())


async def run_master_alignment_job():
    logger.info("JOB MESTRE (FASE 2): Iniciando.")
    try:
        ids = await _find_imports_with_pending_alignment()
        if not ids:
            logger.info("JOB MESTRE (FASE 2): Nada pendente.")
            return

        for i, imp_id in enumerate(ids):
            logger.info(f"JOB MESTRE (FASE 2): Livro {i + 1}/{len(ids)} - ID {imp_id}")
            await run_alignment_for_all_pending(imp_id)
    except Exception as e:
        logger.error(f"Erro Mestre Fase 2: {e}")


# ==============================================================================
# FASE 3: VALIDAÇÃO E FASE 4: EXPORTAÇÃO
# ==============================================================================

async def run_corpus_validation(import_id: int, batch_size: int = 5000):
    logger.info(f"--- INICIANDO FASE 3 (VALIDAÇÃO) PARA IMPORT ID: {import_id} ---")
    async with AsyncSessionLocal() as session:
        try:
            total_validado = 0
            while True:
                stmt = select(TmAlignedSentences).where(TmAlignedSentences.import_id == import_id,
                                                        TmAlignedSentences.validation_status == "pending").limit(
                    batch_size)
                pares = (await session.execute(stmt)).scalars().all()
                if not pares: break

                for par in pares:
                    len_en = len(par.source_text)
                    len_pt = len(par.target_text)
                    par.len_ratio = len_pt / max(1, len_en)
                    nums_en = _get_numbers_from_text(par.source_text)
                    nums_pt = _get_numbers_from_text(par.target_text)
                    par.number_mismatch = (nums_en != nums_pt)
                    par.validation_status = "validated"

                await session.commit()
                total_validado += len(pares)
                logger.info(f"Fase 3: Lote salvo. Total: {total_validado}")
            return {"message": "Validação concluída.", "total": total_validado}
        except Exception as e:
            return {"error": str(e)}


async def _find_imports_with_pending_validation() -> List[int]:
    async with AsyncSessionLocal() as session:
        stmt = select(distinct(TmAlignedSentences.import_id)).where(
            TmAlignedSentences.validation_status == "pending").order_by(TmAlignedSentences.import_id)
        return list((await session.execute(stmt)).scalars().all())


async def run_master_validation_job():
    logger.info("JOB MESTRE (FASE 3): Iniciando.")
    try:
        ids = await _find_imports_with_pending_validation()
        for imp_id in ids:
            logger.info(f"JOB MESTRE (FASE 3): Livro ID {imp_id}")
            await run_corpus_validation(imp_id)
    except Exception as e:
        logger.error(f"Erro Mestre Fase 3: {e}")


async def export_corpus_to_tsv(import_id: int, min_score: float = 0.7, max_len_ratio: float = 3.0,
                               require_number_match: bool = True):
    logger.info(f"--- INICIANDO FASE 4 (EXPORTAÇÃO) ---")
    output_file = os.path.join(OUTPUT_DIR, f'{import_id}_corpus_premium.tsv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    async with AsyncSessionLocal() as session:
        try:
            stmt = select(TmAlignedSentences.source_text, TmAlignedSentences.target_text).where(
                TmAlignedSentences.import_id == import_id,
                TmAlignedSentences.validation_status == "validated",
                TmAlignedSentences.similarity_score >= min_score,
                TmAlignedSentences.len_ratio <= max_len_ratio
            )
            if require_number_match:
                stmt = stmt.where(TmAlignedSentences.number_mismatch == False)
            stmt = stmt.order_by(TmAlignedSentences.ch_src, TmAlignedSentences.id)

            pares = (await session.execute(stmt)).all()
            if not pares: return {"error": "Nenhum par encontrado."}

            with open(output_file, 'w', encoding='utf-8') as f:
                for en, pt in pares:
                    f.write(f"{en}\t{pt}\n")
            return {"message": "Exportação concluída.", "file": output_file}
        except Exception as e:
            return {"error": str(e)}


# --- ARQUIVO: services/corpus_builder_service.py (Final) ---

# ... (todo o código anterior) ...

# --- HELPER PRIVADO: Calcula status detalhado de UM livro ---
async def _calculate_stats_for_id(session: AsyncSession, import_id: int) -> Dict[str, Any]:
    """
    Faz o trabalho pesado de contar capítulos e frases para UM import_id.
    Inicializa com zeros para o JSON ficar limpo.
    """
    # 1. Contar Capítulos (na TmWinMapping)
    stmt_chapters = (
        select(TmWinMapping.micro_status, func.count(TmWinMapping.ch_src))
        .where(TmWinMapping.import_id == import_id, TmWinMapping.llm_verdict == True)
        .group_by(TmWinMapping.micro_status)
    )
    resultado_chapters = (await session.execute(stmt_chapters)).all()

    # Inicializa o dicionário zerado (para garantir que todas as chaves existam)
    stats = {
        "status_capitulos": {
            "pending": 0, "aligned": 0, "error": 0, "processing": 0, "total_mapeado": 0
        },
        "status_frases": {
            "pending_validation": 0, "validated": 0, "total_alinhado": 0
        }
    }

    # Preenche com os resultados do banco (Capítulos)
    for status, count in resultado_chapters:
        if status in stats["status_capitulos"]:
            stats["status_capitulos"][status] = count
        stats["status_capitulos"]["total_mapeado"] += count

    # 2. Contar Frases (na TmAlignedSentences)
    stmt_sentences = (
        select(TmAlignedSentences.validation_status, func.count(TmAlignedSentences.id))
        .where(TmAlignedSentences.import_id == import_id)
        .group_by(TmAlignedSentences.validation_status)
    )
    resultado_sentences = (await session.execute(stmt_sentences)).all()

    # Preenche com os resultados do banco (Frases)
    for status, count in resultado_sentences:
        if status == "pending":
            stats["status_frases"]["pending_validation"] = count
        elif status == "validated":
            stats["status_frases"]["validated"] = count
        stats["status_frases"]["total_alinhado"] += count

    return {"import_id": import_id, "stats": stats}


# --- FUNÇÃO PÚBLICA 5 (A REAL): O STATUS INTELIGENTE ---
async def get_corpus_status(import_id: Optional[int] = None):
    """
    Se receber um ID: Retorna o status daquele ID.
    Se for None: Retorna uma LISTA com o status de TODOS os projetos (Importados).
    """
    async with AsyncSessionLocal() as session:

        # CASO 1: Usuário pediu um ID específico
        if import_id is not None:
            dados = await _calculate_stats_for_id(session, import_id)
            return dados

        # CASO 2: Usuário quer ver TUDO (Sem ID)
        else:
            # --- A MUDANÇA ESTÁ AQUI ---
            # Antes: Buscava em TmWinMapping (só mostrava quem já começou).
            # Agora: Busca em models.Import (mostra TODOS os livros importados).
            stmt_all_ids = select(models.Import.id).order_by(models.Import.id)
            # ---------------------------

            resultado_ids = (await session.execute(stmt_all_ids)).scalars().all()

            lista_de_status = []

            for id_encontrado in resultado_ids:
                # O helper vai retornar tudo zerado para quem ainda não começou,
                # o que é perfeito para o seu dashboard.
                dados_do_id = await _calculate_stats_for_id(session, id_encontrado)
                lista_de_status.append(dados_do_id)

            return lista_de_status


# --- HELPER DE LOG FASE 2 (NOVO) ---
async def _log_align_db(import_id: int, ch_src: str, step: str, status: str, message: str):
    """Salva log da Fase 2 no banco."""
    if status == 'ERROR':
        logger.error(f"[Align] ID {import_id} [{ch_src}] - {step}: {message}")
    else:
        logger.info(f"[Align] ID {import_id} [{ch_src}] - {step}: {message}")

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                insert(TmAlignmentLog).values(
                    import_id=import_id,
                    ch_src=ch_src,
                    step=step,
                    status=status,
                    message=str(message)
                )
            )
            await session.commit()
        except Exception as e:
            logger.error(f"FATAL: Falha ao salvar log Alignment no banco: {e}")


# --- ADICIONE NO FINAL DE services/corpus_builder_service.py ---

async def run_chapter_audit(import_id: int, auto_fix: bool = True):
    """
    Fase 3-B (Auditoria de Integridade).
    Verifica se os pares de capítulos fazem sentido e LIMPA os dados ruins.
    """
    logger.info(f"--- INICIANDO AUDITORIA (Fase 3B) PARA IMPORT ID: {import_id} ---")

    # Carrega o modelo Qwen GGUF (Leve e Rápido)
    llm = None
    try:
        logger.info("Auditoria: Carregando Juiz IA (Qwen GGUF)...")
        # Usamos contexto de 2048 pois só vamos ler o início dos capítulos
        llm = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False
        )
        logger.info("Auditoria: Juiz IA carregado na GPU.")
    except Exception as e:
        logger.warning(f"Auditoria: Falha ao carregar LLM ({e}). Rodando apenas verificação matemática.")

    async with AsyncSessionLocal() as session:
        # Busca mapeamentos que dizem ser "sucesso" mas podem estar errados
        stmt = select(TmWinMapping).where(
            TmWinMapping.import_id == import_id,
            TmWinMapping.micro_status.in_(['aligned', 'pending']),
            TmWinMapping.ch_tgt.is_not(None)
        ).order_by(TmWinMapping.ch_src)

        mapeamentos = (await session.execute(stmt)).scalars().all()

        audit_report = {"checked": 0, "cleaned_size": 0, "cleaned_llm": 0, "passed": 0}

        for mapa in mapeamentos:
            audit_report["checked"] += 1

            # 1. Busca os textos completos (usando HREF)
            txt_en = await _get_full_text_by_href(session, import_id, 'en', mapa.ch_src)
            # ch_tgt pode ser None, mas o filtro da query já pegou só os não-nulos
            txt_pt = await _get_full_text_by_href(session, import_id, 'pt', str(mapa.ch_tgt))

            if not txt_en or not txt_pt: continue

            len_en = len(txt_en)
            len_pt = len(txt_pt)

            # --- ETAPA 1: O "PORTEIRO" MATEMÁTICO ---
            ratio = len_pt / max(1, len_en)
            is_size_mismatch = (ratio > 4.0) or (ratio < 0.25)

            if is_size_mismatch:
                logger.error(f"[Audit] ERRO TAMANHO: '{mapa.ch_src}' ({len_en}) vs '{mapa.ch_tgt}' ({len_pt}). Ratio: {ratio:.2f}. LIMPANDO.")

                mapa.micro_status = "error_size_mismatch"
                mapa.llm_reason = f"Audit: Size Mismatch (Ratio {ratio:.2f})"

                if auto_fix:
                    # Deleta usando a nova tabela TmAlignedSentences
                    # (Certifique-se de ter 'from sqlalchemy import delete' no topo)
                    from sqlalchemy import delete
                    await session.execute(
                        delete(TmAlignedSentences).where(
                            TmAlignedSentences.import_id == import_id,
                            TmAlignedSentences.ch_src == mapa.ch_src
                        )
                    )

                audit_report["cleaned_size"] += 1
                continue

                # --- ETAPA 2: O "JUIZ" LLM (Qwen) ---
            if llm:
                # Pega intro (600 chars)
                sample_en = txt_en[:600].replace("\n", " ")
                sample_pt = txt_pt[:600].replace("\n", " ")

                # Prompt estilo ChatML
                prompt = f"""<|im_start|>system
You are a specialized editor verifying book translations.
Compare the two text excerpts below.
Task: Determine if they represent the SAME scene or chapter start.
Reply ONLY with 'YES' (match) or 'NO' (mismatch).
<|im_end|>
<|im_start|>user
Text EN: "{sample_en}"
Text PT: "{sample_pt}"
Match?<|im_end|>
<|im_start|>assistant
"""
                # Geração
                output = llm(
                    prompt,
                    max_tokens=5,
                    temperature=0.0,
                    stop=["<|im_end|>"]
                )
                verdict = output['choices'][0]['text'].strip().upper()

                # Limpeza básica da resposta
                import string
                verdict = verdict.strip(string.punctuation)

                if "NO" in verdict:
                    logger.warning(f"[Audit] ERRO SEMÂNTICO (LLM): '{mapa.ch_src}' != '{mapa.ch_tgt}'. LIMPANDO.")

                    mapa.micro_status = "error_content_mismatch"
                    mapa.llm_reason = f"Audit: LLM rejected match."

                    if auto_fix:
                        from sqlalchemy import delete
                        await session.execute(
                            delete(TmAlignedSentences).where(
                                TmAlignedSentences.import_id == import_id,
                                TmAlignedSentences.ch_src == mapa.ch_src
                            )
                        )

                    audit_report["cleaned_llm"] += 1
                else:
                    if mapa.micro_status == 'aligned':
                        mapa.micro_status = "audit_pass"
                    audit_report["passed"] += 1

            # Commit periódico (por capítulo)
            await session.commit()

        logger.info(f"--- AUDITORIA CONCLUÍDA ---")
        return audit_report


async def _find_imports_pending_chapter_audit() -> List[int]:
    """
    Busca todos os import_ids que têm capítulos com status 'aligned' ou 'pending'
    que ainda não passaram pela auditoria (ou seja, não são 'audit_pass' nem 'error...').
    """
    logger.info("Fase 3B (Master): Buscando livros pendentes de auditoria de capítulos...")
    async with AsyncSessionLocal() as session:
        # Procuramos IDs que tenham pelo menos um capítulo "sucesso" (aligned)
        # mas que ainda não virou 'audit_pass'.
        stmt = select(distinct(TmWinMapping.import_id)).where(
            TmWinMapping.micro_status.in_(['aligned', 'pending']),
            TmWinMapping.llm_verdict == True
        ).order_by(TmWinMapping.import_id)

        resultado = (await session.execute(stmt)).scalars().all()
        return list(resultado)


async def run_master_chapter_audit_job():
    """
    Função Mestra da Fase 3B.
    Roda a auditoria (Tamanho + LLM) para todos os livros pendentes.
    """
    logger.info("JOB MESTRE (FASE 3B) INICIADO: Auditando todos os capítulos.")
    try:
        pending_ids = await _find_imports_pending_chapter_audit()
        if not pending_ids:
            logger.info("JOB MESTRE (FASE 3B): Nada pendente de auditoria.")
            return

        logger.info(f"JOB MESTRE (FASE 3B): {len(pending_ids)} livros na fila.")

        for i, imp_id in enumerate(pending_ids):
            logger.info(f"JOB MESTRE (FASE 3B): Processando {i + 1}/{len(pending_ids)} - ID {imp_id}...")

            # Chama a função de auditoria que já criamos
            # (Certifique-se de que 'run_chapter_audit' está definida acima neste arquivo)
            await run_chapter_audit(imp_id, auto_fix=True)

        logger.info("JOB MESTRE (FASE 3B): Ciclo de auditoria concluído.")

    except Exception as e:
        logger.error(f"Erro Mestre Fase 3B: {e}")