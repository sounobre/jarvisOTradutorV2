# --- ARQUIVO ATUALIZADO: services/corpus_builder_service.py ---

"""
Este é o "Motor" (Engine) do Pipeline.
*** ATUALIZADO: Fase 1 agora compara CONTEÚDO (não títulos) para maior precisão ***
"""

import logging
import os
import re
import asyncio
import subprocess
import shutil
import uuid
import stanza
from typing import Dict, List, Any, Optional
import io
import numpy as np

from db import models
# --- Importações do Projeto ---
from db.session import AsyncSessionLocal
from db.models import TmWinMapping, ChapterText, ChapterIndex, TmAlignedSentences
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select, func, update, bindparam, cast, String, distinct

# --- Importações de ML ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÃO ---
SENTALIGN_PYTHON_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\.venv\Scripts\python.exe"
SENTALIGN_ROOT_FOLDER = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\SentAlign"
TEMP_PROCESSING_DIR = os.path.join(os.getcwd(), "temp_processing")

# --- Constantes ---
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
MACRO_SIMILARITY_THRESHOLD = 0.5
MICRO_SIMILARITY_THRESHOLD = 0.7
OUTPUT_DIR = 'data'
GPU_BATCH_SIZE = 256

logger = logging.getLogger(__name__)


# --- Funções "Privadas" ---

async def _get_corpus_preview(db: AsyncSession, import_id: int, lang: str) -> Dict[str, str]:
    """
    Busca os primeiros 2000 caracteres de cada capítulo.
    Retorna: { 'caminho/arquivo.xhtml': 'Texto parcial do capítulo...' }
    """
    logger.info(f"Fase 1: Buscando PREVIEW de conteúdo para '{lang}'...")

    # Note que usamos ChapterText.ch_idx (que é o HREF) como chave
    stmt = select(
        ChapterText.ch_idx,  # A CHAVE (HREF)
        func.substr(ChapterText.text, 1, 2000)  # O VALOR (Primeiros 2k chars)
    ).where(
        ChapterText.import_id == import_id,
        ChapterText.lang == lang,
        ChapterText.text.is_not(None),
        func.length(ChapterText.text) > 50
    ).order_by(ChapterText.ch_idx)  # Ordem alfabética dos arquivos (geralmente funciona para EPUB)

    resultado = await db.execute(stmt)
    # Retorna Dict[Href, TextoParcial]
    return {str(row[0]): str(row[1]) for row in resultado.all()}


async def _get_full_text_by_href(db: AsyncSession, import_id: int, lang: str, ch_idx_href: str) -> Optional[str]:
    stmt = select(ChapterText.text).where(
        ChapterText.import_id == import_id,
        ChapterText.lang == lang,
        ChapterText.ch_idx == ch_idx_href
    )
    txt = await db.scalar(stmt)
    return txt or None


async def _upsert_many_mappings(db: AsyncSession, values_list: List[Dict[str, Any]]):
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
    return set(re.findall(r'\d+', text))


# --- FUNÇÃO PÚBLICA 1: O AGENDADOR (Fase 1 - MACRO MAP) ---
async def schedule_macro_map(import_id: int):
    logger.info(f"JOB DE AGENDAMENTO (FASE 1) INICIADO (import_id={import_id})")
    async with AsyncSessionLocal() as session:
        try:
            logger.info(f"Fase 1: Carregando modelo embedding: '{MODELO_EMBEDDING}'...")
            model = SentenceTransformer(MODELO_EMBEDDING, device='cuda')

            # 1. Busca o CONTEÚDO (Preview) usando HREF como chave
            en_map = await _get_corpus_preview(session, import_id, 'en')
            pt_map = await _get_corpus_preview(session, import_id, 'pt')

            if not en_map or not pt_map:
                raise Exception(f"Fase 1: Faltam dados de texto em EN ou PT.")

            # As chaves agora são os HREFs (ex: OEBPS/text/ch1.xhtml)
            en_hrefs: List[str] = list(en_map.keys())
            en_texts: List[str] = list(en_map.values())  # O DNA vem do TEXTO

            pt_hrefs: List[str] = list(pt_map.keys())
            pt_texts: List[str] = list(pt_map.values())  # O DNA vem do TEXTO

            logger.info(f"Fase 1: Gerando embeddings de CONTEÚDO ({len(en_texts)} EN vs {len(pt_texts)} PT)...")
            en_embeddings = model.encode(en_texts, show_progress_bar=False, batch_size=GPU_BATCH_SIZE)
            pt_embeddings = model.encode(pt_texts, show_progress_bar=False, batch_size=GPU_BATCH_SIZE)

            logger.info("Fase 1: Calculando matriz de similaridade...")
            sim_matrix = cosine_similarity(en_embeddings, pt_embeddings)

            logger.info("Fase 1: Encontrando pares e salvando...")
            linhas_para_salvar_no_db: List[Dict[str, Any]] = []

            for i in range(len(en_hrefs)):
                en_href = en_hrefs[i]

                best_pt_j = sim_matrix[i].argmax()
                best_score = float(sim_matrix[i][best_pt_j])
                pt_href = pt_hrefs[best_pt_j]

                foi_match = best_score >= MACRO_SIMILARITY_THRESHOLD

                linhas_para_salvar_no_db.append({
                    "import_id": import_id,
                    "ch_src": en_href,  # Salva o HREF correto
                    "ch_tgt": pt_href if foi_match else None,  # Salva o HREF correto
                    "sim_cosine": best_score,
                    "len_src": len(en_texts[i]),
                    "len_tgt": len(pt_texts[best_pt_j]),
                    "method": "embedding_content_preview",  # Método atualizado
                    "llm_score": best_score,
                    "llm_verdict": foi_match,
                    "llm_reason": f"Content Similarity: {best_score:.4f}",
                    "score": best_score,
                    "micro_status": "pending" if foi_match else "skipped",
                })

            await _upsert_many_mappings(session, linhas_para_salvar_no_db)
            await session.commit()
            logger.info(f"--- FASE 1 (MACRO-MAP) CONCLUÍDA ---")
            return {"message": "Mapeamento macro (por conteúdo) concluído.",
                    "capitulos_mapeados": len(linhas_para_salvar_no_db)}

        except Exception as e:
            await session.rollback()
            logger.exception(f"ERRO CRÍTICO na Fase 1 para import_id={import_id}: {e}")
            return {"error": str(e)}


# --- "MIOLO" DA FASE 2 (Stanza + Subprocess) ---
async def _run_sentalign_subprocess(job_folder: str, file_name: str) -> str:
    logger.info("Fase 2 (Subprocess): Chamando 'files2align.py'...")
    cmd_files2align = [
        SENTALIGN_PYTHON_PATH,
        os.path.join(SENTALIGN_ROOT_FOLDER, "files2align.py"),
        "-dir", job_folder,
        "--source-language", "eng"
    ]
    process1 = await asyncio.create_subprocess_exec(
        *cmd_files2align, cwd=SENTALIGN_ROOT_FOLDER,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout1, stderr1 = await process1.communicate()
    if process1.returncode != 0:
        raise Exception(f"Falha no 'files2align.py': {stderr1.decode('utf-8', errors='ignore')}")

    logger.info("Fase 2 (Subprocess): Chamando 'sentAlign.py' (na 3060)...")
    cmd_sentalign = [
        SENTALIGN_PYTHON_PATH,
        os.path.join(SENTALIGN_ROOT_FOLDER, "sentAlign.py"),
        "-dir", job_folder,
        "-sl", "eng", "-tl", "por", "--proc-device", "cuda"
    ]
    process2 = await asyncio.create_subprocess_exec(
        *cmd_sentalign, cwd=SENTALIGN_ROOT_FOLDER,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout2, stderr2 = await process2.communicate()
    if process2.returncode != 0:
        raise Exception(f"Falha no 'sentAlign.py': {stderr2.decode('utf-8', errors='ignore')}")

    return os.path.join(job_folder, "output", f"{file_name}.aligned")


async def _align_one_chapter(
        db: AsyncSession,
        nlp_en: stanza.Pipeline,
        nlp_pt: stanza.Pipeline,
        import_id: int,
        ch_src_href: str
) -> Dict[str, Any]:
    job_folder = ""
    try:
        # 1. Busca dados (Agora usando HREF como chave)
        mapa = await db.get(TmWinMapping, (import_id, ch_src_href))
        if not mapa:
            raise Exception(f"Capítulo '{ch_src_href}' não encontrado no mapa.")
        if not mapa.llm_verdict or mapa.ch_tgt is None:
            raise Exception(f"Capítulo '{ch_src_href}' não é um match válido.")

        mapa.micro_status = "processing"
        await db.commit()

        ch_tgt_href = str(mapa.ch_tgt)

        # Pega texto completo usando HREF
        texto_en = await _get_full_text_by_href(db, import_id, 'en', ch_src_href)
        texto_pt = await _get_full_text_by_href(db, import_id, 'pt', ch_tgt_href)

        if not texto_en or not texto_pt:
            raise Exception(f"Texto EN ou PT não encontrado.")

        logger.info(f"Fase 2: Segmentando '{ch_src_href}' com Stanza...")
        doc_en = nlp_en(texto_en)
        doc_pt = nlp_pt(texto_pt)

        sentencas_en_lista = [s.text.strip().replace("\n", " ") for s in doc_en.sentences if s.text.strip()]
        sentencas_pt_lista = [s.text.strip().replace("\n", " ") for s in doc_pt.sentences if s.text.strip()]

        if not sentencas_en_lista or not sentencas_pt_lista:
            raise Exception("Listas de sentenças vazias.")

        # Sandbox
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(TEMP_PROCESSING_DIR, job_id)
        eng_folder = os.path.join(job_folder, "eng")
        por_folder = os.path.join(job_folder, "por")
        os.makedirs(eng_folder, exist_ok=True)
        os.makedirs(por_folder, exist_ok=True)

        file_name = "chapter.txt"
        with open(os.path.join(eng_folder, file_name), 'w', encoding='utf-8') as f:
            f.write("\n".join(sentencas_en_lista))
        with open(os.path.join(por_folder, file_name), 'w', encoding='utf-8') as f:
            f.write("\n".join(sentencas_pt_lista))

        # Chama SentAlign
        output_file_path = await _run_sentalign_subprocess(job_folder, file_name)

        # Lê resultados
        pares_para_salvar: List[Dict[str, Any]] = []
        with open(output_file_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    en_line, pt_line, score_str = parts
                    try:
                        score = float(score_str)
                        if score >= MICRO_SIMILARITY_THRESHOLD:
                            pares_para_salvar.append({
                                "import_id": import_id,
                                "ch_src": ch_src_href,  # Salva HREF
                                "source_text": en_line,
                                "target_text": pt_line,
                                "similarity_score": score
                            })
                    except ValueError:
                        pass

        if pares_para_salvar:
            ins = pg_insert(TmAlignedSentences).values(pares_para_salvar)
            stmt = ins.on_conflict_do_nothing(
                index_elements=["import_id", "ch_src", "source_text", "target_text"]
            )
            await db.execute(stmt)

        mapa.micro_status = "aligned"
        await db.commit()
        return {"message": f"Capítulo '{ch_src_href}' alinhado.", "pares_salvos": len(pares_para_salvar)}

    except Exception as e:
        await db.rollback()
        logger.exception(f"ERRO CRÍTICO na Fase 2 para '{ch_src_href}': {e}")
        try:
            mapa = await db.get(TmWinMapping, (import_id, ch_src_href))
            if mapa:
                mapa.micro_status = "error"
                await db.commit()
        except:
            pass
        return {"error": str(e)}

    finally:
        if os.path.exists(job_folder):
            try:
                shutil.rmtree(job_folder)
            except:
                pass


# --- FUNÇÕES DE CONTROLE (2, 2C, 3, 4, 5) ---
# ... (Mantenha as funções run_sentence_alignment, run_alignment_for_all_pending,
# run_corpus_validation, export_corpus_to_tsv, get_corpus_status EXATAMENTE como estavam.
# Elas só chamam os helpers e não precisam mudar).
# (A única diferença é que agora 'ch_src' será um HREF, não um título, e tudo funcionará)

async def run_sentence_alignment(import_id: int, ch_src: str):
    logger.info(f"JOB DE ALINHAMENTO (FASE 2) INICIADO (import_id={import_id}, cap={ch_src})")
    try:
        # Carrega Stanza
        stanza.download('en', logging_level='WARN', processors='tokenize')
        stanza.download('pt', logging_level='WARN', processors='tokenize')
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True)
    except Exception as e:
        return {"error": f"Falha no Stanza: {e}"}

    async with AsyncSessionLocal() as session:
        return await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch_src)


# ... (Copie o resto do arquivo anterior, run_alignment_for_all_pending, etc.) ...
# ... (Não esqueça de colar o resto do arquivo!) ...
# ... (Vou colar aqui para garantir) ...

async def run_alignment_for_all_pending(import_id: int):
    logger.info(f"JOB DE ALINHAMENTO COMPLETO (FASE 2) INICIADO (import_id={import_id})")
    try:
        stanza.download('en', logging_level='WARN', processors='tokenize')
        stanza.download('pt', logging_level='WARN', processors='tokenize')
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True)
    except Exception as e:
        logger.exception(f"Fase 2 (Full): Falha no Stanza: {e}")
        return

    async with AsyncSessionLocal() as session:
        stmt = select(TmWinMapping.ch_src).where(
            TmWinMapping.import_id == import_id,
            TmWinMapping.micro_status == "pending"
        ).order_by(TmWinMapping.ch_src)
        lista = (await session.execute(stmt)).scalars().all()

        if not lista: return

        for i, ch_src in enumerate(lista):
            logger.info(f"Fase 2 (Full): {i + 1}/{len(lista)} - '{ch_src}'...")
            await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch_src)


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


async def export_corpus_to_tsv(import_id: int, min_score: float = 0.7, max_len_ratio: float = 3.0,
                               require_number_match: bool = True):
    logger.info(f"--- INICIANDO FASE 4 (EXPORTAÇÃO) ---")
    output_file = os.path.join(OUTPUT_DIR, f'{import_id}_corpus_premium.tsv')
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

async def _find_pending_macro_map_imports() -> List[int]:
    """
    Busca no 'tm_import' todos os IDs que ainda não
    têm nenhuma entrada no 'tm_win_mapping'.
    """
    logger.info("Fase 1 (Master): Buscando imports pendentes (que não estão no tm_win_mapping)...")
    async with AsyncSessionLocal() as session:
        # Subquery: Pega todos os import_ids que JÁ TÊM entradas no mapa
        sub_q = select(distinct(TmWinMapping.import_id)).subquery()

        # Query Principal: Pega todos os imports ONDE o ID NÃO ESTÁ na subquery
        stmt = select(models.Import.id).where(
            models.Import.id.notin_(
                select(sub_q)
            )
        ).order_by(models.Import.id)

        resultado = (await session.execute(stmt)).scalars().all()
        return list(resultado)


async def schedule_macro_map_for_all_pending():
    """
    Função pública "Mestra" que a API chama em background.
    Ela acha todos os IDs pendentes e roda a Fase 1 para cada um.
    """
    logger.info("JOB MESTRE (FASE 1) INICIADO: Processando todos os imports pendentes.")
    try:
        pending_ids = await _find_pending_macro_map_imports()
        if not pending_ids:
            logger.info("JOB MESTRE (FASE 1): Nenhum import pendente encontrado. Trabalho concluído.")
            return

        logger.info(f"JOB MESTRE (FASE 1): {len(pending_ids)} imports encontrados. Iniciando loop.")

        processados_com_sucesso = 0
        processados_com_erro = 0

        for import_id in pending_ids:
            logger.info(f"JOB MESTRE (FASE 1): Processando import_id={import_id}...")
            # Chama a função original (que abre sua própria sessão)
            result = await schedule_macro_map(import_id)
            if "error" in result:
                processados_com_erro += 1
            else:
                processados_com_sucesso += 1

        logger.info("JOB MESTRE (FASE 1): Todos os imports pendentes foram processados.")
        logger.info(f"Resultados: {processados_com_sucesso} com sucesso, {processados_com_erro} com erro.")

    except Exception as e:
        logger.exception(f"ERRO CRÍTICO no Job Mestre (Fase 1): {e}")


# --- ARQUIVO: services/corpus_builder_service.py ---

# (Certifique-se de que 'distinct' está importado do sqlalchemy lá no topo)
# from sqlalchemy import select, func, update, bindparam, cast, String, distinct

# --- FUNÇÃO PRIVADA (HELPER): Calcula status de UM livro ---
async def _calculate_stats_for_id(session: AsyncSession, import_id: int) -> Dict[str, Any]:
    """
    Esta função faz o trabalho pesado de contar capítulos e frases
    para UM import_id específico.
    """
    # 1. Contar Capítulos (na TmWinMapping)
    stmt_chapters = (
        select(TmWinMapping.micro_status, func.count(TmWinMapping.ch_src))
        .where(TmWinMapping.import_id == import_id, TmWinMapping.llm_verdict == True)
        .group_by(TmWinMapping.micro_status)
    )
    resultado_chapters = (await session.execute(stmt_chapters)).all()

    # Inicializa o dicionário zerado
    stats = {
        "status_capitulos": {
            "pending": 0, "aligned": 0, "error": 0, "processing": 0, "total_mapeado": 0
        },
        "status_frases": {
            "pending_validation": 0, "validated": 0, "total_alinhado": 0
        }
    }

    # Preenche com os resultados do banco
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

    for status, count in resultado_sentences:
        if status == "pending":
            stats["status_frases"]["pending_validation"] = count
        elif status == "validated":
            stats["status_frases"]["validated"] = count
        stats["status_frases"]["total_alinhado"] += count

    return {"import_id": import_id, "stats": stats}


# --- FUNÇÃO PÚBLICA 5 (ATUALIZADA): O STATUS INTELIGENTE ---
async def get_corpus_status(import_id: Optional[int] = None):
    """
    Se receber um ID: Retorna o status daquele ID.
    Se for None: Retorna uma LISTA com o status de TODOS os projetos ativos.
    """
    async with AsyncSessionLocal() as session:

        # CASO 1: Usuário pediu um ID específico
        if import_id is not None:
            dados = await _calculate_stats_for_id(session, import_id)
            return dados

        # CASO 2: Usuário quer ver TUDO (Sem ID)
        else:
            # Busca todos os IDs únicos que existem na tabela de mapeamento
            stmt_all_ids = select(distinct(TmWinMapping.import_id)).order_by(TmWinMapping.import_id)
            resultado_ids = (await session.execute(stmt_all_ids)).scalars().all()

            lista_de_status = []

            # Loop simples: calcula o status para cada ID encontrado
            for id_encontrado in resultado_ids:
                # Reutiliza a nossa função helper (código limpo!)
                dados_do_id = await _calculate_stats_for_id(session, id_encontrado)
                lista_de_status.append(dados_do_id)

            return lista_de_status


# --- ADICIONE NO FINAL DE 'services/corpus_builder_service.py' ---

async def _find_imports_with_pending_alignment() -> List[int]:
    """
    Busca todos os import_ids que têm pelo menos um capítulo
    com micro_status='pending' na tabela TmWinMapping.
    """
    logger.info("Fase 2 (Master): Buscando imports com alinhamento pendente...")
    async with AsyncSessionLocal() as session:
        stmt = select(distinct(TmWinMapping.import_id)).where(
            TmWinMapping.micro_status == "pending"
        ).order_by(TmWinMapping.import_id)

        resultado = (await session.execute(stmt)).scalars().all()
        return list(resultado)


async def run_master_alignment_job():
    """
    Função pública "Mestra" da FASE 2.
    Acha todos os livros pendentes e roda o alinhamento completo para cada um.
    """
    logger.info("JOB MESTRE (FASE 2) INICIADO: Processando todos os alinhamentos pendentes.")
    try:
        pending_ids = await _find_imports_with_pending_alignment()
        if not pending_ids:
            logger.info("JOB MESTRE (FASE 2): Nenhum alinhamento pendente encontrado. Trabalho concluído.")
            return

        logger.info(f"JOB MESTRE (FASE 2): {len(pending_ids)} livros encontrados. Iniciando fila.")

        for i, import_id in enumerate(pending_ids):
            logger.info(f"JOB MESTRE (FASE 2): Iniciando livro {i + 1}/{len(pending_ids)} (import_id={import_id})...")
            # Chama a função que já existe (ela cria sua própria sessão)
            await run_alignment_for_all_pending(import_id)

        logger.info("JOB MESTRE (FASE 2): Todos os livros foram processados.")

    except Exception as e:
        logger.exception(f"ERRO CRÍTICO no Job Mestre (Fase 2): {e}")


async def _find_imports_with_pending_validation() -> List[int]:
    """
    Busca todos os import_ids que têm pelo menos um par alinhado
    com validation_status='pending'.
    """
    logger.info("Fase 3 (Master): Buscando imports com validação pendente...")
    async with AsyncSessionLocal() as session:
        # Busca por IDs que têm entradas na tabela TmAlignedSentences
        # e que o status de validação ainda é 'pending'
        stmt = select(distinct(TmAlignedSentences.import_id)).where(
            TmAlignedSentences.validation_status == "pending"
        ).order_by(TmAlignedSentences.import_id)

        resultado = (await session.execute(stmt)).scalars().all()
        return list(resultado)


async def run_master_validation_job():
    """
    Função pública "Mestra" da FASE 3.
    Acha todos os livros com pares alinhados e os valida em sequência.
    """
    logger.info("JOB MESTRE (FASE 3) INICIADO: Processando todas as validações pendentes.")
    try:
        pending_ids = await _find_imports_with_pending_validation()
        if not pending_ids:
            logger.info("JOB MESTRE (FASE 3): Nenhum par pendente de validação encontrado. Trabalho concluído.")
            return

        logger.info(f"JOB MESTRE (FASE 3): {len(pending_ids)} livros com pares pendentes. Iniciando validação em loop.")

        for i, import_id in enumerate(pending_ids):
            logger.info(f"JOB MESTRE (FASE 3): Processando {i + 1}/{len(pending_ids)} (import_id={import_id})...")
            # Chama a função original (ela já é o nosso 'Botão 3')
            await run_corpus_validation(import_id)

        logger.info("JOB MESTRE (FASE 3): Todas as validações foram processadas.")

    except Exception as e:
        logger.exception(f"ERRO CRÍTICO no Job Mestre (Fase 3): {e}")