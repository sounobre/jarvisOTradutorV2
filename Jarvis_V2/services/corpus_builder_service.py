# --- ARQUIVO ATUALIZADO (O "MOTOR"): services/corpus_builder_service.py ---

"""
Este é o "Motor" (Engine) do Pipeline "Embeddings-Only".
*** ARQUITETURA FINAL: 'Stanza' (Segmenta) + 'SentAlign' (Alinha) ***
*** ATUALIZAÇÃO CRÍTICA: A "Chave" (ch_idx) é o Href, não o Title. ***
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

# --- Importações do Projeto ---
from db.session import AsyncSessionLocal
from db.models import TmWinMapping, ChapterText, ChapterIndex, TmAlignedSentences
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select, func, update, bindparam, cast, String

# --- Importações de ML/Processamento ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- ☢️☢️☢️ CONFIGURAÇÃO CRÍTICA (SentAlign) ☢️☢️☢️ ---
SENTALIGN_PYTHON_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\.venv\Scripts\python.exe"
SENTALIGN_ROOT_FOLDER = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\sentalign\pythonProject\SentAlign"
TEMP_PROCESSING_DIR = os.path.join(os.getcwd(), "temp_processing")
# --- FIM DA CONFIGURAÇÃO ---

# --- Constantes ---
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
MACRO_SIMILARITY_THRESHOLD = 0.5  # Limite para TÍTULOS
MICRO_SIMILARITY_THRESHOLD = 0.0  # Limite para SENTENÇAS (no SentAlign)
OUTPUT_DIR = 'data'
GPU_BATCH_SIZE = 256

logger = logging.getLogger(__name__)


# --- Funções "Privadas" (Helpers) ---

async def _get_titles_and_hrefs(db: AsyncSession, import_id: int, lang: str) -> Dict[str, str]:
    """
    NOVO HELPER: Busca o mapa de {href: title} da ChapterIndex.
    """
    logger.info(f"Fase 1: Buscando Títulos e Hrefs de 'ChapterIndex' para '{lang}'...")
    stmt = select(
        ChapterIndex.ch_idx,  # O Href (nossa chave)
        ChapterIndex.title  # O Título (nosso "DNA")
    ).where(
        ChapterIndex.import_id == import_id,
        ChapterIndex.lang == lang,
        ChapterIndex.title.is_not(None)
    ).order_by(ChapterIndex.ch_idx)

    resultado = (await db.execute(stmt)).all()
    mapa_href_titulo: Dict[str, str] = {str(href): str(title) for href, title in resultado}
    return mapa_href_titulo


async def _get_full_text_by_href(db: AsyncSession, import_id: int, lang: str, ch_idx_href: str) -> Optional[str]:
    """
    HELPER ATUALIZADO: Busca o texto usando o Href (que agora é o ch_idx).
    """
    stmt = select(ChapterText.text).where(
        ChapterText.import_id == import_id,
        ChapterText.lang == lang,
        ChapterText.ch_idx == ch_idx_href  # ch_idx AGORA É O HREF
    )
    txt = (await db.execute(stmt)).scalar_one_or_none()

    if not txt:
        logger.warning(f"Texto não encontrado para import_id={import_id}, lang={lang}, href='{ch_idx_href}'")
        return None
    return txt


async def _upsert_many_mappings(db: AsyncSession, values_list: List[Dict[str, Any]]):
    # (Esta função fica 100% IGUAL, ela só salva o que recebe)
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


# --- FUNÇÃO PÚBLICA 1: O AGENDADOR (Fase 1) ---
# --- (Totalmente Refatorada) ---
async def schedule_macro_map(import_id: int):
    logger.info(f"JOB DE AGENDAMENTO (FASE 1) INICIADO (import_id={import_id})")
    async with AsyncSessionLocal() as session:
        try:
            logger.info(f"Fase 1: Carregando modelo embedding: '{MODELO_EMBEDDING}'...")
            model = SentenceTransformer(MODELO_EMBEDDING, device='cuda')

            # 1. Busca os mapas de {href: title}
            en_map = await _get_titles_and_hrefs(session, import_id, 'en')
            pt_map = await _get_titles_and_hrefs(session, import_id, 'pt')

            if not en_map or not pt_map:
                raise Exception(f"Fase 1: Faltam dados em ChapterIndex (EN ou PT).")

            # 2. Prepara as listas para o "DNA"
            # Nossas "chaves" (índices) agora são os HREFS
            en_hrefs: List[str] = list(en_map.keys())
            en_corpus_titulos: List[str] = list(en_map.values())  # O "DNA" vem dos Títulos

            pt_hrefs: List[str] = list(pt_map.keys())
            pt_corpus_titulos: List[str] = list(pt_map.values())  # O "DNA" vem dos Títulos

            # 3. Gerar Embeddings (Macro) dos TÍTULOS
            logger.info(f"Fase 1: Gerando embeddings (Macro) dos TÍTULOS...")
            en_embeddings = model.encode(en_corpus_titulos, show_progress_bar=False, batch_size=GPU_BATCH_SIZE)
            pt_embeddings = model.encode(pt_corpus_titulos, show_progress_bar=False, batch_size=GPU_BATCH_SIZE)

            logger.info("Fase 1: Calculando similaridade (Macro)...")
            sim_matrix = cosine_similarity(en_embeddings, pt_embeddings)

            # 4. Achar os pares e salvar
            linhas_para_salvar_no_db: List[Dict[str, Any]] = []
            for i in range(len(en_hrefs)):
                en_href_chave = en_hrefs[i]  # A chave (ex: "chap1.xhtml")

                best_pt_j = sim_matrix[i].argmax()
                best_score = float(sim_matrix[i][best_pt_j])
                pt_href_chave = pt_hrefs[best_pt_j]  # A chave PT (ex: "cap1.xhtml")

                foi_match = best_score >= MACRO_SIMILARITY_THRESHOLD

                # (Não podemos calcular len_src/len_tgt aqui, pois só temos os títulos)
                # (Vamos deixar o 'len_src/len_tgt' zerado por enquanto)
                linhas_para_salvar_no_db.append({
                    "import_id": import_id,
                    "ch_src": en_href_chave,  # SALVA O HREF
                    "ch_tgt": pt_href_chave if foi_match else None,  # SALVA O HREF
                    "sim_cosine": best_score,
                    "len_src": 0,  # (Não temos o texto completo aqui)
                    "len_tgt": 0,  # (Não temos o texto completo aqui)
                    "method": "embedding_titles",
                    "llm_score": best_score,
                    "llm_verdict": foi_match,
                    "llm_reason": f"Title Similarity: {best_score:.4f}",
                    "score": best_score,
                    "micro_status": "pending" if foi_match else "skipped",
                })

            await _upsert_many_mappings(session, linhas_para_salvar_no_db)
            await session.commit()
            logger.info(f"--- FASE 1 (MACRO-MAP) CONCLUÍDA ---")
            return {"message": "Mapeamento macro (Hrefs por Títulos) concluído.",
                    "capitulos_mapeados": len(linhas_para_salvar_no_db)}

        except Exception as e:
            await session.rollback()
            logger.exception(f"ERRO CRÍTICO na Fase 1 para import_id={import_id}: {e}")
            return {"error": str(e)}


# --- "MIOLO" DA FASE 2 (Stanza + "Gambiarra" SentAlign) ---
async def _run_sentalign_subprocess(job_folder: str, file_name: str) -> str:
    """
    Função helper que chama o sentAlign.py externo.
    """
    logger.info("Fase 2 (Subprocess): Chamando 'files2align.py'...")

    cmd_files2align = [
        SENTALIGN_PYTHON_PATH,
        os.path.join(SENTALIGN_ROOT_FOLDER, "files2align.py"),
        "-dir", job_folder,
        "--source-language", "eng"
    ]

    process1 = await asyncio.create_subprocess_exec(
        *cmd_files2align,
        cwd=SENTALIGN_ROOT_FOLDER,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout1, stderr1 = await process1.communicate()
    if process1.returncode != 0:
        raise Exception(f"Falha no 'files2align.py': {stderr1.decode('utf-8', errors='ignore')}")

    logger.info("Fase 2 (Subprocess): Chamando 'sentAlign.py' (na 3060)...")

    cmd_sentalign = [
        SENTALIGN_PYTHON_PATH,
        os.path.join(SENTALIGN_ROOT_FOLDER, "sentAlign.py"),
        "-dir", job_folder,
        "-sl", "eng",
        "-tl", "por",
        "--proc-device", "cuda"
    ]

    process2 = await asyncio.create_subprocess_exec(
        *cmd_sentalign,
        cwd=SENTALIGN_ROOT_FOLDER,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout2, stderr2 = await process2.communicate()
    if process2.returncode != 0:
        raise Exception(f"Falha no 'sentAlign.py': {stderr2.decode('utf-8', errors='ignore')}")

    logger.info(f"Fase 2 (Subprocess): 'sentAlign.py' concluído. {stdout2.decode('utf-8', errors='ignore')}")

    return os.path.join(job_folder, "output", f"{file_name}.aligned")


async def _align_one_chapter(
        db: AsyncSession,
        nlp_en: stanza.Pipeline,  # Modelo de Segmentação EN
        nlp_pt: stanza.Pipeline,  # Modelo de Segmentação PT
        import_id: int,
        ch_src_href: str  # <-- Agora recebemos o HREF
) -> Dict[str, Any]:
    """
    Esta é a lógica interna da Fase 2.
    *** ATUALIZADO: Recebe Hrefs (ch_idx) e chama o SentAlign ***
    """

    job_folder = ""

    try:
        # 1. Buscar os dados do capítulo
        mapa = await db.get(TmWinMapping, (import_id, ch_src_href))
        if not mapa:
            raise Exception(f"Capítulo (href) '{ch_src_href}' não encontrado no mapa.")
        if not mapa.llm_verdict or mapa.ch_tgt is None:
            raise Exception(f"Capítulo '{ch_src_href}' não é um match válido.")

        mapa.micro_status = "processing"
        await db.commit()

        ch_tgt_href = str(mapa.ch_tgt)

        # Busca os textos usando os HREFS (que são o ch_idx)
        texto_en = await _get_full_text_by_href(db, import_id, 'en', ch_src_href)
        texto_pt = await _get_full_text_by_href(db, import_id, 'pt', ch_tgt_href)

        if not texto_en or not texto_pt:
            raise Exception(f"Texto EN ou PT não encontrado para o par '{ch_src_href}' -> '{ch_tgt_href}'.")

        # 2. Segmentar (Stanza)
        logger.info(f"Fase 2: Segmentando '{ch_src_href}' com Stanza (na GPU)...")
        doc_en = nlp_en(texto_en)
        doc_pt = nlp_pt(texto_pt)

        sentencas_en_lista = [s.text.strip().replace("\n", " ") for s in doc_en.sentences if s.text.strip()]
        sentencas_pt_lista = [s.text.strip().replace("\n", " ") for s in doc_pt.sentences if s.text.strip()]

        sentencas_en_lista = [s for s in sentencas_en_lista if s]
        sentencas_pt_lista = [s for s in sentencas_pt_lista if s]

        if not sentencas_en_lista or not sentencas_pt_lista:
            raise Exception("Listas de sentenças estão vazias.")

        logger.info(
            f"Fase 2: Segmentado (com Stanza) '{ch_src_href}'. {len(sentencas_en_lista)} EN vs {len(sentencas_pt_lista)} PT.")

        # 3. Criar a "sandbox" (arquivos temporários)
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(TEMP_PROCESSING_DIR, job_id)
        eng_folder = os.path.join(job_folder, "eng")
        por_folder = os.path.join(job_folder, "por")
        os.makedirs(eng_folder, exist_ok=True)
        os.makedirs(por_folder, exist_ok=True)

        file_name = "chapter.txt"
        eng_path = os.path.join(eng_folder, file_name)
        por_path = os.path.join(por_folder, file_name)

        with open(eng_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(sentencas_en_lista))
        with open(por_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(sentencas_pt_lista))

        logger.info(f"Fase 2: Arquivos de SENTENÇAS temporários criados em {job_folder}")

        # 4. Chamar o Subprocess (SentAlign)
        output_file_path = await _run_sentalign_subprocess(job_folder, file_name)

        # 5. Ler o resultado
        logger.info(f"Fase 2: Lendo resultado de {output_file_path}")
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
                                "ch_src": ch_src_href,  # Salva o Href
                                "source_text": en_line,
                                "target_text": pt_line,
                                "similarity_score": score
                            })
                    except ValueError:
                        pass

        logger.info(f"Fase 2: Encontrados {len(pares_para_salvar)} pares (acima de {MICRO_SIMILARITY_THRESHOLD}).")

        # 6. Salvar os pares no banco
        if pares_para_salvar:
            ins = pg_insert(TmAlignedSentences).values(pares_para_salvar)
            stmt = ins.on_conflict_do_nothing(
                index_elements=["import_id", "ch_src", "source_text", "target_text"]
            )
            await db.execute(stmt)

        # 7. ATUALIZAR O STATUS final
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
        # 8. LIMPAR
        if os.path.exists(job_folder):
            try:
                shutil.rmtree(job_folder)
                logger.info(f"Fase 2: Pasta temporária {job_folder} limpa.")
            except Exception as e:
                logger.error(f"Fase 2: Falha ao limpar pasta {job_folder}: {e}")


# --- FUNÇÃO PÚBLICA 2: O ALINHADOR DE 1 CAPÍTULO (Fase 2) ---
async def run_sentence_alignment(import_id: int, ch_src: str):
    logger.info(f"JOB DE ALINHAMENTO (FASE 2) INICIADO (import_id={import_id}, cap_href={ch_src})")

    # 1. Carregar (apenas) o Stanza
    try:
        logger.info("Fase 2: Carregando modelos de segmentação Stanza (pode baixar na 1ª vez)...")
        stanza.download('en', logging_level='WARN', processors='tokenize')
        stanza.download('pt', logging_level='WARN', processors='tokenize')
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True)

    except Exception as e:
        logger.exception(f"Fase 2: Falha ao carregar modelos Stanza. Abortando. {e}")
        return {"error": f"Falha ao carregar modelos Stanza: {e}"}

    # 2. Abrir sessão e chamar o "miolo"
    async with AsyncSessionLocal() as session:
        result = await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch_src)
        return result


# --- FUNÇÃO PÚBLICA 2C: O ALINHADOR DE TUDO (Fase 2 - Loop) ---
async def run_alignment_for_all_pending(import_id: int):
    logger.info(f"JOB DE ALINHAMENTO COMPLETO (FASE 2) INICIADO (import_id={import_id})")

    # 1. Carregar (apenas) o Stanza (uma vez)
    try:
        logger.info("Fase 2 (Full): Carregando modelos de segmentação Stanza...")
        stanza.download('en', logging_level='WARN', processors='tokenize')
        stanza.download('pt', logging_level='WARN', processors='tokenize')
        nlp_en = stanza.Pipeline('en', processors='tokenize', logging_level='WARN', use_gpu=True)
        nlp_pt = stanza.Pipeline('pt', processors='tokenize', logging_level='WARN', use_gpu=True)
    except Exception as e:
        logger.exception(f"Fase 2 (Full): Falha ao carregar os modelos Stanza. Abortando. {e}")
        return

    async with AsyncSessionLocal() as session:
        # 2. Buscar a lista COMPLETA de capítulos pendentes
        stmt = select(TmWinMapping.ch_src).where(
            TmWinMapping.import_id == import_id,
            TmWinMapping.micro_status == "pending"
        ).order_by(TmWinMapping.ch_src)

        lista_de_capitulos_pendentes = (await session.execute(stmt)).scalars().all()
        total_capitulos = len(lista_de_capitulos_pendentes)
        logger.info(f"Fase 2 (Full): Encontrados {total_capitulos} capítulos pendentes.")

        if total_capitulos == 0:
            logger.info("Fase 2 (Full): Nenhum capítulo pendente. Trabalho concluído.")
            return

        # 3. Loop de processamento
        processados_com_sucesso = 0
        processados_com_erro = 0
        for i, ch_src in enumerate(lista_de_capitulos_pendentes):
            logger.info(f"Fase 2 (Full): Processando {i + 1}/{total_capitulos} - '{ch_src}'...")

            result = await _align_one_chapter(session, nlp_en, nlp_pt, import_id, ch_src)

            if "error" in result:
                processados_com_erro += 1
            else:
                processados_com_sucesso += 1

    logger.info(f"--- FASE 2 (ALINHAMENTO COMPLETO) CONCLUÍDA ---")
    logger.info(f"Resultados: {processados_com_sucesso} com sucesso, {processados_com_erro} com erro.")


# --- (FUNÇÃO PÚBLICA 3: O VALIDADOR (Fase 3) - Fica 100% IGUAL) ---
async def run_corpus_validation(import_id: int, batch_size: int = 5000):
    logger.info(f"--- INICIANDO FASE 3 (VALIDAÇÃO) PARA IMPORT ID: {import_id} ---")

    async with AsyncSessionLocal() as session:
        try:
            total_validado = 0
            while True:
                stmt = (
                    select(TmAlignedSentences)
                    .where(
                        TmAlignedSentences.import_id == import_id,
                        TmAlignedSentences.validation_status == "pending"
                    )
                    .limit(batch_size)
                )
                pares_para_validar = (await session.execute(stmt)).scalars().all()

                if not pares_para_validar:
                    logger.info("Fase 3: Nenhum par pendente de validação encontrado.")
                    break

                logger.info(f"Fase 3: Validando um lote de {len(pares_para_validar)} pares...")

                for par in pares_para_validar:
                    len_en = len(par.source_text)
                    len_pt = len(par.target_text)
                    par.len_ratio = len_pt / max(1, len_en)

                    nums_en = _get_numbers_from_text(par.source_text)
                    nums_pt = _get_numbers_from_text(par.target_text)
                    par.number_mismatch = (nums_en != nums_pt)

                    par.validation_status = "validated"

                await session.commit()
                total_validado += len(pares_para_validar)
                logger.info(f"Fase 3: Lote salvo. Total validado até agora: {total_validado}")

            logger.info(f"--- FASE 3 (VALIDAÇÃO) CONCLUÍDA ---")
            return {"message": "Validação concluída.", "total_pares_validados": total_validado}

        except Exception as e:
            await session.rollback()
            logger.exception(f"ERRO CRÍTICO na Fase 3 (Validação) para import_id={import_id}: {e}")
            return {"error": str(e)}


# --- (FUNÇÃO PÚBLICA 4: O EXPORTADOR (Fase 4) - Fica 100% IGUAL) ---
async def export_corpus_to_tsv(
        import_id: int,
        min_score: float = 0.7,
        max_len_ratio: float = 3.0,
        require_number_match: bool = True
):
    logger.info(f"--- INICIANDO FASE 4 (EXPORTAÇÃO) PARA IMPORT ID: {import_id} ---")
    logger.info(
        f"Filtros de Qualidade: score >= {min_score}, len_ratio <= {max_len_ratio}, number_match={require_number_match}")

    output_file = os.path.join(OUTPUT_DIR, f'{import_id}_corpus_premium.tsv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async with AsyncSessionLocal() as session:
        try:
            stmt = (
                select(TmAlignedSentences.source_text, TmAlignedSentences.target_text)
                .where(
                    TmAlignedSentences.import_id == import_id,
                    TmAlignedSentences.validation_status == "validated",
                    TmAlignedSentences.similarity_score >= min_score,
                    TmAlignedSentences.len_ratio <= max_len_ratio
                )
            )

            if require_number_match:
                stmt = stmt.where(TmAlignedSentences.number_mismatch == False)

            stmt = stmt.order_by(TmAlignedSentences.ch_src, TmAlignedSentences.id)

            pares = (await session.execute(stmt)).all()

            if not pares:
                logger.warning(f"Fase 4: Nenhum par alinhado passou nos filtros de qualidade.")
                return {"error": "Nenhum par 'premium' encontrado."}

            with open(output_file, 'w', encoding='utf-8') as f_out:
                for en_line, pt_line in pares:
                    f_out.write(f"{en_line}\t{pt_line}\n")

            logger.info(f"--- FASE 4 (EXPORTAÇÃO) CONCLUÍDA ---")
            return {"message": "Exportação 'Premium' concluída.", "output_file": output_file,
                    "total_pares_premium": len(pares)}

        except Exception as e:
            logger.exception(f"ERRO CRÍTICO na Fase 4 (Exportação) para import_id={import_id}: {e}")
            return {"error": str(e)}


# --- (FUNÇÃO PÚBLICA 5: O STATUS - Fica 100% IGUAL) ---
async def get_corpus_status(import_id: int):
    async with AsyncSessionLocal() as session:

        # Query 1: Status dos Capítulos
        stmt_chapters = (
            select(TmWinMapping.micro_status, func.count(TmWinMapping.ch_src))
            .where(TmWinMapping.import_id == import_id, TmWinMapping.llm_verdict == True)
            .group_by(TmWinMapping.micro_status)
        )
        resultado_chapters = (await session.execute(stmt_chapters)).all()
        stats = {
            "status_capitulos": {
                "pending": 0, "aligned": 0, "error": 0, "processing": 0, "total_mapeado": 0
            },
            "status_frases": {
                "pending_validation": 0, "validated": 0, "total_alinhado": 0
            }
        }
        for status, count in resultado_chapters:
            if status in stats["status_capitulos"]:
                stats["status_capitulos"][status] = count
            stats["status_capitulos"]["total_mapeado"] += count

        # Query 2: Status da Validação de Frases
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

        return {"import_id": import_id, "status": stats}