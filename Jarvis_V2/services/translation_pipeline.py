# --- ARQUIVO ATUALIZADO: services/translation_pipeline.py ---
# COM SISTEMA DE LOGS (TELEMETRIA)

import logging
import os
import re
import shutil
import zipfile
import tempfile
import time  # <-- Importante para cronometrar
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio

# --- MOTOR: llama_cpp ---
from llama_cpp import Llama
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import AsyncSessionLocal

from sqlalchemy import select, insert
from db import models

logger = logging.getLogger(__name__)

# --- Constantes ---
REPO_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-7b-instruct-q5_k_m.gguf"
HTML_TAG_REGEX = re.compile(r'(<[^>]+>)')

# Configurações de Geração (Salvas no Log)
GEN_CTX = 4096
GEN_TEMP = 0.3
GEN_MAX_TOKENS = 2048
MAX_CHUNK_SIZE = 1500


# --- Funções de Preparação (IGUAIS) ---

async def _get_import_paths(session: AsyncSession, import_id: int) -> Optional[Tuple[Path, str]]:
    stmt = select(models.Import.file_en, models.Import.name).where(models.Import.id == import_id)
    result = await session.execute(stmt)
    return result.one_or_none()


def _split_and_tokenize_html(html_chunk: str) -> Tuple[str, Dict[str, str]]:
    tags_list = HTML_TAG_REGEX.findall(html_chunk)
    if not tags_list:
        return html_chunk, {}
    parts = HTML_TAG_REGEX.split(html_chunk)
    clean_text_parts = []
    placeholder_map = {}
    tag_counter = 0
    for part in parts:
        if part in tags_list:
            placeholder = f"[TAG_{tag_counter:04d}]"
            placeholder_map[placeholder] = part
            clean_text_parts.append(placeholder)
            tag_counter += 1
        else:
            clean_text_parts.append(part)
    clean_text = "".join(clean_text_parts)
    return clean_text, placeholder_map


def _reconstitute_html(translated_text: str, placeholder_map: Dict[str, str]) -> str:
    if not placeholder_map:
        return translated_text
    for placeholder, original_tag in placeholder_map.items():
        translated_text = translated_text.replace(f" {placeholder}", placeholder)
        translated_text = translated_text.replace(f"{placeholder} ", placeholder)
        translated_text = translated_text.replace(placeholder, original_tag)
    return translated_text


def _create_output_epub(source_dir: Path, output_path: Path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(source_dir)
                zf.write(file_path, rel_path)
    logger.info(f"EPUB reconstruído com sucesso em: {output_path}")


def _smart_chunking(text: str, max_chars: int) -> List[str]:
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    for para in paragraphs:
        if len(para) > max_chars:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(para)
            continue
        if current_length + len(para) > max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += len(para) + 1
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


# --- NOVO HELPER: Salvar Log ---
async def _log_translation_event(
        session: AsyncSession,
        import_id: int,
        file_name: str,
        chunk_idx: int,
        source: str,
        target: str,
        prompt: str,
        duration: float
):
    """Salva os metadados da tradução no banco para análise futura."""
    try:
        await session.execute(
            insert(models.TmTranslationLog).values(
                import_id=import_id,
                file_name=file_name,
                chunk_index=chunk_idx,
                source_text=source,
                translated_text=target,
                model_name=REPO_ID,
                prompt_snapshot=prompt,
                temperature=GEN_TEMP,
                duration_seconds=duration
            )
        )
        # Não fazemos commit aqui para não desacelerar o loop principal demais,
        # o commit acontece junto com o update do capítulo ou periodicamente.
    except Exception as e:
        logger.error(f"Falha ao salvar log de tradução: {e}")


# --- O CORAÇÃO DO SERVIÇO (QWEN 2.5 + TELEMETRIA) ---

async def translate_epub_book(import_id: int):
    logger.info(f"TRADUÇÃO GGUF INICIADA para import_id={import_id} (Modelo: {REPO_ID}).")

    temp_dir = None

    # 1. Carregar o Modelo
    try:
        logger.info(f"Carregando Modelo na GPU...")
        model = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            n_gpu_layers=-1,
            n_ctx=GEN_CTX,
            verbose=True
        )
        logger.info(f"Modelo carregado com sucesso na GPU!")

    except Exception as e:
        logger.error(f"Falha ao carregar o modelo GGUF. Abortando. {e}")
        return

    # 2. Lógica de Arquivos
    async with AsyncSessionLocal() as session:
        paths = await _get_import_paths(session, import_id)
        if not paths: return
        original_path, book_name = paths
        original_path = Path(original_path)
        if not original_path.exists():
            logger.error(f"Arquivo EPUB não encontrado: {original_path}")
            return

        try:
            temp_dir = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(original_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            logger.info(f"EPUB descompactado em: {temp_dir}")
            content_files = list(temp_dir.rglob('*.xhtml')) + list(temp_dir.rglob('*.html'))

            # 3. Loop de Tradução
            for i, file_path in enumerate(content_files):
                try:
                    raw_html = file_path.read_text(encoding='utf-8')
                    clean_text, placeholder_map = _split_and_tokenize_html(raw_html)

                    if len(clean_text.strip()) < 5:
                        continue

                    logger.info(f"Processando {file_path.name} ({len(clean_text)} chars)...")

                    text_chunks = _smart_chunking(clean_text, MAX_CHUNK_SIZE)
                    translated_chunks = []

                    for j, chunk in enumerate(text_chunks):
                        # --- PROMPT (Com Injeção de Glossário futuramente) ---
                        prompt = f"""<|im_start|>system
You are a professional literary translator specializing in High Fantasy novels. 
Translate the text below from English to Portuguese (Brazil).

Guidelines:
1. Maintain the tone, style, and atmosphere of the original text.
2. Keep all formatting tags (like [TAG_0000]) EXACTLY where they are.
3. Translate idioms and cultural references naturally for a Brazilian audience.
4. Do NOT provide explanations. Output ONLY the translation.
<|im_end|>
<|im_start|>user
{chunk}<|im_end|>
<|im_start|>assistant
"""

                        # --- GERAÇÃO COM TIMER ---
                        start_time = time.time()

                        output = model(
                            prompt,
                            max_tokens=GEN_MAX_TOKENS,
                            temperature=GEN_TEMP,
                            stop=["<|im_end|>"],
                            echo=False
                        )

                        end_time = time.time()
                        duration = end_time - start_time

                        chunk_translation = output['choices'][0]['text'].strip()
                        translated_chunks.append(chunk_translation)

                        # --- SALVA O LOG (Telemetria) ---
                        await _log_translation_event(
                            session, import_id, file_path.name, j,
                            chunk, chunk_translation, prompt, duration
                        )

                        if (j + 1) % 5 == 0:
                            logger.info(f"  -> Chunk {j + 1}/{len(text_chunks)} traduzido ({duration:.2f}s).")

                    # Commit dos logs deste capítulo
                    await session.commit()

                    full_translated_text = "\n".join(translated_chunks)
                    final_pt_html = _reconstitute_html(full_translated_text, placeholder_map)

                    file_path.write_text(final_pt_html, encoding='utf-8')
                    logger.info(f"[{i + 1}/{len(content_files)}] Arquivo concluído: {file_path.name}")

                except Exception as e:
                    logger.error(f"Erro no arquivo {file_path.name}: {e}")
                    continue

                    # 4. Recompactação
            output_dir = Path("data/translated")
            output_dir.mkdir(exist_ok=True)
            final_epub_name = f"{book_name}_PT_QWEN.epub"
            output_path = output_dir / final_epub_name

            _create_output_epub(temp_dir, output_path)
            logger.info(f"SUCESSO! EPUB FINAL: {output_path}")

        except Exception as e:
            logger.exception(f"Erro no pipeline: {e}")
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)