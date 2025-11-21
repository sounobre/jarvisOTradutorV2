# --- ARQUIVO ATUALIZADO: services/translation_pipeline.py ---
# ARQUITETURA: GGUF via llama-cpp-python
# MODELO: Qwen 2.5 7B Instruct (Otimizado para Fantasia)

import logging
import os
import re
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import torch

# Importamos o carregador GGUF
from llama_cpp import Llama
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoTokenizer

from db.session import AsyncSessionLocal
from sqlalchemy import select
from db import models

logger = logging.getLogger(__name__)

# --- Constantes do Modelo (Qwen 2.5) ---
REPO_ID = "bartowski/Qwen2.5-7B-Instruct-GGUF"
# Usamos Q5_K_M para um equilíbrio perfeito entre qualidade e velocidade na 3060
MODEL_FILE = "Qwen2.5-7B-Instruct-Q5_K_M.gguf"

HTML_TAG_REGEX = re.compile(r'(<[^>]+>)')
MAX_CHUNK_SIZE = 1500  # Reduzi um pouco para dar margem ao Qwen raciocinar


# --- Funções de Preparação ---

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
        # Remove espaços extras que o modelo possa ter colocado em volta da tag
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


async def _fetch_glossary_string(session: AsyncSession, import_id: int) -> str:
    """
    Busca o glossário no banco e formata como uma string de regras.
    Retorna vazio se não houver glossário.
    """
    stmt = select(models.TmGlossary).where(models.TmGlossary.import_id == import_id)
    terms = (await session.execute(stmt)).scalars().all()

    if not terms:
        return ""

    # Formata para o LLM entender claramente
    glossary_lines = ["GLOSSARY (Mandatory Terminology):"]
    for term in terms:
        glossary_lines.append(f"- '{term.term_source}' must be translated as '{term.term_target}'")

    return "\n".join(glossary_lines)
# --- O CORAÇÃO DO SERVIÇO (QWEN 2.5) ---

async def translate_epub_book(import_id: int):
    logger.info(f"TRADUÇÃO GGUF INICIADA para import_id={import_id} (Modelo: Qwen 2.5 7B).")

    temp_dir = None

    # 1. Carregar o Modelo
    try:
        logger.info(f"Carregando Qwen 2.5 GGUF na GPU...")

        model = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            n_gpu_layers=-1,  # Tenta jogar tudo pra GPU (sua 3060 aguenta)
            n_ctx=4096,  # Contexto suficiente para chunks de 1500 chars
            verbose=True  # Mantenha True para ver o 'offloaded layers'
        )
        logger.info(f"Modelo carregado com sucesso na GPU!")

    except Exception as e:
        logger.error(f"Falha ao carregar o modelo GGUF. Abortando. {e}")
        return

    # 2. Lógica de Arquivos
    async with AsyncSessionLocal() as session:

        glossary_instruction = await _fetch_glossary_string(session, import_id)
        if glossary_instruction:
            logger.info(f"Glossário encontrado e injetado com {glossary_instruction.count('-')} termos.")

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
                        # --- PROMPT ESPECIALIZADO PARA QWEN (ChatML) ---
                        # O Qwen adora 'system prompts' detalhados.
                        prompt = f"""<|im_start|>system
You are a professional literary translator specializing in High Fantasy novels. 
Translate the text below from English to Portuguese (Brazil).

Guidelines:
1. Maintain the tone, style, and atmosphere of the original text.
2. Keep all formatting tags (like [TAG_0000]) EXACTLY where they are in the structure.
3. Translate idioms and cultural references naturally for a Brazilian audience.
4. Do NOT provide explanations, notes, or introductory text. Output ONLY the translation.
<|im_end|>
<|im_start|>user
{chunk}<|im_end|>
<|im_start|>assistant
"""

                        # Geração
                        output = model(
                            prompt,
                            max_tokens=2048,
                            temperature=0.3,  # Um pouco mais de criatividade que o 0.1, bom para literatura
                            stop=["<|im_end|>"],  # Parada correta do Qwen
                            echo=False
                        )

                        chunk_translation = output['choices'][0]['text'].strip()
                        translated_chunks.append(chunk_translation)

                        if (j + 1) % 5 == 0:
                            logger.info(f"  -> Chunk {j + 1}/{len(text_chunks)} traduzido.")

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