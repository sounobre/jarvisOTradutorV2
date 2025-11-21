# --- ARQUIVO ATUALIZADO: services/translation_pipeline.py ---
# ARQUITETURA FINAL: GGUF via llama-cpp-python (Estável e Rápido no Windows)
import os
# --- FORÇA A GPU NVIDIA ---
# Isso diz ao Python: "A única placa de vídeo que existe é a primeira placa CUDA que você achar".
# Como a AMD não tem CUDA, ele vai pegar a RTX 3060.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import re
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import asyncio

# --- NOVO MOTOR: llama_cpp ---
from llama_cpp import Llama
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import AsyncSessionLocal
from sqlalchemy import select
from db import models

logger = logging.getLogger(__name__)

# --- Constantes ---
REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
HTML_TAG_REGEX = re.compile(r'(<[^>]+>)')


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
        # Corrige espaçamento que o LLM possa ter adicionado
        translated_text = translated_text.replace(f" {placeholder}", placeholder)
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


# --- O CORAÇÃO DO SERVIÇO (LLAMA-CPP) ---

# --- SUBSTITUA NO services/translation_pipeline.py ---

async def translate_epub_book(import_id: int):
    logger.info(f"TRADUÇÃO GGUF INICIADA para import_id={import_id} (Backend: Llama.cpp).")

    temp_dir = None

    # 1. Carregar o Modelo
    try:
        logger.info(f"Carregando Mistral GGUF na GPU (Forçando 50 camadas)...")

        model = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            n_gpu_layers=-1,  # Volte para -1 (Tudo), agora que vamos liberar memória
            n_ctx=2048,  # <--- MUDANÇA 1: Reduzimos para liberar VRAM
            verbose=True,
            # Parâmetros extras para forçar GPU
            n_batch=512,  # Tamanho do lote de processamento
            offload_kqv=True  # <--- MUDANÇA 2: Força o cache KV para a GPU
        )
        logger.info(f"Modelo carregado! Verifique o log para 'offloaded layers'.")

    except Exception as e:
        logger.error(f"Falha ao carregar o modelo GGUF. Abortando. {e}")
        return

    # ... (O resto da função continua IGUAL: lógica de arquivos, loop, etc.)
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

            for i, file_path in enumerate(content_files):
                try:
                    raw_html = file_path.read_text(encoding='utf-8')
                    clean_text, placeholder_map = _split_and_tokenize_html(raw_html)

                    if len(clean_text.strip()) < 5:
                        continue

                    # Prompt
                    prompt = f"[INST] Translate the following text from English to Portuguese (Brazil). Maintain the formatting tags (like [TAG_0000]) exactly as they are. Do not add explanations. Text: \n\n{clean_text} [/INST]"

                    logger.info(f"Traduzindo {file_path.name} ({len(clean_text)} chars)...")

                    # Geração
                    output = model(
                        prompt,
                        max_tokens=4096,
                        temperature=0.1,
                        stop=["[/INST]"],
                        echo=False
                    )

                    translated_text = output['choices'][0]['text'].strip()

                    if "[/INST]" in translated_text:
                        translated_text = translated_text.split("[/INST]")[-1]

                    final_pt_html = _reconstitute_html(translated_text, placeholder_map)
                    file_path.write_text(final_pt_html, encoding='utf-8')

                except Exception as e:
                    logger.error(f"Erro no arquivo {file_path.name}: {e}")
                    continue

            output_dir = Path("data/translated")
            output_dir.mkdir(exist_ok=True)
            final_epub_name = f"{book_name}_PT_MISTRAL.epub"
            output_path = output_dir / final_epub_name

            _create_output_epub(temp_dir, output_path)
            logger.info(f"SUCESSO! EPUB FINAL: {output_path}")

        except Exception as e:
            logger.exception(f"Erro no pipeline: {e}")
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)