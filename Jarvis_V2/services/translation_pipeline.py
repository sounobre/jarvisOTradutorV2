# --- ARQUIVO CORRIGIDO: services/translation_pipeline.py ---


import logging
import os
import re
import shutil
import zipfile
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import AsyncSessionLocal
from db import models

logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO ---
# Se o Jarvis estiver em outra porta, mude aqui (ex: 8001 se rodar junto)
JARVIS_API_URL = "http://localhost:8001/translate"
MAX_CHUNK_SIZE = 1500
HTML_TAG_REGEX = re.compile(r'(<[^>]+>)')


# --- Funções Auxiliares (Mantidas) ---
async def _get_import_paths(session: AsyncSession, import_id: int) -> Optional[Tuple[Path, str]]:
    stmt = select(models.Import.file_en, models.Import.name).where(models.Import.id == import_id)
    result = await session.execute(stmt)
    return result.one_or_none()


def _split_and_tokenize_html(html_chunk: str) -> Tuple[str, Dict[str, str]]:
    tags_list = HTML_TAG_REGEX.findall(html_chunk)
    if not tags_list: return html_chunk, {}
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
    return "".join(clean_text_parts), placeholder_map


def _reconstitute_html(translated_text: str, placeholder_map: Dict[str, str]) -> str:
    if not placeholder_map: return translated_text
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


async def _log_translation_event(session: AsyncSession, import_id: int, file_name: str, chunk_idx: int, source: str, target: str, duration: float):
    try:
        # Prompt genérico para não quebrar o NOT NULL
        dummy_prompt = "REMOTE_INFERENCE_API_CALL"

        await session.execute(
            insert(models.TmTranslationLog).values(
                import_id=import_id,
                file_name=file_name,
                chunk_index=chunk_idx,
                source_text=source,
                translated_text=target,
                model_name="Jarvis-LoRA-API",
                prompt_snapshot=dummy_prompt,
                temperature=0.1,  # <--- ADICIONE ESTA LINHA AQUI (Pode ser 0.1, 0.3, qualquer float)
                duration_seconds=duration
            )
        )
    except Exception as e:
        logger.error(f"Falha ao salvar log: {e}")
        # Importante: Se der erro no log, não queremos travar a transação principal
        # Mas como estamos na mesma session, o ideal é ignorar ou fazer rollback parcial se suportado.
        # No asyncpg, um erro invalida a transação toda, então o try/except aqui é vital.


# --- PIPELINE PRINCIPAL ---

async def translate_epub_book(import_id: int):
    logger.info(f"Iniciando Pipeline Jarvis (HTTP) para import_id={import_id}")
    temp_dir = None

    async with httpx.AsyncClient(timeout=None) as client:

        # Teste de Conexão Rápido
        try:
            # Tenta bater na raiz ou docs só pra ver se está vivo
            await client.get("http://localhost:8000/docs")
        except Exception:
            logger.warning(f"⚠️ AVISO: Não consegui contatar {JARVIS_API_URL}. O Jarvis está rodando? Vou tentar mesmo assim.")

        async with AsyncSessionLocal() as session:
            try:
                paths = await _get_import_paths(session, import_id)
                if not paths: return
                original_path, book_name = paths
                original_path = Path(original_path)

                if not original_path.exists():
                    logger.error(f"Arquivo não encontrado: {original_path}")
                    return

                temp_dir = Path(tempfile.mkdtemp())
                with zipfile.ZipFile(original_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                content_files = list(temp_dir.rglob('*.xhtml')) + list(temp_dir.rglob('*.html'))
                content_files.sort()  # Garante ordem

                logger.info(f"Arquivos para traduzir: {len(content_files)}")

                for i, file_path in enumerate(content_files):
                    try:
                        raw_html = file_path.read_text(encoding='utf-8')
                        clean_text, placeholder_map = _split_and_tokenize_html(raw_html)

                        if len(clean_text.strip()) < 5: continue

                        text_chunks = _smart_chunking(clean_text, MAX_CHUNK_SIZE)
                        translated_chunks = []

                        logger.info(f"Traduzindo {file_path.name} ({len(text_chunks)} partes)...")

                        for j, chunk in enumerate(text_chunks):
                            start_time = time.time()
                            translated_text = ""

                            try:
                                # Chama a API
                                response = await client.post(
                                    JARVIS_API_URL,
                                    json={"text": chunk}
                                )

                                if response.status_code == 200:
                                    translated_text = response.json().get("translation", "")
                                else:
                                    logger.error(f"Erro API Jarvis ({response.status_code}): {response.text}")
                                    translated_text = chunk  # Fallback

                            except Exception as e:
                                logger.error(f"Erro conexão API (Chunk {j}): {e}")
                                translated_text = chunk

                            duration = time.time() - start_time
                            translated_chunks.append(translated_text)

                            # Loga no banco (Agora protegido contra erro de Null)
                            await _log_translation_event(
                                session, import_id, file_path.name, j,
                                chunk, translated_text, duration
                            )

                        # Commit por arquivo para salvar progresso
                        await session.commit()

                        full_text = "\n".join(translated_chunks)
                        final_html = _reconstitute_html(full_text, placeholder_map)
                        file_path.write_text(final_html, encoding='utf-8')

                    except Exception as e:
                        logger.error(f"Erro no arquivo {file_path.name}: {e}")
                        await session.rollback()  # CORREÇÃO 2: Limpa a transação se der erro
                        continue

                # Finalização
                output_dir = Path("data/translated")
                output_dir.mkdir(exist_ok=True, parents=True)
                final_epub_name = f"{book_name}_PT_JARVIS.epub"
                output_path = output_dir / final_epub_name

                _create_output_epub(temp_dir, output_path)
                logger.info(f"✅ SUCESSO! Livro salvo em: {output_path}")

            except Exception as e:
                logger.exception(f"Erro fatal: {e}")
            finally:
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)