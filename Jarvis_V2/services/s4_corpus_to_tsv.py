# --- ARQUIVO ATUALIZADO: services/s4_corpus_to_tsv.py ---
# ATUALIZAÇÃO: Adicionado auto-incremento no nome do arquivo (batch_1, batch_2...)

import logging
import os
import time
from typing import Optional
from sqlalchemy import select, update, func
from db.models import TmAlignedSentences
from db.session import AsyncSessionLocal
from tqdm import tqdm

logger = logging.getLogger(__name__)

OUTPUT_DIR = 'data'


# --- Helper para atualizar a data ---
async def _mark_as_exported(ids_list):
    """Atualiza a coluna exported_at para a lista de IDs."""
    if not ids_list: return
    async with AsyncSessionLocal() as update_session:
        try:
            stmt = (
                update(TmAlignedSentences)
                .where(TmAlignedSentences.id.in_(ids_list))
                .values(exported_at=func.now())
            )
            await update_session.execute(stmt)
            await update_session.commit()
        except Exception as e:
            logger.error(f"Falha ao marcar IDs como exportados: {e}")


# --- Helper de Filtros ---
def _apply_filters(stmt, import_id, min_score, max_len_ratio, require_number_match, min_cross_score, only_new, context_mode):
    if context_mode:
        effective_min_score = 0.4
        effective_max_ratio = 6.0
        effective_num_match = False
    else:
        effective_min_score = min_score
        effective_max_ratio = max_len_ratio
        effective_num_match = require_number_match

    stmt = stmt.where(
        TmAlignedSentences.validation_status == "validated",
        TmAlignedSentences.len_ratio <= effective_max_ratio
    )

    if import_id:
        stmt = stmt.where(TmAlignedSentences.import_id == import_id)

    if min_cross_score > 0 and not context_mode:
        stmt = stmt.where(TmAlignedSentences.cross_encoder_score >= min_cross_score)
    else:
        stmt = stmt.where(TmAlignedSentences.similarity_score >= effective_min_score)

    if effective_num_match:
        stmt = stmt.where(TmAlignedSentences.number_mismatch == False)

    if only_new:
        stmt = stmt.where(TmAlignedSentences.exported_at.is_(None))

    return stmt


# --- FUNÇÃO PRINCIPAL ---
async def export_corpus_to_tsv(
        import_id: Optional[int],
        min_score: float = 0.7,
        max_len_ratio: float = 3.0,
        require_number_match: bool = True,
        min_cross_score: float = 0.0,
        limit: Optional[int] = None,
        only_new: bool = False,
        context_mode: bool = False
):
    prefix = str(import_id) if import_id else "GLOBAL"
    mode_suffix = "CONTEXT" if context_mode else ("PILOT" if limit else "FULL")
    if only_new: mode_suffix += "_INC"

    logger.info(f"--- FASE 4 INICIADA ({prefix} - {mode_suffix}) ---")

    # --- LÓGICA DE AUTO-INCREMENTO ---
    # Cria o nome base, ex: GLOBAL_corpus_pilot_inc
    base_name = f'{prefix}_corpus_{mode_suffix.lower()}'

    # Tenta o nome base primeiro
    filename = f"{base_name}.tsv"
    output_file = os.path.join(OUTPUT_DIR, filename)

    # Se já existe, começa a contar: _1, _2, _3...
    counter = 1
    while os.path.exists(output_file):
        filename = f"{base_name}_{counter}.tsv"
        output_file = os.path.join(OUTPUT_DIR, filename)
        counter += 1
    # ----------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    async with AsyncSessionLocal() as session:
        try:
            # 1. CONTAGEM
            logger.info("Calculando total de registros...")
            count_stmt = select(func.count()).select_from(TmAlignedSentences)
            count_stmt = _apply_filters(count_stmt, import_id, min_score, max_len_ratio, require_number_match, min_cross_score, only_new, context_mode)

            total_rows = await session.scalar(count_stmt)
            if limit and limit < total_rows:
                total_rows = limit

            logger.info(f"Total estimado: {total_rows} linhas.")

            if total_rows == 0:
                logger.warning("Fase 4: Nenhum dado novo encontrado.")
                return {"error": "Nenhum dado novo encontrado."}

            # 2. SELEÇÃO
            stmt = select(TmAlignedSentences.id, TmAlignedSentences.source_text, TmAlignedSentences.target_text)
            stmt = _apply_filters(stmt, import_id, min_score, max_len_ratio, require_number_match, min_cross_score, only_new, context_mode)

            if limit:
                stmt = stmt.order_by(func.random()).limit(limit)
            else:
                stmt = stmt.order_by(TmAlignedSentences.id)

            stmt = stmt.execution_options(yield_per=10000)
            result_stream = await session.stream(stmt)

            count = 0
            exported_ids = []

            # 3. ESCRITA
            with open(output_file, 'w', encoding='utf-8') as f, \
                    tqdm(total=total_rows, desc=f"Exportando {filename}", unit="linhas") as pbar:

                buffer = []
                BATCH_WRITE_SIZE = 5000

                async for row in result_stream:
                    row_id = row[0]
                    src = row[1].replace('\t', ' ').replace('\n', ' ').strip()
                    tgt = row[2].replace('\t', ' ').replace('\n', ' ').strip()

                    if src and tgt:
                        buffer.append(f"{src}\t{tgt}\n")
                        if only_new: exported_ids.append(row_id)
                        count += 1
                        pbar.update(1)

                    if len(buffer) >= BATCH_WRITE_SIZE:
                        f.writelines(buffer)
                        buffer = []
                        if only_new and len(exported_ids) > 10000:
                            await _mark_as_exported(exported_ids)
                            exported_ids = []

                if buffer: f.writelines(buffer)
                if only_new and exported_ids: await _mark_as_exported(exported_ids)

            duration = time.time() - start_time
            logger.info(f"Exportação concluída: {count} linhas em {output_file} ({duration:.2f}s)")

            return {
                "message": "Sucesso",
                "file": output_file,
                "count": count,
                "duration": duration,
                "mode": mode_suffix
            }

        except Exception as e:
            logger.exception(f"Erro Fase 4: {e}")
            return {"error": str(e)}