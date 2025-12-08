# File services/sall_get_corpus_status.py
# Descrição: Serviço para obter o status do corpus, incluindo contagem de capítulos e frases

import logging
from typing import Optional, Dict, Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db import models
from db.models import TmWinMapping, TmAlignedSentences
from db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


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
