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
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# --- ☢️☢️☢️ CONFIGURAÇÃO CRÍTICA (SentAlign) ☢️☢️☢️ ---
# Ajuste estes caminhos para o SEU ambiente de teste onde o SentAlign funciona


# --- FIM DA CONFIGURAÇÃO ---

REPO_ID = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q5_K_M.gguf"

# --- Constantes ---
MODELO_EMBEDDING = 'sentence-transformers/LaBSE'


GPU_BATCH_SIZE = 256


logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER: LOGS NO BANCO (FASE 1)
# ==============================================================================




# ==============================================================================
# HELPERS DE BUSCA E BANCO DE DADOS
# ==============================================================================





















# ==============================================================================
# FASE 2: MICRO ALIGNMENT (STANZA + SENTALIGN)
# ==============================================================================

# --- SUBSTITUA ESTA FUNÇÃO EM: services/corpus_builder_service.py ---







# --- JOB MESTRE FASE 2 ---






# ==============================================================================
# FASE 3: VALIDAÇÃO E FASE 4: EXPORTAÇÃO
# ==============================================================================

# ==============================================================================
# FASE 3: VALIDAÇÃO (AGORA COM CROSS-ENCODER)
# ==============================================================================



# --- FASE 4: EXPORTAÇÃO (Atualizada para usar o novo score) ---










# --- FUNÇÃO PÚBLICA 5 (A REAL): O STATUS INTELIGENTE ---



# --- HELPER DE LOG FASE 2 (NOVO) ---

# --- ADICIONE NO FINAL DE services/corpus_builder_service.py ---







