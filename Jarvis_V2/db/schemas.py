# --- NEW FILE: db/schemas.py ---
"""
Esquemas Pydantic para requests/responses.
- Comentários explicam sintaxe de tipos e validação.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class ImportOut(BaseModel):
    """Resposta do endpoint de import: id + caminhos salvos."""
    id: int
    name: str
    file_en: str
    file_pt: str


class MapChaptersRequest(BaseModel):
    """Payload JSON para mapear capítulos de um import existente."""
    import_id: int = Field(..., description="ID do registro em tm_import")
    max_shift: int = Field(6, description="Deslocamento máximo para buscar offset global")
    lam: float = Field(0.12, description="Peso do termo de deslocamento no custo")
    mu: float = Field(0.06, description="Peso do termo de divergência de tamanho no custo")
    reindex: bool = Field(True, description="Regerar índice de capítulos (tm_chapter_index) antes do mapa?")


class ChapterLinkOut(BaseModel):
    ch_src: int
    ch_tgt: int
    sim_cosine: float
    len_src: int
    len_tgt: int
    file_src: str
    file_tgt: str
    title_src: Optional[str] = None
    title_tgt: Optional[str] = None
    flag_low_sim: bool
    flag_len_divergent: bool
    flag_size_divergence: bool


class MapChaptersResponse(BaseModel):
    import_id: int
    offset_global: int
    total_links: int
    warnings_low_sim: int
    warnings_len: int
    warnings_size: int
    links: List[ChapterLinkOut]
