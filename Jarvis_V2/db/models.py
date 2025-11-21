# --- ARQUIVO ATUALIZADO: db/models.py ---

import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    BigInteger, Integer, Text, String, DateTime, func,
    UniqueConstraint, Index, ForeignKey, Float, Boolean
)


class Base(DeclarativeBase):
    pass


# ... (Suas classes Import, Book, ChapterText, ChapterIndex ficam AQUI, sem mudanças) ...
class Import(Base):
    __tablename__ = "tm_import"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_en: Mapped[str] = mapped_column(Text, nullable=False)
    file_pt: Mapped[str | None] = mapped_column(Text, nullable=True)


class Book(Base):
    __tablename__ = "tm_book"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("tm_import.id", ondelete="CASCADE"), nullable=False)
    lang: Mapped[str] = mapped_column(String(2), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    spine_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("import_id", "lang", name="uq_tm_book_import_lang"),)


class ChapterText(Base):
    __tablename__ = "tm_chapter_text"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("tm_import.id", ondelete="CASCADE"), nullable=False)
    lang: Mapped[str] = mapped_column(String(2), nullable=False)
    ch_idx: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    #__table_args__ = (UniqueConstraint("import_id", "lang", "ch_idx", "char_count", name="uq_tm_chapter_text_import_lang_idx"),)


class ChapterIndex(Base):
    __tablename__ = "tm_chapter_index"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("tm_import.id", ondelete="CASCADE"), nullable=False, index=True)
    lang: Mapped[str] = mapped_column(String(2), nullable=False)
    ch_idx: Mapped[str] = mapped_column(Text, nullable=False)
    file_href: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    para_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    __table_args__ = (Index("idx_chapter_index_job_lang", "import_id", "lang", "ch_idx"),)


class ChapterMap(Base):
    __tablename__ = "tm_chapter_map"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("tm_import.id", ondelete="CASCADE"), nullable=False)
    ch_src: Mapped[int] = mapped_column(Integer, nullable=False)
    ch_tgt: Mapped[int] = mapped_column(Integer, nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False, default="win_llm")
    score: Mapped[float | None] = mapped_column(Float)
    sim_cosine: Mapped[float | None] = mapped_column(Float)
    len_src: Mapped[int | None] = mapped_column(Integer)
    len_tgt: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("import_id", "ch_src", name="uq_chapter_map_src"),
        Index("idx_ch_map_by_import", "import_id"),
    )


class ChapterLocateLog(Base):
    __tablename__ = "tm_chapter_locate_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("tm_import.id", ondelete="CASCADE"), nullable=False, index=True)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ch_tgt: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(32), default="llm_locate", nullable=False)
    sample_len: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (Index("idx_loclog_import_hash", "import_id", "text_hash"),)


# --- 1. MODIFICAR TmWinMapping ---
class TmWinMapping(Base):
    __tablename__ = "tm_win_mapping"

    import_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ch_src: Mapped[str] = mapped_column(String(512), primary_key=True)  # Chave de Texto
    ch_tgt: Mapped[str | None] = mapped_column(Text, nullable=True)  # Chave de Texto

    sim_cosine: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    len_src: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    len_tgt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ... (outras colunas llm_score, etc. continuam iguais) ...
    method: Mapped[str] = mapped_column(String(64), nullable=False, default="win_llm_ver")
    llm_score: Mapped[float | None] = mapped_column(Float)
    llm_verdict: Mapped[bool | None] = mapped_column(Boolean)
    llm_reason: Mapped[str | None] = mapped_column(Text)
    anchors: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- 2. ADICIONAR A COLUNA DE STATUS (O que você pediu) ---
    micro_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)

    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    __table_args__ = (
        Index("ix_tm_win_mapping_import_src", "import_id", "ch_src"),
        Index("ix_tm_win_mapping_import_tgt", "import_id", "ch_tgt"),
    )


# --- 3. ADICIONAR A NOVA TABELA de Pares de Sentenças ---
class TmAlignedSentences(Base):
    """
    Esta é a tabela final do "produto".
    Ela armazena os pares de sentenças 1-para-1 encontrados
    e também as suas métricas de validação.
    """
    __tablename__ = "tm_aligned_sentences"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("tm_import.id", ondelete="CASCADE"), nullable=False, index=True)

    ch_src: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)

    # --- 1. NOVAS COLUNAS PARA VALIDAÇÃO ---

    # Status da validação: 'pending' ou 'validated'
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)

    # Métrica 1: Length Ratio (len(pt) / len(en))
    len_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Métrica 2: Number Mismatch (True se os números não baterem)
    number_mismatch: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # (Poderíamos adicionar mais, como "name_mismatch", mas vamos começar com estas)

    # --- FIM DAS NOVAS COLUNAS ---

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("import_id", "ch_src", "source_text", "target_text", name="uq_aligned_pair"),
        Index("idx_aligned_import_status", "import_id", "validation_status"),  # Novo índice
    )