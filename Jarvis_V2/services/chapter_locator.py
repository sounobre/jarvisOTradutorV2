# # --- NEW FILE: services/chapter_locator.py ---
# from __future__ import annotations
# from typing import Dict, List, Tuple, Optional
# from sqlalchemy import select, delete, insert, text as sql_text
# from sqlalchemy.ext.asyncio import AsyncSession
# from db import models
# from utils.ollama_client import OllamaClient
# import hashlib
# import re as _re
#
#
# def _excerpt_three_spots(text: str, each: int = 420) -> Dict[str, str]:
#     """Pega 3 trechos (início, meio, fim) para dar contexto ao LLM."""
#     t = (text or "").strip()
#     n = len(t)
#     if n <= each * 3:
#         first = t[: min(each, n)]
#         mid = t[n // 2 : n // 2 + each]
#         last = t[-each:]
#     else:
#         first = t[:each]
#         mid = t[n // 2 : n // 2 + each]
#         last = t[-each:]
#     return {"first": first, "mid": mid, "last": last}
#
#
# async def _fetch_pt_chapters(db: AsyncSession, import_id: int) -> List[Dict]:
#     q = select(models.ChapterText.ch_idx, models.ChapterText.text).where(
#         models.ChapterText.import_id == import_id,
#         models.ChapterText.lang == "pt",
#     ).order_by(models.ChapterText.ch_idx.asc())
#     rows = (await db.execute(q)).all()
#     return [{"ch_idx": ch, "text": txt or ""} for (ch, txt) in rows]
#
#
# async def _fetch_en_text(db: AsyncSession, import_id: int, ch_src: int) -> Optional[str]:
#     q = select(models.ChapterText.text).where(
#         models.ChapterText.import_id == import_id,
#         models.ChapterText.lang == "en",
#         models.ChapterText.ch_idx == ch_src,
#     )
#     row = (await db.execute(q)).first()
#     return row[0] if row else None
#
#
# async def _keywords_pt_from_en(client: OllamaClient, text_en: str, limit: int = 12) -> List[str]:
#     """Usa Qwen para gerar palavras/termos em PT que devem aparecer na tradução."""
#     sys = (
#         "Você ajuda a localizar o capítulo traduzido em português para um trecho inglês. "
#         "Gere apenas palavras/termos em português que provavelmente aparecem na tradução. "
#         "Responda somente com uma lista separada por vírgulas; sem texto extra."
#     )
#     user = (
#         f"Trecho EN:\n{text_en[:1800]}\n\n"
#         f"Gere {limit} palavras/termos (minúsculas). Apenas a lista separada por vírgulas."
#     )
#     out = await client.chat(sys, user)
#     items = [p.strip().lower() for p in out.replace("\n", " ").split(",")]
#     return [x for x in items if x][:limit]
#
#
# def _to_websearch_query(keywords: List[str]) -> str:
#     """Monta uma consulta estilo 'websearch_to_tsquery' a partir de keywords PT."""
#     toks: List[str] = []
#     for w in keywords:
#         w = (w or "").strip().lower()
#         w = _re.sub(r"[^0-9a-záàâãéêíóôõúç\s-]", " ", w)
#         w = _re.sub(r"\s+", " ", w).strip()
#         if not w:
#             continue
#         toks.append(w)
#     return " ".join(toks[:12]) or ""
#
#
# async def _fts_shortlist(db: AsyncSession, import_id: int, web_q: str, k: int) -> List[Dict]:
#     """Shortlist por FTS PT (requer extensões 'unaccent' e índice FTS)."""
#     if not web_q:
#         return []
#     sql = sql_text(
#         """
#         SELECT ch_idx, text
#         FROM tm_chapter_text
#         WHERE import_id = :imp AND lang = 'pt'
#           AND to_tsvector('portuguese', unaccent(text))
#               @@ websearch_to_tsquery('portuguese', unaccent(:wq))
#         LIMIT :k
#         """
#     )
#     rows = (await db.execute(sql, {"imp": import_id, "wq": web_q, "k": k})).all()
#     return [{"ch_idx": r[0], "text": r[1]} for r in rows]
#
#
# async def _trgm_shortlist(db: AsyncSession, import_id: int, keywords: List[str], k: int) -> List[Dict]:
#     """Fallback por trigram similarity (requer pg_trgm)."""
#     if not keywords:
#         return []
#     kw = [w for w in keywords[:6] if w]
#     if not kw:
#         return []
#     sel = "GREATEST(" + ", ".join([f"similarity(text, :kw{i})" for i, _ in enumerate(kw)]) + ") AS sim"
#     sql = sql_text(
#         f"""
#         SELECT ch_idx, text, {sel}
#         FROM tm_chapter_text
#         WHERE import_id = :imp AND lang = 'pt'
#         ORDER BY sim DESC
#         LIMIT :k
#         """
#     )
#     params = {"imp": import_id, "k": k}
#     params.update({f"kw{i}": kwv for i, kwv in enumerate(kw)})
#     rows = (await db.execute(sql, params)).all()
#     return [{"ch_idx": r[0], "text": r[1]} for r in rows]
#
#
# async def _llm_choose_best(client: OllamaClient, text_en: str, shortlist: List[Dict]) -> Tuple[int, float, str]:
#     """Pede ao Qwen para escolher 1 capítulo PT vencedor dentre a shortlist."""
#     sys = (
#         "Você recebe um trecho em inglês e uma lista de candidatos de capítulos em português (cada um com 3 trechos). "
#         "Escolha apenas um candidato cuja passagem provavelmente é a tradução do trecho. "
#         "Responda exclusivamente em JSON: {\"ch_tgt\": <int>, \"score\": <0..1>, \"explanation\": \"...\"}."
#     )
#     lines = ["Candidatos PT:"]
#     for c in shortlist:
#         ex = _excerpt_three_spots(c["text"], each=320)
#         lines.append(
#             f"- ch_idx={c['ch_idx']} | begin=<<<{ex['first']}>>> | mid=<<<{ex['mid']}>>> | end=<<<{ex['last']}>>>"
#         )
#     user = (
#         f"Trecho EN:\n{text_en[:2500]}\n\n"
#         + "\n".join(lines)
#         + '\n\nRetorne JSON: {"ch_tgt": <int>, "score": <0..1>, "explanation": "..."}'
#     )
#     raw = await client.chat(sys, user)
#     data = OllamaClient.extract_json_block(raw) or {}
#     ch_tgt = int(data.get("ch_tgt", shortlist[0]["ch_idx"])) if shortlist else -1
#     score = float(data.get("score", 0.0))
#     explanation = str(data.get("explanation", "llm choose (fallback)"))
#     return ch_tgt, score, explanation
#
#
# async def locate_pt_chapter(
#     db: AsyncSession,
#     import_id: int,
#     text_en: Optional[str],
#     ch_src: Optional[int],
#     shortlist_k: int = 12,
# ) -> Dict:
#     """Fluxo Qwen-only RAG:
#        1) LLM -> gera termos PT; 2) Postgres FTS/TRGM -> shortlist; 3) LLM -> decide; 4) persiste.
#     """
#     if not text_en and ch_src is None:
#         raise ValueError("Forneça 'text_en' ou 'ch_src'.")
#
#     if not text_en and ch_src is not None:
#         text_en = await _fetch_en_text(db, import_id, ch_src)
#     if not text_en:
#         raise ValueError("Texto EN não encontrado.")
#
#     client = OllamaClient()
#     try:
#         keywords = await _keywords_pt_from_en(client, text_en, limit=12)
#     except Exception:
#         keywords = []
#
#     web_q = _to_websearch_query(keywords)
#
#     shortlist: List[Dict] = []
#     # 1) FTS
#     shortlist = await _fts_shortlist(db, import_id, web_q, shortlist_k)
#     # 2) Fallback TRGM
#     if not shortlist:
#         shortlist = await _trgm_shortlist(db, import_id, keywords, shortlist_k)
#     # 3) Fallback total
#     if not shortlist:
#         shortlist = await _fetch_pt_chapters(db, import_id)
#         shortlist = shortlist[:shortlist_k]
#
#     # Escolha final (Qwen)
#     ch_tgt, score, explanation = await _llm_choose_best(client, text_en, shortlist)
#
#     # Persistência
#     persisted = False
#     if ch_src is not None:
#         await db.execute(
#             delete(models.ChapterMap).where(
#                 models.ChapterMap.import_id == import_id,
#                 models.ChapterMap.ch_src == ch_src,
#             )
#         )
#         await db.execute(
#             insert(models.ChapterMap).values(
#                 import_id=import_id,
#                 ch_src=ch_src,
#                 ch_tgt=ch_tgt,
#                 method="llm_locate",
#                 score=score,
#             )
#         )
#         await db.commit()
#         persisted = True
#     else:
#         h = hashlib.sha256(text_en.encode("utf-8")).hexdigest()
#         await db.execute(
#             insert(models.ChapterLocateLog).values(
#                 import_id=import_id,
#                 text_hash=h,
#                 ch_tgt=ch_tgt,
#                 score=score,
#                 method="llm_locate",
#                 sample_len=len(text_en),
#             )
#         )
#         await db.commit()
#         persisted = True
#
#     await client.aclose()
#
#     return {
#         "import_id": import_id,
#         "ch_src": ch_src,
#         "ch_tgt": ch_tgt,
#         "score": score,
#         "persisted": persisted,
#         "explanation": explanation,
#         "shortlist_size": len(shortlist),
#         "keywords": keywords,
#         "fts_query": web_q,
#     }
