# # --- NEW FILE: services/window_mapper.py ---
# """
# Mapeamento EN->PT por capítulo usando JANELA ±K e LLM.
#
# Etapas:
#  1) Para um capítulo EN, extraímos um EXCERPT (início/meio/fim, limitado).
#  2) Montamos uma janela de capítulos PT [ch_src-K .. ch_src+K] (com clamp nas bordas).
#  3) Para cada candidato PT, geramos 3 EXCERTOS curtos (begin/mid/end).
#  4) Chamamos o Qwen perguntando: "o trecho EN está contido em ALGUM destes candidatos PT?"
#  5) Se sim, persistimos em tm_chapter_map (idempotente por import_id,ch_src); senão expandimos a janela.
#
# Inclui:
#  - logs em cada tentativa, janela usada, tempos, e logs específicos de entrada/saída do LLM (resumo).
#  - função para um capítulo e para o LIVRO TODO (range ou todos).
# """
# from __future__ import annotations
# from typing import Dict, List, Optional, Tuple, Any
# import logging, time
# from sqlalchemy import select, insert, delete, text as sql_text
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import exc as sa_exc
#
# from db import models
# from utils.ollama_client import OllamaClient
# from core.config import settings
#
# from utils.text_norm import contains, normalize
#
# logger = logging.getLogger("services.window_mapper")
#
#
# # ------------------------- Excertos (limitadores de contexto) -------------------------
# # ---------- excertos ----------
# def _excerpt_three_spots(text: str, each: int) -> Dict[str, str]:
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
# def _excerpt_for_llm_en(text: str, max_chars: int) -> str:
#     each = max(120, max_chars // 3)
#     ex = _excerpt_three_spots(text, each=each)
#     combo = f"{ex['first']}\n...\n{ex['mid']}\n...\n{ex['last']}"
#     return combo[:max_chars]
#
#
#
# # ------------------------- Consultas DB -------------------------
# async def _get_chapter_count(db: AsyncSession, import_id: int, lang: str) -> int:
#     q = select(models.ChapterText.ch_idx).where(
#         models.ChapterText.import_id == import_id,
#         models.ChapterText.lang == lang,
#     )
#     rows = (await db.execute(q)).all()
#     return len(rows)
#
# async def _fetch_chapter_text(db: AsyncSession, import_id: int, lang: str, ch_idx: int) -> Optional[str]:
#     q = select(models.ChapterText.text).where(
#         models.ChapterText.import_id == import_id,
#         models.ChapterText.lang == lang,
#         models.ChapterText.ch_idx == ch_idx,
#     )
#     row = (await db.execute(q)).first()
#     return row[0] if row else None
#
# async def _fetch_pt_window(db: AsyncSession, import_id: int, start_idx: int, end_idx: int) -> List[Dict]:
#     q = select(models.ChapterText.ch_idx, models.ChapterText.text).where(
#         models.ChapterText.import_id == import_id,
#         models.ChapterText.lang == "pt",
#         models.ChapterText.ch_idx >= start_idx,
#         models.ChapterText.ch_idx <= end_idx,
#     ).order_by(models.ChapterText.ch_idx.asc())
#     rows = (await db.execute(q)).all()
#     return [{"ch_idx": ch, "text": txt or ""} for (ch, txt) in rows]
#
# async def _mapped_already(db: AsyncSession, import_id: int, ch_src: int) -> bool:
#     q = select(models.ChapterMap.id).where(
#         models.ChapterMap.import_id == import_id,
#         models.ChapterMap.ch_src == ch_src,
#     )
#     return (await db.execute(q)).first() is not None
#
#
#
# # ------------------------- LLM -------------------------
# def _build_user_prompt(ch_src: int, excerpt_en: str, pt_candidates: List[Dict], strict: bool) -> str:
#     cand_sz = max(120, settings.LOG_LLM_CAND_EXCERPT_CHARS)
#     lines = [
#         f"Capítulo EN (ch_src={ch_src}) — excerto:\n<<<{excerpt_en}>>>\n",
#         "Candidatos PT (cada um com trechos begin/mid/end):"
#     ]
#     for c in pt_candidates:
#         ex = _excerpt_three_spots(c["text"], each=cand_sz)
#         lines.append(
#             f"- ch_idx={c['ch_idx']} | begin=<<<{ex['first']}>>> | mid=<<<{ex['mid']}>>> | end=<<<{ex['last']}>>>"
#         )
#     rules = (
#         "Regras de saída (JSON **único**): "
#         '{"found": true|false, "best": <int|null>, "score": 0..1, "reason": "..."'
#         ', "anchors": ["Tamlin","Lucien",...], '
#         '"pt_quotes": ["citação1","citação2",...], '
#         '"anchor_hits": ["citação contendo uma âncora", "..."], '
#         '"candidates": [{"ch_idx": X, "score": 0..1}]}\n'
#         f"- As **pt_quotes** e **anchor_hits** DEVEM ser copiadas **literalmente** de begin/mid/end do capítulo escolhido.\n"
#         f"- Mínimo de {settings.EVIDENCE_MIN_QUOTES} pt_quotes, cada uma com ≥{settings.QUOTE_MIN_CHARS} caracteres.\n"
#         "- Inclua ao menos 2 âncoras (nomes próprios/termos raros) extraídas do EN (ex.: Tamlin, Lucien, Suriel, Alis, Prythian, naga, etc.).\n"
#     )
#     if strict:
#         rules += (
#             "- **Se você não conseguir fornecer as citações exatas do capítulo escolhido ou não achar âncoras válidas nele, responda {\"found\": false}.**\n"
#         )
#     return "\n".join(lines) + "\n\n" + rules + 'Retorne apenas o JSON.'
#
# def _build_system_prompt(strict: bool) -> str:
#     base = (
#         "Você é um verificador bilíngue disciplinado. Receberá um excerto em INGLÊS e vários candidatos de capítulos em PORTUGUÊS.\n"
#         "Tarefa: decidir se o excerto EN corresponde (tradução) a **algum** candidato PT. Se sim, escolha UM (best) e PROVE com trechos PT copiados.\n"
#         "Evite alucinar. Nunca invente trechos PT; apenas copie dos excertos fornecidos (begin/mid/end)."
#     )
#     if strict:
#         base += "\nModo estrito: sem evidências literais suficientes, retorne found=false."
#     return base
#
# # ---------- Verificação local ----------
# def _validate_evidence(chosen_pt_text: str, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
#     """
#     Regras:
#       - ≥ EVIDENCE_MIN_QUOTES pt_quotes presentes (normalizados) e com tamanho mínimo
#       - ≥ ANCHOR_MIN_HITS anchor_hits presentes e contendo ao menos 1 âncora normalizada
#     """
#     min_quotes = settings.EVIDENCE_MIN_QUOTES
#     min_hits = settings.ANCHOR_MIN_HITS
#     min_qchars = settings.QUOTE_MIN_CHARS
#
#     pt_quotes = data.get("pt_quotes") or []
#     anchor_hits = data.get("anchor_hits") or []
#     anchors = [a for a in (data.get("anchors") or []) if isinstance(a, str)]
#
#     ok_quotes = 0
#     for q in pt_quotes:
#         if isinstance(q, str) and len(q.strip()) >= min_qchars and contains(chosen_pt_text, q):
#             ok_quotes += 1
#
#     ok_anchors = 0
#     for hit in anchor_hits:
#         if not isinstance(hit, str) or not contains(chosen_pt_text, hit):
#             continue
#         # hit deve conter alguma âncora literal (normalizada)
#         if any(normalize(a) and normalize(a) in normalize(hit) for a in anchors):
#             ok_anchors += 1
#
#     details = {"ok_quotes": ok_quotes, "ok_anchors": ok_anchors, "need_quotes": min_quotes, "need_anchors": min_hits}
#     passed = (ok_quotes >= min_quotes) and (ok_anchors >= min_hits)
#     return passed, details
#
# async def _ask_llm_contains_any(
#     client: OllamaClient, ch_src: int, excerpt_en: str, pt_candidates: List[Dict], strict: bool = False
# ) -> Tuple[bool, Optional[int], float, str, str, Dict[str, Any]]:
#     sys = _build_system_prompt(strict)
#     user = _build_user_prompt(ch_src, excerpt_en, pt_candidates, strict)
#
#     # LOG entrada
#     req_preview = user[:settings.LOG_LLM_REQUEST_MAX_CHARS].replace("\n", "\\n")
#     logger.info("[llm->] ch_src=%s | strict=%s | prompt_chars=%d | preview='%s...'",
#                 ch_src, strict, len(user), req_preview)
#
#     t0 = time.perf_counter()
#     raw = await client.chat(sys, user)
#     dt = time.perf_counter() - t0
#
#     resp_preview = raw[:settings.LOG_LLM_RESPONSE_MAX_CHARS].replace("\n", "\\n")
#     logger.info("[llm<-] ch_src=%s | elapsed=%.2fs | response_preview='%s...'", ch_src, dt, resp_preview)
#
#     data = OllamaClient.extract_json_block(raw) or {}
#     found = bool(data.get("found", False))
#     best = data.get("best", None)
#     best_int = int(best) if (best is not None) else None
#     score = float(data.get("score", 0.0))
#     reason = str(data.get("reason", ""))
#     logger.info("[llm-p] ch_src=%s | found=%s | best=%s | score=%.3f | reason=%s",
#                 ch_src, found, best_int, score, reason)
#     return found, best_int, score, reason, raw, data
#
#
# # ------------------------- Persistência (compatível com schema legado) -------------------------
# async def _save_map(db: AsyncSession, import_id: int, ch_src: int, ch_tgt: int, score: float) -> None:
#     try:
#         en_text = await _fetch_chapter_text(db, import_id, "en", ch_src) or ""
#         pt_text = await _fetch_chapter_text(db, import_id, "pt", ch_tgt) or ""
#         len_src = len(en_text)
#         len_tgt = len(pt_text)
#         sim_cos = float(score or 0.0)
#
#         upsert_sql = sql_text("""
#             INSERT INTO tm_chapter_map (import_id, ch_src, ch_tgt, method, score, sim_cosine, len_src, len_tgt)
#             VALUES (:imp, :cs, :ct, :m, :sc, :sim, :ls, :lt)
#             ON CONFLICT (import_id, ch_src) DO UPDATE SET
#                 ch_tgt     = EXCLUDED.ch_tgt,
#                 method     = EXCLUDED.method,
#                 score      = EXCLUDED.score,
#                 sim_cosine = EXCLUDED.sim_cosine,
#                 len_src    = EXCLUDED.len_src,
#                 len_tgt    = EXCLUDED.len_tgt
#         """)
#         await db.execute(
#             upsert_sql,
#             {"imp": import_id, "cs": ch_src, "ct": ch_tgt,
#              "m": "win_llm_ver", "sc": sim_cos, "sim": sim_cos,
#              "ls": len_src, "lt": len_tgt}
#         )
#         await db.commit()
#         logger.info("[persist] MAPPED(verified) | import_id=%s ch_src=%s -> ch_tgt=%s | score=%.3f len_src=%s len_tgt=%s",
#                     import_id, ch_src, ch_tgt, score, len_src, len_tgt)
#
#     except sa_exc.DBAPIError as e:
#         logger.warning("[persist] upsert legado falhou; tentando insert minimal | err=%s", e)
#         await db.rollback()
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
#                 method="win_llm_ver",
#                 score=float(score or 0.0),
#             )
#         )
#         await db.commit()
#         logger.info("[persist] MAPPED(minimal,verified) | import_id=%s ch_src=%s -> ch_tgt=%s | score=%.3f",
#                     import_id, ch_src, ch_tgt, score)
#
# # ------------------------- API-facing: capítulo único -------------------------
# async def map_single_by_window(
#     db: AsyncSession,
#     import_id: int,
#     ch_src: int,
#     window_k: int = 3,
#     expand_step: int = 3,
#     max_expansions: int = 3,
#     en_max_chars: int = 1600,
# ) -> Dict:
#     log_ctx = {"imp": import_id, "ch_src": ch_src}
#     logger.info("[map_single] start | ctx=%s", log_ctx)
#
#     total_en = await _get_chapter_count(db, import_id, "en")
#     total_pt = await _get_chapter_count(db, import_id, "pt")
#     if total_en == 0 or total_pt == 0:
#         logger.warning("[map_single] livros vazios | total_en=%s total_pt=%s | %s", total_en, total_pt, log_ctx)
#         return {"mapped": False, "reason": "no chapters", **log_ctx}
#
#     en_text = await _fetch_chapter_text(db, import_id, "en", ch_src)
#     if not en_text:
#         logger.warning("[map_single] en_text vazio | %s", log_ctx)
#         return {"mapped": False, "reason": "empty en_text", **log_ctx}
#     excerpt_en = _excerpt_for_llm_en(en_text, max_chars=en_max_chars)
#
#     min_idx, max_idx = 0, max(0, total_pt - 1)
#     attempt = 0
#     K = max(0, int(window_k))
#     client = OllamaClient()
#
#     try:
#         while True:
#             attempt += 1
#             start = max(min_idx, ch_src - K)
#             end = min(max_idx, ch_src + K)
#             candidates = await _fetch_pt_window(db, import_id, start, end)
#             logger.info("[map_single] attempt=%d | window=[%d,%d] | cand=%d | %s",
#                         attempt, start, end, len(candidates), log_ctx)
#
#             if not candidates:
#                 if attempt > max_expansions:
#                     logger.warning("[map_single] sem candidatos e esgotou expansões | %s", log_ctx)
#                     return {"mapped": False, "reason": "no candidates", "attempts": attempt, **log_ctx}
#                 K += max(1, int(expand_step))
#                 continue
#
#             # 1ª chamada (normal)
#             found, ch_tgt, score, reason, raw, data = await _ask_llm_contains_any(
#                 client, ch_src, excerpt_en, candidates, strict=False
#             )
#
#             # Se claimed found, validar evidências
#             if found and (ch_tgt is not None):
#                 # localizar texto do escolhido (dentro da janela)
#                 chosen = next((c for c in candidates if c["ch_idx"] == ch_tgt), None)
#                 if not chosen:
#                     logger.info("[verify] escolhido fora da janela? | ch_tgt=%s | %s", ch_tgt, log_ctx)
#                 else:
#                     passed, details = _validate_evidence(chosen["text"], data)
#                     logger.info("[verify] ch_src=%s -> ch_tgt=%s | passed=%s | details=%s",
#                                 ch_src, ch_tgt, passed, details)
#                     if passed:
#                         await _save_map(db, import_id, ch_src, ch_tgt, score)
#                         return {
#                             "mapped": True,
#                             "import_id": import_id,
#                             "ch_src": ch_src,
#                             "ch_tgt": ch_tgt,
#                             "score": score,
#                             "attempts": attempt,
#                             "window_last": [start, end],
#                             "reason": reason,
#                             "verify": details,
#                             "llm_raw_len": len(raw),
#                             "mode": "normal+verified",
#                         }
#
#             # 2ª chamada (strict) se a 1ª falhou a verificação
#             logger.info("[map_single] retry strict | %s", log_ctx)
#             found2, ch_tgt2, score2, reason2, raw2, data2 = await _ask_llm_contains_any(
#                 client, ch_src, excerpt_en, candidates, strict=True
#             )
#             if found2 and (ch_tgt2 is not None):
#                 chosen2 = next((c for c in candidates if c["ch_idx"] == ch_tgt2), None)
#                 if chosen2:
#                     passed2, details2 = _validate_evidence(chosen2["text"], data2)
#                     logger.info("[verify] strict ch_src=%s -> ch_tgt=%s | passed=%s | details=%s",
#                                 ch_src, ch_tgt2, passed2, details2)
#                     if passed2:
#                         await _save_map(db, import_id, ch_src, ch_tgt2, score2)
#                         return {
#                             "mapped": True,
#                             "import_id": import_id,
#                             "ch_src": ch_src,
#                             "ch_tgt": ch_tgt2,
#                             "score": score2,
#                             "attempts": attempt,
#                             "window_last": [start, end],
#                             "reason": reason2,
#                             "verify": details2,
#                             "llm_raw_len": len(raw2),
#                             "mode": "strict+verified",
#                         }
#
#             # expansão de janela
#             if attempt >= max(1, int(max_expansions)):
#                 logger.info("[map_single] NOT FOUND/NOT VERIFIED | esgotou expansões | %s", log_ctx)
#                 return {
#                     "mapped": False,
#                     "import_id": import_id,
#                     "ch_src": ch_src,
#                     "attempts": attempt,
#                     "window_last": [start, end],
#                     "reason": "not found or failed verification within window expansions",
#                 }
#             K += max(1, int(expand_step))
#     finally:
#         await client.aclose()
#
#
# # ------------------------- API-facing: LIVRO INTEIRO -------------------------
# async def map_book_by_window(
#     db: AsyncSession,
#     import_id: int,
#     start_en: Optional[int] = None,
#     end_en: Optional[int] = None,
#     window_k: int = 3,
#     expand_step: int = 3,
#     max_expansions: int = 3,
#     en_max_chars: int = 1600,
#     skip_existing: bool = True,
# ) -> Dict:
#     """
#     Mapeia um RANGE (ou todos) os capítulos EN do livro.
#       - Por padrão, pula capítulos já mapeados (skip_existing).
#       - Você pode limitar o range via start_en/end_en (inclusive).
#     """
#     total_en = await _get_chapter_count(db, import_id, "en")
#     if total_en == 0:
#         return {"ok": False, "reason": "no EN chapters", "import_id": import_id}
#
#     if start_en is None:
#         start_en = 0
#     if end_en is None:
#         end_en = total_en - 1
#     start_en = max(0, int(start_en))
#     end_en = min(total_en - 1, int(end_en))
#
#     logger.info("[map_book] start | import_id=%s | range=[%d..%d] | window_k=%d expand_step=%d max_exp=%d",
#                 import_id, start_en, end_en, window_k, expand_step, max_expansions)
#
#     results: List[Dict] = []
#     mapped_count = 0
#
#     for ch in range(start_en, end_en + 1):
#         if skip_existing and await _mapped_already(db, import_id, ch):
#             logger.info("[map_book] skip existing | ch_src=%d", ch)
#             results.append({"mapped": True, "import_id": import_id, "ch_src": ch, "skipped": True})
#             continue
#
#         try:
#             r = await map_single_by_window(
#                 db=db,
#                 import_id=import_id,
#                 ch_src=ch,
#                 window_k=window_k,
#                 expand_step=expand_step,
#                 max_expansions=max_expansions,
#                 en_max_chars=en_max_chars,
#             )
#             results.append(r)
#             if r.get("mapped"):
#                 mapped_count += 1
#         except Exception as e:
#             logger.exception("[map_book] erro mapeando ch_src=%d | imp=%d | err=%s", ch, import_id, e)
#             results.append({"mapped": False, "import_id": import_id, "ch_src": ch, "error": str(e)})
#
#     logger.info("[map_book] done | import_id=%s | mapped=%d/%d", import_id, mapped_count, len(results))
#     return {"ok": True, "import_id": import_id, "mapped": mapped_count, "total": len(results), "results": results}
