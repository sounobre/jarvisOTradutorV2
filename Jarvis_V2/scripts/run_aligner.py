# # --- NOVO ARQUIVO: scripts/run_aligner.py ---
#
# """
# Este é o SCRIPT "ALIGNER" (Alinhador) da Fase 3.
#
# Ele deve ser rodado APENAS UMA VEZ, DEPOIS que o
# 'run_worker.py' tiver traduzido todas as sentenças.
#
# Ele vai:
# 1. Buscar o texto EN original (da fila).
# 2. Buscar o texto EN traduzido (da fila).
# 3. Buscar o texto PT original (dos capítulos).
# 4. Rodar o Bleualign (matemática rápida).
# 5. Salvar o 'corpus_paralelo.tsv' final.
#
# Como rodar (do terminal, na pasta raiz 'Jarvis_v2'):
# $ python -m scripts.run_aligner --import_id 19
# """
#
# import asyncio
# import logging
# import argparse
# import os
# import nltk
# import io
# from typing import List, Dict, Any, Optional
#
# # --- Importações do Projeto ---
# from db.session import AsyncSessionLocal, engine
# from db.models import TmTranslationQueue, TmWinMapping, ChapterText, ChapterIndex
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select, String, cast
#
# # --- Importações de ML ---
# from bleualign.align import Aligner
#
# # --- Constantes ---
# OUTPUT_DIR = 'data'
#
# # Configura o logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# # --- Funções Auxiliares (copiadas) ---
#
# async def _get_full_text_by_title(db: AsyncSession, import_id: int, lang: str, title: str) -> Optional[str]:
#     """
#     Busca o conteúdo (ChapterText.text) usando o TÍTULO (ChapterIndex.title) como chave.
#     *** CORRIGIDO: Adiciona CAST para forçar o JOIN de ch_idx como STRING ***
#     """
#     stmt = select(ChapterText.text).join(
#         ChapterIndex,
#         (ChapterIndex.import_id == ChapterText.import_id) &
#         (ChapterIndex.lang == ChapterText.lang) &
#         # --- A CORREÇÃO ESTÁ AQUI ---
#         (cast(ChapterText.ch_idx, String) == cast(ChapterIndex.ch_idx, String))
#         # --- FIM DA CORREÇÃO ---
#     ).where(
#         ChapterText.import_id == import_id,
#         ChapterText.lang == lang,
#         ChapterIndex.title == title
#     )
#     txt = await db.scalar(stmt)
#     return txt or None
#
#
# # --- O "Coração" do Alinhador ---
#
# async def main(import_id: int):
#     logger.info(f"--- INICIANDO FASE 3 (ALINHAMENTO) PARA IMPORT ID: {import_id} ---")
#
#     sentencas_en_lista: List[str] = []
#     traducoes_pt_lista: List[str] = []
#     sentencas_pt_lista: List[str] = []
#
#     async with AsyncSessionLocal() as session:
#         # 1. Buscar o trabalho da Fila (EN original e EN traduzido)
#         logger.info("Fase 3: Buscando sentenças traduzidas (EN) da fila...")
#         stmt_fila = (
#             select(TmTranslationQueue.source_text, TmTranslationQueue.translated_text)
#             .where(
#                 TmTranslationQueue.import_id == import_id,
#                 TmTranslationQueue.status == "translated"
#             )
#             .order_by(TmTranslationQueue.ch_src, TmTranslationQueue.sent_idx)  # Mantém a ordem
#         )
#         resultado_fila = (await session.execute(stmt_fila)).all()
#
#         for row in resultado_fila:
#             sentencas_en_lista.append(row.source_text)
#             traducoes_pt_lista.append(row.translated_text)
#
#         logger.info(f"Fase 3: Encontradas {len(sentencas_en_lista)} sentenças EN (original e traduzida).")
#
#         if not sentencas_en_lista:
#             logger.error("Fase 3: Nenhuma sentença traduzida encontrada. Rode o 'worker' primeiro. Abortando.")
#             return
#
#         # 2. Buscar o texto PT (original)
#         logger.info("Fase 3: Buscando e segmentando o texto PT (original) dos capítulos...")
#
#         # Garante que o NLTK está baixado
#         try:
#             nltk.data.find('tokenizers/punkt')
#             nltk.data.find('tokenizers/punkt_tab')
#         except LookupError:
#             nltk.download('punkt')
#             nltk.download('punkt_tab')
#
#         # Pega a lista de capítulos PT que deram "match"
#         stmt_mapa_pt = (
#             select(TmWinMapping.ch_tgt)
#             .where(
#                 TmWinMapping.import_id == import_id,
#                 TmWinMapping.llm_verdict == True
#             ).order_by(TmWinMapping.ch_src)  # Ordena pela chave EN (para manter a ordem)
#         )
#         mapa_capitulos_pt = (await session.execute(stmt_mapa_pt)).scalars().all()
#
#         for titulo_pt in mapa_capitulos_pt:
#             texto_pt = await _get_full_text_by_title(session, import_id, 'pt', str(titulo_pt))
#             if texto_pt:
#                 for s in nltk.sent_tokenize(texto_pt, language='portuguese'):
#                     sentenca_limpa = s.strip().replace("\n", " ")
#                     if sentenca_limpa: sentencas_pt_lista.append(sentenca_limpa)
#
#         logger.info(f"Fase 3: Encontradas {len(sentencas_pt_lista)} sentenças PT (original).")
#
#         if not sentencas_pt_lista:
#             logger.error("Fase 3: Texto PT (original) não encontrado. Abortando.")
#             return
#
#     # 3. Configurar e Chamar a Aligner (a parte RÁPIDA)
#     logger.info("Fase 3: Preparando o Aligner (matemática)...")
#
#     # Cria os "arquivos falsos" em memória
#     src_stream = io.StringIO('\n'.join(traducoes_pt_lista))  # A tradução (EN -> PT)
#     target_stream = io.StringIO('\n'.join(sentencas_pt_lista))  # O PT original
#     src_original_stream = io.StringIO('\n'.join(sentencas_en_lista))  # O EN original
#
#     options = {
#         'srcfile': src_original_stream, 'targetfile': target_stream,
#         'srctotarget': [src_stream], 'targettosrc': [],
#         'output-src': io.StringIO(), 'output-target': io.StringIO(),
#         'no_translation_override': False, 'bleu_ngrams': 2,
#         'galechurch': None, 'Nto1': 2,
#         'gapfillheuristics': ["bleu1to1", "galechurch"], 'verbosity': 0
#     }
#
#     try:
#         aligner = Aligner(options)
#         logger.info("Fase 3: Iniciando alinhamento Bleualign (rápido)...")
#         aligner.mainloop()
#         out_src, out_target = aligner.results()
#         output_en = out_src.getvalue().splitlines()
#         output_pt = out_target.getvalue().splitlines()
#
#         # 4. Salvar o resultado final
#         ARQUIVO_CORPUS_FINAL = os.path.join(OUTPUT_DIR, f'{import_id}_corpus_final.tsv')
#         logger.info(f"Fase 3: Alinhamento concluído. Salvando {len(output_en)} pares...")
#         with open(ARQUIVO_CORPUS_FINAL, 'w', encoding='utf-8') as f_out:
#             for en_line, pt_line in zip(output_en, output_pt):
#                 f_out.write(f"{en_line}\t{pt_line}\n")
#
#         logger.info(f"--- \U0001F3C1 PIPELINE COMPLETO CONCLUÍDO! \U0001F3C1 ---")
#         logger.info(f"Seu corpus final está pronto em: {ARQUIVO_CORPUS_FINAL}")
#
#     except Exception as e:
#         logger.exception(f"ERRO CRÍTICO durante o Aligner.mainloop(): {e}")
#     finally:
#         await engine.dispose()
#
#
# # --- Ponto de Entrada do Script ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Fase 3 (Aligner): Roda o alinhamento final."
#     )
#     parser.add_argument(
#         "--import_id", type=int, required=True,
#         help="O 'import_id' (da tabela tm_import) que você quer processar."
#     )
#
#     args = parser.parse_args()
#
#     asyncio.run(main(import_id=args.import_id))