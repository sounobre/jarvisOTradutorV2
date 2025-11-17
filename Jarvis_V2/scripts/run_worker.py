# # --- NOVO ARQUIVO: scripts/run_worker.py ---
#
# """
# Este é o SCRIPT "WORKER" (Trabalhador) da Fase 2b.
#
# Ele é feito para rodar continuamente no terminal.
# Ele vai "pedir" trabalhos (sentenças) ao banco de dados,
# traduzi-los com a 3060, e salvar o resultado.
#
# Ele é "resumable": você pode pará-lo (Ctrl+C) e ligá-lo
# de novo, e ele continuará de onde parou.
#
# Como rodar (do terminal, na pasta raiz 'Jarvis_v2'):
# $ python -m scripts.run_worker --import_id 19 --batch_size 500
# """
#
# import asyncio
# import logging
# import argparse
# import time
# from typing import List, Dict, Any, Tuple
#
# # --- Importações do Projeto ---
# from db.session import AsyncSessionLocal, engine
# from db.models import TmTranslationQueue
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select, update, bindparam
#
# # --- Importações de ML ---
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import torch
#
# # --- Constantes ---
# MODELO_TRADUCAO = 'Helsinki-NLP/opus-mt-en-ROMANCE'
# # Quantas sentenças o worker pega do DB de uma vez
# DEFAULT_WORKER_BATCH_SIZE = 500
# # Quantas sentenças a 3060 traduz de uma vez
# DEFAULT_GPU_BATCH_SIZE = 128
#
# # Configura o logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# async def fetch_pending_jobs(db: AsyncSession, import_id: int, limit: int) -> List[TmTranslationQueue]:
#     """
#     Busca um lote de jobs "pending" no banco.
#     Usa "FOR UPDATE SKIP LOCKED" para que, se você rodar
#     dois workers ao mesmo tempo, eles não peguem o mesmo trabalho.
#     """
#     stmt = (
#         select(TmTranslationQueue)
#         .where(
#             TmTranslationQueue.import_id == import_id,
#             TmTranslationQueue.status == "pending"
#         )
#         .limit(limit)
#         .with_for_update(skip_locked=True)  # Mágica do "lock"
#     )
#
#     resultado = await db.execute(stmt)
#     jobs = resultado.scalars().all()  # Retorna a lista de objetos
#     return jobs
#
#
# async def update_jobs_batch(db: AsyncSession, jobs_data: List[Dict[str, Any]]):
#     """
#     Atualiza (UPDATE) um lote de jobs no banco com o status
#     e o texto traduzido.
#     """
#     # Esta é uma query "bulk update" (atualização em massa)
#     stmt = (
#         update(TmTranslationQueue)
#         .where(TmTranslationQueue.id == bindparam("_id"))  # bindparam é um placeholder
#         .values(
#             translated_text=bindparam("_translated_text"),
#             status=bindparam("_status"),
#             updated_at=func.now()  # Atualiza o timestamp
#         )
#     )
#
#     # Mapeia os dados para os placeholders
#     update_data = [
#         {
#             "_id": job["id"],
#             "_translated_text": job["translated_text"],
#             "_status": job["status"]
#         } for job in jobs_data
#     ]
#
#     await db.execute(stmt, update_data)
#
#
# # --- O "Coração" do Worker ---
#
# async def main(import_id: int, worker_batch_size: int):
#     """
#     Função principal do worker.
#     """
#     logger.info(f"--- WORKER INICIADO para import_id={import_id} (Lotes de {worker_batch_size}) ---")
#
#     # 1. Carregar o Modelo (só uma vez)
#     logger.info(f"Carregando modelo de tradução: '{MODELO_TRADUCAO}' (aguarde)...")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODELO_TRADUCAO)
#         model = AutoModelForSeq2SeqLM.from_pretrained(MODELO_TRADUCAO)
#         model.to('cuda')  # Manda para a 3060!
#         logger.info("Modelo carregado na GPU. Worker pronto para traduzir.")
#     except Exception as e:
#         logger.exception(f"ERRO CRÍTICO: Falha ao carregar o modelo. O worker não pode iniciar. {e}")
#         return
#
#     # 2. Loop Infinito (até não ter mais jobs)
#     try:
#         while True:
#             total_traduzido_nesta_sessao = 0
#
#             async with AsyncSessionLocal() as session:
#                 async with session.begin():  # Inicia uma transação
#
#                     # 3. Pega o próximo lote de trabalhos
#                     jobs = await fetch_pending_jobs(session, import_id, worker_batch_size)
#
#                     if not jobs:
#                         logger.info("Nenhum job 'pending' encontrado. O trabalho parece estar concluído.")
#                         logger.info("Desligando o worker...")
#                         break  # Quebra o loop "while True"
#
#                     logger.info(f"Pegou {len(jobs)} jobs do banco. Iniciando tradução...")
#
#                     # Prepara os dados para a 3060
#                     ids_dos_jobs = [job.id for job in jobs]
#                     sentencas_para_traduzir = [job.source_text for job in jobs]
#                     traducoes_feitas: List[str] = []
#
#                     # 4. Loop de Tradução (a parte da 3060)
#                     for i in range(0, len(sentencas_para_traduzir), DEFAULT_GPU_BATCH_SIZE):
#                         lote_en = sentencas_para_traduzir[i: i + DEFAULT_GPU_BATCH_SIZE]
#
#                         inputs = tokenizer(lote_en, return_tensors="pt", padding=True, truncation=True, max_length=512)
#                         inputs = {k: v.to('cuda') for k, v in inputs.items()}
#                         outputs = model.generate(**inputs)
#
#                         traducoes_feitas.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
#
#                         logger.info(
#                             f"  Progresso do lote: {len(traducoes_feitas)} / {len(sentencas_para_traduzir)} traduzidas...")
#
#                     # 5. Prepara os dados para o UPDATE
#                     dados_para_atualizar: List[Dict[str, Any]] = []
#                     for i in range(len(ids_dos_jobs)):
#                         dados_para_atualizar.append({
#                             "id": ids_dos_jobs[i],
#                             "translated_text": traducoes_feitas[i],
#                             "status": "translated"  # Marca como concluído
#                         })
#
#                     # 6. Salva o resultado no banco
#                     await update_jobs_batch(session, dados_para_atualizar)
#
#                 # `session.begin()` faz o commit automático aqui
#
#                 total_traduzido_nesta_sessao += len(jobs)
#                 logger.info(f"Lote de {len(jobs)} jobs concluído e salvo no DB.")
#                 logger.info(f"Total traduzido nesta sessão: {total_traduzido_nesta_sessao}")
#
#             # Pequena pausa para não sobrecarregar o DB
#             await asyncio.sleep(1)
#
#     except KeyboardInterrupt:
#         logger.info("--- Ctrl+C detectado. Desligando o worker... ---")
#     except Exception as e:
#         logger.exception(f"ERRO CRÍTICO no loop do worker: {e}")
#     finally:
#         # Garante que o motor do DB seja fechado
#         await engine.dispose()
#
#
# # --- Ponto de Entrada do Script ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Fase 2b (Worker): Roda o loop de tradução."
#     )
#     parser.add_argument(
#         "--import_id", type=int, required=True,
#         help="O 'import_id' (da tabela tm_import) que você quer processar."
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=DEFAULT_WORKER_BATCH_SIZE,
#         help="Quantas sentenças pegar do DB de uma vez."
#     )
#
#     args = parser.parse_args()
#
#     asyncio.run(main(import_id=args.import_id, worker_batch_size=args.batch_size))