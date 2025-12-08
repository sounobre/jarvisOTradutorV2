# --- ARQUIVO ATUALIZADO: api/training_api.py ---
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import subprocess
import logging
import os
import uuid

router = APIRouter(prefix="/training", tags=["training"])
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO DE CAMINHOS ---
# Ajuste conforme seu ambiente
PYTHON_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\Jarvis_V2\.venv\Scripts\python.exe"
TRAIN_SCRIPT_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\Jarvis_V2\scripts\train_lora.py"
PREPARE_SCRIPT_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\Jarvis_V2\scripts\prepare_dataset.py"

DATA_DIR = "data"


# --- MODELOS DE REQUEST ---

class TrainRequest(BaseModel):
    dataset_folder: str = Field(..., description="Nome da pasta dentro de data/ onde estão os .jsonl (ex: dataset_ready)")
    max_samples: Optional[int] = None  # Se preenchido, roda modo piloto (ex: 1000)


class PrepareDatasetRequest(BaseModel):
    tsv_filename: str = Field(..., description="Nome do arquivo TSV em 'data/' (ex: 174_corpus_full.tsv)")
    output_folder: str = Field("dataset_ready", description="Nome da pasta de saída (ex: dataset_blocks)")
    mode: str = Field("blocks", description="'blocks' (junta parágrafos) ou 'pairs' (frase a frase)")
    max_chars: int = Field(1500, description="Tamanho do bloco para concatenação (apenas modo blocks)")
    val_split: float = Field(0.05, description="Porcentagem de validação (0.05 = 5%)")


# --- WORKERS (Subprocessos) ---

def _run_preparation_subprocess(req: PrepareDatasetRequest, job_id: str):
    input_path = os.path.join(DATA_DIR, req.tsv_filename)
    output_path = os.path.join(DATA_DIR, req.output_folder)

    if not os.path.exists(input_path):
        logger.error(f"Preparação {job_id} falhou: Arquivo {input_path} não existe.")
        return

    logger.info(f"[{job_id}] Iniciando PREPARAÇÃO DE DATASET...")
    logger.info(f"[{job_id}] Input: {req.tsv_filename} | Modo: {req.mode}")

    cmd = [
        PYTHON_PATH,
        PREPARE_SCRIPT_PATH,
        "--input", input_path,
        "--output", output_path,
        "--mode", req.mode
        # (Poderíamos passar max_chars e val_split se o script suportar args,
        # mas vamos usar os defaults do script ou você pode atualizar o script prepare_dataset para aceitar args)
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"[{job_id}] Preparação concluída! Dataset salvo em: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{job_id}] Erro no script de preparação: {e}")


def _run_training_subprocess(dataset_folder_name: str, job_id: str, max_samples: int = None):
    # O caminho agora aponta para a pasta preparada (dataset_ready)
    dataset_path = os.path.join(DATA_DIR, dataset_folder_name)
    output_dir = os.path.join(DATA_DIR, "models", f"lora_{job_id}")

    if not os.path.exists(os.path.join(dataset_path, "train.jsonl")):
        logger.error(f"Treino falhou: 'train.jsonl' não encontrado em {dataset_path}")
        return

    logger.info(f"[{job_id}] Iniciando TREINO LoRA...")

    cmd = [PYTHON_PATH, TRAIN_SCRIPT_PATH, "--input_dir", dataset_path, "--output", output_dir]

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"[{job_id}] Treino concluído com sucesso! Modelo: {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{job_id}] Erro no script de treino: {e}")


# --- ENDPOINTS ---

@router.post("/prepare-dataset")
async def prepare_dataset_endpoint(req: PrepareDatasetRequest, background_tasks: BackgroundTasks):
    """
    Transforma o TSV bruto em JSONL formatado (ChatML) pronto para treino.
    """
    job_id = str(uuid.uuid4())[:8]

    # Verifica se o arquivo existe antes de agendar
    if not os.path.exists(os.path.join(DATA_DIR, req.tsv_filename)):
        raise HTTPException(status_code=404, detail=f"Arquivo {req.tsv_filename} não encontrado na pasta data/")

    background_tasks.add_task(_run_preparation_subprocess, req, job_id)

    return {
        "status": "preparation_scheduled",
        "job_id": job_id,
        "message": f"Processando {req.tsv_filename} em background."
    }


@router.post("/start")
async def start_training_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Inicia o fine-tuning (LoRA) usando uma pasta de dataset preparada.
    """
    job_id = str(uuid.uuid4())[:8]

    # Verifica se a pasta existe
    if not os.path.exists(os.path.join(DATA_DIR, req.dataset_folder)):
        raise HTTPException(status_code=404, detail=f"Pasta {req.dataset_folder} não encontrada.")

    background_tasks.add_task(_run_training_subprocess, req.dataset_folder, job_id, req.max_samples)

    mode = f"PILOT ({req.max_samples})" if req.max_samples else "FULL"
    return {
        "status": "training_scheduled",
        "job_id": job_id,
        "mode": mode,
        "message": "Treinamento iniciado em background. GPU será ocupada."
    }