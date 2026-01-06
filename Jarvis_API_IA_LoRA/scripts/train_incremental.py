import torch
import os
import gc
import logging
from threading import Lock
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JarvisTrainer")

# --- CONFIGURAÇÕES PADRÃO (Ajuste para seus caminhos) ---
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PATH_MODELO_ANTERIOR = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\Jarvis_V2\data\models\modelo_final_100k"
PATH_OUTPUT_NOVO = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\Jarvis_V2\data\models\jarvis_v2_incremental"

app = FastAPI(title="Jarvis Training API")

# Trava para impedir dois treinos ao mesmo tempo
training_lock = Lock()
is_training = False


# --- MODELO DE DADOS (O que você manda no Postman) ---
class TrainRequest(BaseModel):
    dataset_path: str  # Caminho da pasta com os novos .jsonl
    resume_checkpoint: bool = True  # True = Continua; False = Começa do Zero
    num_epochs: int = 3
    learning_rate: float = 2e-5


# --- FUNÇÃO DE TREINO (Executada em Background) ---
def run_training_task(params: TrainRequest):
    global is_training
    try:
        logger.info("=== 🚀 INICIANDO PROCESSO DE TREINO INCREMENTAL ===")

        # 1. Limpeza de Memória Pré-Treino
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        gc.collect()

        # 2. Carregar Dataset
        logger.info(f"Carregando dados de: {params.dataset_path}")
        dataset = load_dataset("json", data_files={
            "train": os.path.join(params.dataset_path, "novos_treino.jsonl"),
            "validation": os.path.join(params.dataset_path, "novos_val.jsonl")
        })

        # 3. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        def formatting_prompts_func(examples):
            output_texts = []
            for messages in examples['messages']:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                output_texts.append(text)
            return output_texts

        # 4. Modelo Base (4-bit)
        logger.info("Carregando Qwen Base...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto", use_cache=False
        )
        model = prepare_model_for_kbit_training(model)

        # 5. Carregar LoRA Antigo como "Treinável"
        logger.info(f"Carregando LoRA V1 de: {PATH_MODELO_ANTERIOR}")
        model = PeftModel.from_pretrained(
            model,
            PATH_MODELO_ANTERIOR,
            is_trainable=True  # <--- O SEGREDO DO INCREMENTAL
        )

        # 6. Configuração do Treino
        training_args = TrainingArguments(
            output_dir=PATH_OUTPUT_NOVO,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            optim="paged_adamw_8bit",

            num_train_epochs=params.num_epochs,
            learning_rate=params.learning_rate,
            fp16=True,
            logging_steps=10,

            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset.get('validation'),
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=formatting_prompts_func,
            max_seq_length=512,
            packing=False
        )

        # 7. Lógica de Checkpoint (resume ou não)
        checkpoint_dir = None
        if params.resume_checkpoint and os.path.isdir(PATH_OUTPUT_NOVO):
            checkpoint_dir = get_last_checkpoint(PATH_OUTPUT_NOVO)

        if checkpoint_dir:
            logger.info(f"🔄 Retomando do checkpoint: {checkpoint_dir}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            logger.info("▶️ Iniciando treino do zero (Incremental V1 -> V2)")
            trainer.train()

        # 8. Salvar
        logger.info(f"💾 Salvando modelo final em: {PATH_OUTPUT_NOVO}")
        trainer.model.save_pretrained(PATH_OUTPUT_NOVO)
        tokenizer.save_pretrained(PATH_OUTPUT_NOVO)

        logger.info("✅ TREINO CONCLUÍDO COM SUCESSO!")

    except Exception as e:
        logger.error(f"❌ Erro fatal no treino: {e}")
    finally:
        is_training = False
        torch.cuda.empty_cache()
        gc.collect()


# --- ENDPOINTS ---

@app.post("/train/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    global is_training

    # Verifica se já tem alguém treinando
    if training_lock.acquire(blocking=False):
        try:
            if is_training:
                training_lock.release()
                raise HTTPException(status_code=409, detail="O treino já está em andamento.")

            is_training = True
            background_tasks.add_task(run_training_task, request)
            training_lock.release()

            return {
                "status": "started",
                "message": "Treino iniciado em background. Verifique o console para logs.",
                "config": request
            }
        except Exception as e:
            training_lock.release()
            raise e
    else:
        raise HTTPException(status_code=409, detail="Sistema ocupado.")


@app.get("/train/status")
async def check_status():
    global is_training
    return {"is_training": is_training}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Porta 8001 para não bater com a API de tradução