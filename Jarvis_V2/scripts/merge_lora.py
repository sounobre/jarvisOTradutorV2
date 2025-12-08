# --- ARQUIVO CORRIGIDO: scripts/merge_lora.py ---
import argparse
import torch
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_model(base_model_id, lora_path, output_dir):
    logger.info(f"--> Iniciando Merge (Modo CPU para economizar VRAM)...")
    logger.info(f"--> Base: {base_model_id}")
    logger.info(f"--> LoRA: {lora_path}")

    # 1. Carregar Tokenizer
    logger.info("--> Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # 2. Carregar Modelo Base (NA CPU)
    # Usamos low_cpu_mem_usage=True para não explodir a RAM
    logger.info("--> Carregando Modelo Base na CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",  # <--- FORÇA CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 3. Carregar e Aplicar LoRA
    logger.info("--> Aplicando Adaptadores LoRA...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map="cpu"  # <--- FORÇA CPU TAMBÉM
    )

    # 4. O Merge Real
    logger.info("--> Realizando a fusão (Merge & Unload)...")
    model = model.merge_and_unload()

    # 5. Salvar
    logger.info(f"--> Salvando modelo final em: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    logger.info("--> SUCESSO! O modelo merged está pronto.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merge_model(args.base_model, args.lora_path, args.output)