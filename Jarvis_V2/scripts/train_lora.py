# --- ARQUIVO CORRIGIDO: scripts/train_lora.py ---
import os
import torch
import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(dataset_folder, output_dir, base_model_id="Qwen/Qwen2.5-7B-Instruct", max_samples=None, resume_from_checkpoint=None):
    logger.info(f"--> Iniciando Treino a partir de pasta JSONL...")

    # 1. Validar Arquivos
    train_file = os.path.join(dataset_folder, "train.jsonl")
    val_file = os.path.join(dataset_folder, "validation.jsonl")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_file}")

    # 2. Carregar Dataset
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})

    if max_samples is not None and max_samples > 0:
        logger.info(f"--> Modo PILOTO: Limitando a {max_samples} exemplos.")
        dataset['train'] = dataset['train'].select(range(min(len(dataset['train']), max_samples)))

    # 3. Carregar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Formatar Prompt
    def formatting_prompts_func(example):
        text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(formatting_prompts_func)

    # 5. Configuração QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # --- MUDANÇA CRÍTICA: Use float32 para estabilidade no Windows ---
        bnb_4bit_compute_dtype=torch.float32,
        # -----------------------------------------------------------------
        bnb_4bit_use_double_quant=True,
    )

    # 6. Carregar Modelo Base
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    model = prepare_model_for_kbit_training(model)

    # 7. Configuração LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 8. Argumentos do Treino (SFTConfig)
    # *** ATENÇÃO AQUI ***
    training_args = SFTConfig(
        output_dir=output_dir,

        # Configs Específicas de SFT
        dataset_text_field="text",  # Fica no Config
        packing=False,  # Fica no Config
        # max_seq_length REMOVIDO DAQUI

        # Configs Gerais
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=50,
        report_to="none"
    )

    # 9. O Treinador
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,

        # *** AQUI ESTÁ O max_seq_length ***
        # A maioria das versões do TRL ainda exige ele aqui no construtor
        max_seq_length=2048
    )

    # 10. Iniciar Treino
    print("--> Iniciando Treino...")

    if resume_from_checkpoint:
        print(f"--> RETOMANDO do checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        last_checkpoint = None
        if os.path.isdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                last_checkpoint = os.path.join(output_dir, checkpoints[-1])

        if last_checkpoint:
            print(f"--> Retomando automaticamente de {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

    # 11. Salvar Final
    print(f"--> Treino finalizado. Salvando modelo final em {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    train(args.input_dir, args.output, max_samples=args.max_samples, resume_from_checkpoint=args.resume)