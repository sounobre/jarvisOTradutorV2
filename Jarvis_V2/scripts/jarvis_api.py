# ARQUIVO: scripts/jarvis_api.py
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURAÇÕES ---
# Ajuste o caminho se necessário (use r"" para caminhos Windows)
ADAPTER_PATH = r"C:\Users\souno\Desktop\Projects2025\jarvis_v2\Jarvis_V2\data\models\modelo_final_100k"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

app = FastAPI(title="Jarvis Translation API")

# Variáveis globais para manter o modelo na memória GPU
model = None
tokenizer = None


class TranslationRequest(BaseModel):
    text: str


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("--> 🚀 Inicializando Jarvis API na RTX 3060...")

    # 1. Configuração de Quantização (Igual ao seu console)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Carregar Base
    print("--> Carregando Modelo Base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 3. Acoplar LoRA
    print(f"--> Carregando Adapters de: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("--> ✅ Jarvis pronto e ouvindo na porta 8000!")


@app.post("/translate")
async def translate(req: TranslationRequest):
    global model, tokenizer
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    text = req.text

    # PROMPT FEW-SHOT (O segredo do sucesso que validamos)
    prompt = f"""<|im_start|>system
You are a precise translator for High Fantasy literature. 
Translate the text from English to Brazilian Portuguese.
Maintain the style but DO NOT add new information, dialogue, or narration.
Stop immediately after the translation.

Examples:
Input: The sword gleamed in the twilight.
Output: A espada brilhou no crepúsculo.

Input: "Run!" he shouted, turning back.
Output: — Corra! — gritou ele, virando-se.
<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Cálculo de segurança de tamanho
    input_len = len(text)
    max_tokens_safe = int(input_len * 3) + 50

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens_safe,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n")[-1]]
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Limpeza da resposta
    traducao = output_text
    if "assistant\n" in output_text:
        traducao = output_text.split("assistant\n")[1].strip()
        traducao = traducao.split("\n")[0]  # Pega só a primeira linha

    return {"translation": traducao}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)