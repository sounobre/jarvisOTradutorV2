#C:\Users\souno\Desktop\Projects2025\jarvis_v2\Jarvis_V2\jarvis_console.py
import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURAÇÕES ---
# Caminho para onde você baixou a pasta do Drive
ADAPTER_PATH = "data/models/modelo_final_100k"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

print(f"--> Inicializando Jarvis na RTX 3060...")

# 1. Configuração de Quantização (Para caber nos 12GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 2. Carregar o Modelo Base (Qwen)
# O download do Qwen acontece automático na primeira vez
print("--> Carregando Base Model (pode demorar um pouco se for a 1ª vez)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",  # Joga pra GPU
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# 3. Acoplar o seu LoRA (A "Inteligência" que treinamos)
print(f"--> Carregando Adapters de: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)


# Função de Tradução
def jarvis_translate(text):
    # ESTRATÉGIA FEW-SHOT (Damos exemplos para ele copiar o comportamento)
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

Input: She looked at the stone walls. Cold and unforgiving.
Output: Ela olhou para as paredes de pedra. Frias e implacáveis.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Cálcula o tamanho da entrada para criar um limite de segurança
    # Se a frase tem 10 tokens, a tradução não pode ter 100.
    input_len = len(text)
    max_tokens_safe = int(input_len * 2.5) + 20  # Margem segura

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens_safe,  # TRAVA DE SEGURANÇA DE TAMANHO
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,

            # Força o modelo a parar se tentar pular linha (iniciar novo parágrafo)
            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n")[-1]]
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant\n" in output_text:
        traducao = output_text.split("assistant\n")[1].strip()
        # Trava Extra: Pega só a primeira linha caso ele tente continuar na mesma linha
        return traducao.split("\n")[0]

    return output_text


# Loop de Conversa
print("\n" + "=" * 50)
print("🤖 JARVIS ESTÁ ONLINE (Digite 'sair' para fechar)")
print("=" * 50 + "\n")

while True:
    texto_original = input("\n📝 Texto em Inglês: ")
    if texto_original.lower() in ["sair", "exit", "quit"]:
        break

    print("\n⏳ Traduzindo...")
    try:
        traducao = jarvis_translate(texto_original)
        print(f"\n🇧🇷 JARVIS: \n{traducao}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ Erro: {e}")