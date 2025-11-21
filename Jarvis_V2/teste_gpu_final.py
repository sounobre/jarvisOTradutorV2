import os

# Força a GPU antes de carregar qualquer coisa
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from llama_cpp import Llama

print(">>> Tentando carregar o modelo na GPU...")

try:
    # Substitua pelo caminho real do seu modelo se precisar, ou use o repo_id
    llm = Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_gpu_layers=-1,  # Tenta jogar tudo pra GPU
        verbose=True
    )
    print("\n>>> SUCESSO! Veja o log acima.")
    print(">>> Procure por: 'llm_load_tensors: offloaded 33/33 layers to GPU'")

except Exception as e:
    print(f">>> ERRO: {e}")