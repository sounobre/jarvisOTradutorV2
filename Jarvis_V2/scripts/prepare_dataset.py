# --- ARQUIVO CORRIGIDO E "FALADOR": scripts/prepare_dataset.py ---
import csv
import json
import os
import argparse
import random
import sys
from tqdm import tqdm

# --- CORREÇÃO DO OVERFLOW (WINDOWS) ---
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)
# --------------------------------------

SYSTEM_PROMPT = """You are a professional literary translator specializing in High Fantasy novels. 
Translate the text below from English to Portuguese (Brazil).
Guidelines:
1. Maintain the tone, style, and atmosphere of the original text.
2. Keep all formatting tags (like [TAG_0000]) EXACTLY where they are.
3. Do NOT provide explanations, only the translated text."""


def count_file_lines(filepath):
    print(f"--> Calculando tamanho total do arquivo...")
    try:
        with open(filepath, 'rb') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Aviso: Erro ao contar linhas ({e}).")
        return None


def prepare_dataset(input_tsv, output_dir, mode="blocks", max_chars=1500, val_split=0.05):
    print(f"--> INICIANDO PREPARAÇÃO: {input_tsv}")
    print(f"--> Modo: {mode.upper()}")

    # 1. Leitura
    total_lines = count_file_lines(input_tsv)
    raw_data = []

    with open(input_tsv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, total=total_lines, desc="Lendo TSV", unit="linhas"):
            if len(row) >= 2 and row[0] and row[1]:
                raw_data.append({'en': row[0], 'pt': row[1]})

    print(f"--> Frases carregadas: {len(raw_data)}")
    if len(raw_data) == 0:
        print("ERRO: Nenhuma frase encontrada! Verifique o TSV.")
        return

    final_samples = []

    # 2. Processamento (Blocos vs Pares)
    if mode == "blocks":
        current_en, current_pt, current_len = [], [], 0
        for item in tqdm(raw_data, desc="Criando Blocos", unit="frases"):
            len_item = len(item['en']) + len(item['pt'])
            if current_len + len_item > max_chars:
                if current_en:
                    final_samples.append({'en': "\n".join(current_en), 'pt': "\n".join(current_pt)})
                current_en, current_pt = [item['en']], [item['pt']]
                current_len = len_item
            else:
                current_en.append(item['en']);
                current_pt.append(item['pt'])
                current_len += len_item
        if current_en:
            final_samples.append({'en': "\n".join(current_en), 'pt': "\n".join(current_pt)})

    elif mode == "pairs":
        final_samples = raw_data

    print(f"--> Total de exemplos gerados: {len(final_samples)}")

    # 3. Formatação ChatML
    formatted_data = []
    for sample in final_samples:
        formatted_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample['en']},
                {"role": "assistant", "content": sample['pt']}
            ]
        })

    # 4. Divisão (Split)
    print("--> Dividindo Treino/Validação...")
    random.shuffle(formatted_data)

    # Conta simples de índice
    split_idx = int(len(formatted_data) * (1 - val_split))

    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]

    # Lógica de segurança: Se Validação ficou vazia, rouba um do treino
    if len(val_data) == 0 and len(train_data) > 0:
        print("AVISO: Validação estava vazia. Movendo 1 item do treino para validação.")
        val_data = [train_data.pop()]

    print(f"--> Dataset Treino: {len(train_data)} itens")
    print(f"--> Dataset Valid : {len(val_data)} itens")

    # 5. Salvar (Com verificação)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "validation.jsonl")

    print(f"--> Escrevendo {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"--> Escrevendo {val_path}...")
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Verificação final
    if os.path.exists(val_path):
        print(f"SUCESSO! Arquivo de validação criado: {val_path}")
    else:
        print(f"ERRO ESTRANHO: O arquivo {val_path} não foi encontrado no disco após escrita.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["blocks", "pairs"], default="blocks")
    args = parser.parse_args()
    prepare_dataset(args.input, args.output, mode=args.mode)