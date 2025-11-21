# --- ARQUIVO: scripts/prepare_dataset.py ---
import csv
import json
import os
import argparse

# O System Prompt que o modelo vai aprender a obedecer
SYSTEM_PROMPT = """You are a professional literary translator specializing in High Fantasy novels. 
Translate the text below from English to Portuguese (Brazil).
Guidelines:
1. Maintain the tone, style, and atmosphere of the original text.
2. Keep all formatting tags (like [TAG_0000]) EXACTLY where they are.
3. Translate idioms and cultural references naturally for a Brazilian audience.
4. Do NOT provide explanations, only the translated text."""


def convert_tsv_to_jsonl(input_file, output_file):
    print(f"Lendo {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        reader = csv.reader(f_in, delimiter='\t')
        count = 0

        for row in reader:
            if len(row) < 2: continue

            source_text = row[0]
            target_text = row[1]

            # Monta a estrutura de chat
            data = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": source_text},
                    {"role": "assistant", "content": target_text}
                ]
            }

            # Salva uma linha JSON
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1

    print(f"Sucesso! {count} pares convertidos para {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Arquivo .tsv de entrada")
    parser.add_argument("--output", required=True, help="Arquivo .jsonl de saída")
    args = parser.parse_args()

    convert_tsv_to_jsonl(args.input, args.output)