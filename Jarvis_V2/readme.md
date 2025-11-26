Com certeza, amigo\! Este é o documento definitivo do seu projeto.

Escrevi este **README.md** pensando em dois públicos:

1.  **Você do Futuro:** Para quando precisar reinstalar tudo daqui a 6 meses e não lembrar dos "pulos do gato" (como o `llama-cpp-python` com CUDA).
2.  **Investidores/Parceiros:** Para mostrar a robustez da engenharia de dados que você construiu.

Pode copiar e salvar na raiz do seu projeto como `README.md`.

-----

# 📚 Jarvis V2: AI High-Fantasy Book Translator

> **Uma plataforma de Engenharia de Dados e Tradução Neural Local para Alta Fantasia.**

O **Jarvis V2** é um pipeline de ponta a ponta para localização de livros (Inglês -\> Português), focado na preservação de estilo literário e estrutura de formatação. O sistema roda 100% localmente (On-Premises) utilizando uma GPU consumer (RTX 3060 12GB), eliminando custos de API e garantindo privacidade total dos dados.

-----

## 🚀 Arquitetura do Sistema

O projeto é dividido em dois grandes pilares:

### 1\. A "Fábrica de Dados" (Corpus Builder)

Responsável por criar datasets de treinamento ("Fine-Tuning") de altíssima qualidade a partir de livros já traduzidos.

  * **Segmentação Inteligente:** Utiliza **Stanza** (Stanford NLP) para quebrar textos respeitando diálogos e prosa complexa.
  * **Alinhamento Neural:** Utiliza **SentenceTransformers** (Embeddings) + **SentAlign** (Programação Dinâmica) para alinhar sentenças EN-PT, detectando pares 1-para-1, 1-para-2 e 2-para-1.
  * **Validação Automatizada:** Filtros estatísticos (Length Ratio, Number Matching) para garantir pureza no dataset final.

### 2\. O "Motor de Tradução" (Translation Engine)

Responsável pela tradução produtiva de novos livros.

  * **Preservação de EPUB:** Descompacta a estrutura do livro, isola tags HTML/Imagens usando placeholders (`[TAG_001]`) e reconstrói o arquivo final.
  * **Injeção de Glossário:** Sistema dinâmico que injeta terminologia obrigatória (ex: "High Lord" -\> "Grão-Senhor") diretamente no contexto do modelo.
  * **Inferência Local:** Roda LLMs de 7B/14B parâmetros (Qwen 2.5, Mistral) quantizados em 4-bit/5-bit via **GGUF** e **CUDA**.

-----

## 🛠️ Tech Stack

  * **Linguagem:** Python 3.11
  * **API Framework:** FastAPI (Async)
  * **Banco de Dados:** PostgreSQL + AsyncPG + SQLAlchemy 2.0
  * **IA & NLP:**
      * `llama-cpp-python` (Inferência GGUF com aceleração CUDA)
      * `sentence-transformers` (Embeddings Semânticos)
      * `stanza` (Segmentação SOTA)
      * `scikit-learn` & `numpy` (Cálculos Matriciais)
  * **Infraestrutura:** Windows 11 + WSL2 (Opcional) + CUDA Toolkit 12.1

-----

## ⚙️ Instalação e Configuração

### Pré-requisitos

1.  **NVIDIA GPU** (Recomendado: 12GB VRAM ou mais).
2.  **Drivers NVIDIA** atualizados.
3.  **CUDA Toolkit 12.1** instalado no Windows.
4.  **Python 3.11**.

### Instalação do Ambiente (.venv)

A ordem de instalação é crítica para evitar conflitos de driver no Windows.

```bash
# 1. Crie e ative o ambiente
python -m venv .venv
.\.venv\Scripts\activate

# 2. Instale o PyTorch (Versão Estável com CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Instale o Motor de Inferência (A "Bala de Prata" para Windows)
# Nota: Use --no-cache-dir para forçar a versão correta com aceleração de GPU
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 4. Instale o Restante das Dependências
pip install fastapi uvicorn sqlalchemy asyncpg pandas openpyxl stanza sentence-transformers scikit-learn python-multipart
```

-----

## 🕹️ Guia de Uso (Painel de Controle)

O sistema é controlado via API REST (documentada via Swagger/OpenAPI).

### Fluxo A: Criar Corpus (Treinamento)

Este fluxo transforma um par de EPUBs (Original + Tradução Oficial) em um arquivo `.tsv` para treinar a IA.

1.  **Importar Livros:**

      * `POST /epub/import`: Envie os dois arquivos `.epub` (EN e PT).
      * *Resultado:* Cria os registros no banco. Retorna `import_id`.

2.  **Fase 1: Mapeamento Macro (Agendamento)**

      * `POST /epub/corpus_v2/1-schedule-macro-map`
      * *O que faz:* Compara o conteúdo dos capítulos para saber que "Chapter 1" é par de "CAPÍTULO 1".

3.  **Fase 2: Alinhamento Fino (SentAlign)**

      * `POST /epub/corpus_v2/2c-align-all-pending`
      * *O que faz:* Processa o livro inteiro. Usa **Stanza** para quebrar o texto e **SentAlign** para parear as frases. Roda na GPU.
      * *Monitoramento:* Use `GET /epub/corpus_v2/get-status` para ver o progresso.

4.  **Fase 3: Validação de Qualidade**

      * `POST /epub/corpus_v2/3-validate-corpus`
      * *O que faz:* Aplica métricas (Length Ratio, Number Mismatch) para marcar pares suspeitos.

5.  **Fase 4: Exportação Premium**

      * `POST /epub/corpus_v2/4-export-corpus`
      * *O que faz:* Gera o arquivo final `corpus_premium.tsv` contendo apenas os pares validados e de alta qualidade.

-----

### Fluxo B: Traduzir Livro (Produção)

Este fluxo traduz um livro inédito usando o modelo local.

1.  **Importar Livro:**

      * `POST /epub/import-single`: Envie apenas o `.epub` em Inglês.

2.  **Configurar Glossário (Opcional):**

      * `POST /epub/glossary/add-terms`: Envie JSON com termos fixos (ex: `{"High Lord": "Grão-Senhor"}`).

3.  **Traduzir:**

      * `POST /epub/translate/translate-book`
      * Payload: `{"import_id": 123}`.
      * *O que acontece:*
          * O sistema carrega o modelo **Qwen 2.5 7B Instruct (GGUF)** na VRAM.
          * Lê o EPUB original e protege tags HTML.
          * Fatia o texto (Chunking Inteligente) e traduz usando Prompt de Sistema especializado.
          * Reconstrói o arquivo `.epub` traduzido na pasta `data/translated/`.

-----

## 📊 Estrutura do Banco de Dados

  * `tm_import`: Metadados dos arquivos.
  * `tm_chapter_index`: Índice estrutural dos capítulos (HREF, Título).
  * `tm_chapter_text`: Conteúdo bruto dos capítulos.
  * `tm_win_mapping`: O mapa de relacionamento entre capítulos (EN \<-\> PT).
  * `tm_aligned_sentences`: O produto final (Pares de sentenças com score de similaridade).
  * `tm_translation_log`: Telemetria completa de cada tradução realizada (Prompt usado, Tempo, Temperatura).
  * `tm_glossary`: Dicionário de termos forçados por livro.

-----

## 🐛 Troubleshooting Comum

**Erro:** `WinError 2: O sistema não pode encontrar o arquivo especificado`

  * **Causa:** Caminho do Python do `SentAlign` incorreto.
  * **Solução:** Verifique a constante `SENTALIGN_PYTHON_PATH` em `services/corpus_builder_service.py`.

**Erro:** `.to is not supported for 4-bit models`

  * **Causa:** Conflito entre `accelerate` e `bitsandbytes` ao mover tensores.
  * **Solução:** Estamos usando `llama-cpp-python` (GGUF), que não sofre desse problema. Certifique-se de ter removido o código antigo que usava `AutoModelForCausalLM` do transformers puro.

**Erro:** GPU Usage 0% / CPU 100%

  * **Causa:** `llama-cpp-python` instalado sem suporte a CUDA.
  * **Solução:** Reinstale usando a flag `--force-reinstall --no-cache-dir` e a URL do repositório `abetlen` com `cu121`.

-----

> **Jarvis V2** - *Construindo a ponte entre mundos, uma sentença de cada vez.*