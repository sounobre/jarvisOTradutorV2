# --- UPDATED FILE: utils/epub.py ---
# """
# Leitor de EPUB simplificado:
# - Abre o .epub (zip), encontra XHTML/HTML e extrai título + parágrafos.
# - Retorna lista de "docs" (cada doc ~ um arquivo do livro), com métricas.
#
# Python 101:
# - `with ZipFile(path) as z:` abre o ZIP e garante fechamento automático.
# - `BeautifulSoup(html, "lxml")` faz parse do XHTML para percorrer tags.
# """
#
import sys
from zipfile import ZipFile
from typing import List, Dict
# Importamos NavigableString para checar se um nó é texto
from bs4 import BeautifulSoup, NavigableString
import re  # Necessário para o _clean_text


# (Vou adicionar uma função _clean_text de exemplo,
# já que você a usa no seu código)
def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove múltiplos espaços/quebras de linha
    return re.sub(r"\s+", " ", text).strip()


# A função _clean_text_init_empty_img não é mais necessária com a nova lógica


def read_epub_docs(path: str) -> List[Dict]:
    docs: List[Dict] = []
    with ZipFile(path, "r") as z:
        # Lista de arquivos "xhtml/html" em ordem (o TOC completo exige NCX/OPF; aqui pegamos simples)
        names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html"))]

        for href in names:
            try:
                with z.open(href) as fp:
                    raw = fp.read()
            except KeyError:
                continue

            # Parse do XHTML/HTML
            soup = BeautifulSoup(raw, "lxml")  # ok para XHTML

            # Título heurístico: h1..h3 ou <title>
            title = None
            for tag in ["h1", "h2", "h3"]:
                t = soup.find(tag)
                if t and t.get_text(strip=True):
                    title = _clean_text(t.get_text())
                    break
            if not title:
                tt = soup.find("title")
                if tt:
                    title = _clean_text(tt.get_text())

            # ================================================================
            # NOVO BLOCO DE CÓDIGO ATUALIZADO - INÍCIO
            # ================================================================
            # Pré-processa imagens que funcionam como letras (capitulares)
            for img in soup.find_all("img"):
                alt_text = img.get("alt")

                if alt_text:
                    alt_text = alt_text.upper()
                    next_sib = img.next_sibling

                    # Checa se o próximo nó existe, é um texto (NavigableString),
                    # e NÃO começa com espaço (indicando um "drop cap" colado)
                    if (next_sib and
                        isinstance(next_sib, NavigableString) and
                        next_sib.string and  # Garante que não é uma string vazia
                        not next_sib.string.startswith((" ", "\n", "\t"))):

                        # Caso "Drop Cap": <img alt="F"/>loresta...
                        # 1. Concatena: "F" + "loresta..."
                        new_text = alt_text + next_sib.string
                        # 2. Substitui o nó de texto "loresta..." pelo novo "Floresta..."
                        next_sib.replace_with(new_text)
                        # 3. Remove a tag <img> original
                        img.decompose()
                    else:
                        # Caso "Artigo": <img alt="A"/> floresta...
                        # Ou qualquer outro caso. Apenas substitui a <img>
                        img.replace_with(alt_text)
            # ================================================================
            # NOVO BLOCO DE CÓDIGO ATUALIZADO - FIM
            # ================================================================

            # Parágrafos: p + (como fallback) <div> que parece parágrafo
            # AGORA o p.get_text() vai "enxergar" o texto processado
            paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            paras = [_clean_text(p) for p in paras if p]
            # Removida a chamada para _clean_text_init_empty_img

            # Métricas
            para_count = len(paras)
            char_count = sum(len(p) for p in paras)

            docs.append({
                "href": href,
                "title": title,
                "para_count": para_count,
                "char_count": char_count,
                "text": "\n".join(paras)  # blob do doc
            })

    # Se necessário, ordenar por nome (mantém uma ordem estável)
    docs.sort(key=lambda d: d["href"])
    return docs