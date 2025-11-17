# --- ARQUIVO ATUALIZADO: utils/epub.py ---
import os
import sys
from zipfile import ZipFile
from typing import List, Dict
from bs4 import BeautifulSoup, NavigableString
import re
import logging

logger = logging.getLogger(__name__)

def _clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def read_epub_docs(path: str) -> List[Dict]:
    """
    *** FUNÇÃO ATUALIZADA ("TUNADA") ***
    Lê o EPUB e usa uma heurística melhor para achar
    o título do capítulo (procurando por h1-h6 e classes comuns).
    """
    docs: List[Dict] = []

    # --- NOVO: Lista de classes CSS comuns para títulos ---
    COMMON_TITLE_CLASSES = [
        "cn", "ct", "chapter-title", "chaptertitle",
        "chapter-head", "chapterhead", "titulo-capitulo",
        "title", "heading"
    ]
    # --- FIM DA NOVIDADE ---

    with ZipFile(path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html"))]

        # 1. Pega o caminho COMPLETO (ex: "C:\\...\\Godsgrave.epub")
        caminho_completo = z.filename

        # 2. "Arranca" só o nome do arquivo
        nome_do_arquivo = os.path.basename(caminho_completo)

        for href in names:
            try:
                with z.open(href) as fp:
                    raw = fp.read()
            except KeyError:
                continue

            soup = BeautifulSoup(raw, "lxml")

            # --- LÓGICA DE TÍTULO "TUNADA" ---
            title = None

            # 1. Tenta tags de cabeçalho (h1...h6)
            for tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                t = soup.find(tag)
                if t and t.get_text(strip=True):
                    title = _clean_text(t.get_text())
                    break

            # 2. Se não achou, tenta tags 'p' com classes comuns
            if not title:
                for p_tag in soup.find_all("p"):
                    tag_classes = p_tag.get("class", [])
                    if any(c.lower() in COMMON_TITLE_CLASSES for c in tag_classes):
                        title_text = p_tag.get_text(strip=True)
                        if title_text:
                            title = _clean_text(title_text)
                            break

            # 3. Se AINDA não achou (fallback), pega o <title> do arquivo
            if not title:
                tt = soup.find("title")
                if tt:
                    title = _clean_text(tt.get_text())

            # 4. Se NADA funcionou, usa o nome do arquivo (último recurso)
            if not title:
                title = href.split('/')[-1]
            # --- FIM DA LÓGICA DE TÍTULO ---

            # --- Lógica de limpeza de <img> (Drop Cap) ---
            for img in soup.find_all("img"):
                alt_text = img.get("alt")
                if alt_text:
                    alt_text = alt_text.upper()
                    next_sib = img.next_sibling
                    if (next_sib and
                            isinstance(next_sib, NavigableString) and
                            next_sib.string and
                            not next_sib.string.startswith((" ", "\n", "\t"))):

                        new_text = alt_text + next_sib.string
                        next_sib.replace_with(new_text)
                        img.decompose()
                    else:
                        img.replace_with(alt_text)

            # --- Lógica de extração de texto ---
            paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            paras = [_clean_text(p) for p in paras if p]

            para_count = len(paras)
            char_count = sum(len(p) for p in paras)

            # --- FILTRO DE LIXO ---
            # Se o capítulo tiver menos de 100 caracteres
            # (provavelmente é uma capa, sumário, etc), NÓS IGNORAMOS.
            if char_count < 10:
                logger.info(f" {caminho_completo} - [epub.py] Pulando {href} (char_count={char_count} < 100)")
                continue
            # --- FIM DO FILTRO ---

            docs.append({
                "href": href,
                "title": title,  # O título "esperto"
                "para_count": para_count,
                "char_count": char_count,
                "text": "\n".join(paras)
            })

    docs.sort(key=lambda d: d["href"])
    return docs