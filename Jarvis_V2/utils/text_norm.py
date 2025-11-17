# --- NEW FILE: utils/text_norm.py ---
from __future__ import annotations
import re, unicodedata

_ws_re = re.compile(r"\s+", re.UNICODE)

def normalize(s: str) -> str:
    """lower + sem acento + espaços colapsados."""
    if not s:
        return ""
    # NFD + remove diacríticos
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = _ws_re.sub(" ", s).strip()
    return s

def contains(haystack: str, needle: str) -> bool:
    """substring check robusto usando normalize()."""
    h = normalize(haystack)
    n = normalize(needle)
    return n in h if n and h else False
