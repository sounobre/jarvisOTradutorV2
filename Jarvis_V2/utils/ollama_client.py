# --- NEW FILE: utils/ollama_client.py ---
"""
Cliente assíncrono para o Ollama (Qwen).
- Faz checagens (servidor + modelo) para evitar 500 confusos.
- Loga início/fim das chamadas (quem detalha conteúdo é o serviço).
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import json, re, logging
import httpx
from core.config import settings

logger = logging.getLogger("llm.ollama")
logger.setLevel(getattr(logging, settings.LOG_LEVEL_LLM.upper(), logging.INFO))

def _make_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=180.0, write=180.0, pool=None)

class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_LLM_MODEL
        self._client = httpx.AsyncClient(timeout=_make_timeout())

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str):
        return await self._client.get(f"{self.base_url}{path}")

    async def _post(self, path: str, payload: Dict[str, Any]):
        return await self._client.post(f"{self.base_url}{path}", json=payload)

    async def server_ok(self) -> bool:
        try:
            r = await self._get("/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    async def model_exists(self) -> Tuple[bool, str]:
        try:
            r = await self._post("/api/show", {"name": self.model})
            if r.status_code == 200:
                return True, ""
            try:
                data = r.json()
            except Exception:
                data = {}
            return False, data.get("error", r.text)
        except Exception as e:
            return False, str(e)

    async def chat(self, system: str, user: str) -> str:
        # checagens
        if not await self.server_ok():
            raise RuntimeError(f"Ollama OFFLINE em {self.base_url}. Rode 'ollama serve'.")
        ok, why = await self.model_exists()
        if not ok:
            raise RuntimeError(
                f"Modelo '{self.model}' indisponível: {why or 'model not found'}. "
                f"Use 'ollama pull qwen2.5:14b-instruct' (ou similar) e ajuste OLLAMA_LLM_MODEL."
            )

        # 1) chat
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # "options": {"num_ctx": 2048, "temperature": 0.2},
            }
            logger.debug("[ollama] /api/chat POST | model=%s", self.model)
            r = await self._post("/api/chat", payload)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    msg = data.get("message", {}) or {}
                    content = msg.get("content")
                    if content:
                        return content
                if "response" in data:
                    return data["response"]
        except Exception:
            pass  # cai para generate

        # 2) generate
        payload = {"model": self.model, "prompt": f"{system}\n\n{user}", "stream": False}
        logger.debug("[ollama] /api/generate POST | model=%s", self.model)
        r = await self._post("/api/generate", payload)
        if r.status_code != 200:
            try:
                err = r.json().get("error", r.text)
            except Exception:
                err = r.text
            raise RuntimeError(f"Ollama /api/generate falhou: HTTP {r.status_code} - {err}")
        data = r.json()
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        raise RuntimeError("Resposta inesperada do Ollama (sem 'response').")

    @staticmethod
    def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        try:
            start = text.index("{"); end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            for m in re.finditer(r"\{[\s\S]*?\}", text):
                try:
                    return json.loads(m.group(0))
                except Exception:
                    continue
        return None
