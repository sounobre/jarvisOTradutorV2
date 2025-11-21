# --- NOVO ARQUIVO: api/translation_api.py ---
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
import logging

from services.translation_pipeline import translate_epub_book

router = APIRouter(prefix="/epub/translate", tags=["epub-translation"])
logger = logging.getLogger(__name__)


class TranslationRequest(BaseModel):
    import_id: int = Field(..., description="ID do livro que deve ser traduzido.")


@router.post("/translate-book")
async def translate_book_endpoint(
        req: TranslationRequest,
        background_tasks: BackgroundTasks,
):
    """
    Dispara o processo de tradução do livro (EN -> PT) em background.
    Usa o modelo NLLB-200.
    """
    try:
        # Agenda o trabalho pesado
        background_tasks.add_task(translate_epub_book, req.import_id)

        logger.info(f"Tradução agendada para import_id={req.import_id}.")
        return {
            "status": "translation_scheduled",
            "import_id": req.import_id,
            "message": "Tradução do livro iniciada em background. Verifique os logs para o progresso."
        }
    except Exception as e:
        logger.exception(f"Falha ao agendar tradução: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Falha ao agendar tradução.")