# --- NEW FILE: core/logging_setup.py ---
"""
Configura o logging do projeto de forma centralizada.

Ideia principal:
- Definir um handler que escreve no stdout (console)
- Colocar nível do root logger em INFO (assim .info() aparece)
- Não "matar" os loggers do Uvicorn (disable_existing_loggers=False)
"""

from logging.config import dictConfig

def setup_logging() -> None:
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,  # mantém loggers já criados (uvicorn, etc.)
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "detailed",
                "stream": "ext://sys.stdout",  # imprime no console
            },
        },
        "root": {
            # Nível do root. Se estiver WARNING, seus .info() somem.
            "level": "INFO",
            "handlers": ["console"],
        },
        "loggers": {
            # Mantém uvicorn integrado; não duplique excessos.
            "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "uvicorn.error": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "uvicorn.access": {"level": "INFO", "handlers": ["console"], "propagate": False},
        },
    })
