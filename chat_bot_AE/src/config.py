"""
Configuración central del pipeline RAG.

Todos los parámetros se pueden sobreescribir con variables de entorno
(o con el archivo .env). Así se pueden hacer experimentos de evaluación y
desplegar en la nube sin tocar el código.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(nombre: str, defecto: int) -> int:
    return int(os.environ.get(nombre, defecto))


def _float(nombre: str, defecto: float) -> float:
    return float(os.environ.get(nombre, defecto))


# --- Credenciales (NUNCA en el repositorio: .env local o secrets de la plataforma) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # se valida en src/llm.py (no al importar)

# --- Corpus ---
DOCS_DIR = BASE_DIR / os.environ.get("DOCS_DIR", "corpus")
EXTENSIONES_SOPORTADAS = {".pdf", ".docx", ".txt", ".html", ".htm"}

# Líneas de ruido que se eliminan en la ingesta (encabezados/pies repetidos en cada página).
PATRONES_RUIDO = [
    r"^.*Carrera 9 Bis No\. 62-43.*$",            # membrete del Reglamento
    r"^.*juridico@konradlorenz\.edu\.co.*$",
    r"^\s*Evaluation Copy.*$",                     # marca de agua TOGAF
    r"^.*©\s*2009-2011 The Open Group.*$",
    r"^\s*Part II: Architecture Development Method \(ADM\)\s+\d+\s*$",
    r"^\s*\d+\s+Open Group Standard \(2011\)\s*$",
    r"^\s*(Introduction\s+)?Architecture Development Cycle(\s+Introduction)?\s*$",
    r"^\s*Scoping the Architecture\s+Introduction\s*$",
    r"^\s*Introduction\s+Scoping the Architecture\s*$",
]

# --- Embeddings (locales, CPU) ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# --- Chunking ---
CHUNK_SIZE = _int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 100)

# --- Recuperación ---
TOP_K = _int("TOP_K", 5)
# Relevancia mínima (1 - distancia coseno). Por debajo de este valor un fragmento se descarta;
# si ninguno la supera, el sistema responde "no encontré información" sin llamar al LLM.
MIN_RELEVANCE = _float("MIN_RELEVANCE", 0.20)

# --- Vector store ---
INDEX_TAG = os.environ.get("INDEX_TAG", "base")          # un índice por configuración
PERSIST_DIR = BASE_DIR / "chroma" / INDEX_TAG
COLLECTION_NAME = "tutor_academico"

# --- LLM ---
# Para producción/pruebas finales cambia a "openai/gpt-oss-120b"
# Durante desarrollo "openai/gpt-oss-20b"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0)

# --- Conversación ---
MAX_HISTORY_MESSAGES = _int("MAX_HISTORY_MESSAGES", 8)   # últimos mensajes (usuario+asistente)
